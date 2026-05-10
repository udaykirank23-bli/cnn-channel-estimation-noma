% =========================================================================
% DIRECT CHANNEL DATASET GENERATION — V=3000, U=10
% =========================================================================
% Paper: "Deep Channel Learning for Large Intelligent Surfaces Aided
%         mm-Wave Massive MIMO Systems", Elbir et al., 2020
%
% KEY CHANGE FROM PREVIOUS VERSION:
%   Before: V=300,  U=100  → 300 unique channels, each seen 100 times
%           Problem: model MEMORIZED the 300 scenarios instead of learning
%           Symptom: val NMSE = -19 dB but new test data gives only -4 dB
%
%   Now:    V=3000, U=10   → 3000 unique channels, each seen only 10 times
%           T = 3000 × 10 × 8 = 240,000 (same total samples)
%           Model is forced to GENERALIZE across channel diversity
%
% After training with this data, a fresh rng test set should give
% NMSE close to the val NMSE — confirming true generalization.
%
% ALSO GENERATES: held-out test set from scenarios V=3001 to 3100
%   (continues same rng stream — no overlap with training)
% =========================================================================

clear all; clc;
rng(4096);   % same seed as before — keeps results reproducible

if gpuDeviceCount("available") > 0
    g = gpuDevice;
    fprintf('🚀 GPU ACTIVATED: %s (%.1f GB)\n', g.Name, g.TotalMemory/1e9);
else
    fprintf('⚠️ NO GPU DETECTED. Training will run on CPU and be much slower.\n');
end

% =========================================================================
% SYSTEM PARAMETERS
% =========================================================================
M  = 64;      % BS antennas
K  = 8;       % users
Np = 10;      % channel paths
fc = 60e9;    % 60 GHz
fs = 4e9;     % bandwidth

sqrtM = sqrt(M);   % = 8

% =========================================================================
% DATASET SIZE — key change
% =========================================================================
U = 6;     % noise realisations per channel scenario  (was 100)
V = 10000;   % channel scenarios                        (was 300)
% T = V * U * K = 3000 * 10 * 8 = 240,000  (unchanged)

SNR_list  = [-10, 0, 10, 20, 30,40,50];   % dB
%SNR_list = [10,20,30];
SNRh_list = [30, 40];       % dB label noise

T = V * U * K;

fprintf('=========================================\n');
fprintf('  DIRECT CHANNEL DATASET GENERATION\n');
fprintf('  V=5000, U=8  (generalization fix)\n');
fprintf('=========================================\n');
fprintf('M=%d, K=%d, Np=%d, fc=%.0f GHz\n', M, K, Np, fc/1e9);
fprintf('V=%d scenarios x U=%d realisations x K=%d users\n', V, U, K);
fprintf('Total T = %d samples\n', T);
fprintf('=========================================\n\n');

% =========================================================================
% PILOT MATRIX
% =========================================================================
X = eye(M);

% =========================================================================
% PRE-ALLOCATE TRAINING DATASET
% =========================================================================
X_dc = zeros(sqrtM, sqrtM, 2, T, 'single');
Y_dc = zeros(T, 2*M, 'single');

t = 1;
tic;

% =========================================================================
% ALGORITHM 1 — THREE NESTED LOOPS (identical structure, V/U changed)
% =========================================================================
for v = 1:V

    if mod(v, 300) == 0
        fprintf('  Scenario %4d / %d | samples: %7d | %.1f sec\n', ...
                 v, V, t-1, toc);
    end

    % Generate K fresh channels for scenario v
    [h_dc_v, ~, ~, ~, ~, ~, ~] = generate_channel_H_LIS(1, M, Np, fs, fc, 1, K);

    for u = 1:U

        snr_pilot = SNR_list(mod(u-1,  length(SNR_list))  + 1);
        snrh      = SNRh_list(mod(u-1, length(SNRh_list)) + 1);

        for k = 1:K

            h_k = (1/sqrt(10^0.3)).* h_dc_v(:, 1, k);   % [M x 1]
            %h_k = h_dc_v(:, 1, k);   

            % ---- Label: add small noise (Algorithm 1 line 6) ----
            avg_power   = mean(abs(h_k).^2);
            sigma_h     = sqrt(avg_power) / (10^(snrh/20));
            label_noise = sigma_h/sqrt(2) * (randn(M,1) + 1i*randn(M,1));
            h_k_label   = h_k + label_noise;

            Y_dc(t, :) = single([real(h_k_label); imag(h_k_label)]');

            % ---- Input: simulate eq(4) ----
            y_noisy = awgn(h_k' * X, snr_pilot, 'measured');
            y_2D    = reshape(y_noisy, [sqrtM, sqrtM]);

            X_dc(:,:,1,t) = single(real(y_2D));
            X_dc(:,:,2,t) = single(imag(y_2D));

            t = t + 1;

        end   % k
    end   % u
end   % v

fprintf('\nTraining data done: %d samples in %.1f sec\n\n', t-1, toc);

% =========================================================================
% TRAIN / VAL SPLIT  (70% / 30% — paper Section IV)
% =========================================================================
N_total = t - 1;
idx_all = randperm(N_total);

N_train = floor(0.70 * N_total);   % 168,000
N_val   = N_total - N_train;       %  72,000

idx_train = idx_all(1        : N_train);
idx_val   = idx_all(N_train+1: end);

X_train = X_dc(:,:,:, idx_train);   Y_train = Y_dc(idx_train, :);
X_val   = X_dc(:,:,:, idx_val);     Y_val   = Y_dc(idx_val,   :);

fprintf('Train: %d samples | Val: %d samples\n', size(Y_train,1), size(Y_val,1));

% =========================================================================
% GENERATE HELD-OUT TEST SET  (V=3001 to 3100, same rng stream)
% =========================================================================
% CRITICAL: do NOT call rng() again here.
% The rng state continues from where the training loop ended.
% This guarantees test scenarios 3001-3100 are:
%   (a) statistically identical to training scenarios
%   (b) never seen during training (no overlap)
% =========================================================================
fprintf('\nGenerating held-out test set (V=3001 to 3100)...\n');

V_test    = 100;
J_test    = 100;
T_test    = V_test * J_test * K;   % = 80,000 per SNR

n_snr     = length(SNR_list);
X_te_cell = cell(1, n_snr);
Y_te_cell = cell(1, n_snr);
for s = 1:n_snr
    X_te_cell{s} = zeros(sqrtM, sqrtM, 2, T_test, 'single');
    Y_te_cell{s} = zeros(T_test, 2*M, 'single');
end

tic;
for s = 1:n_snr

    snr_pilot = SNR_list(s);
    tt = 1;
    fprintf('  Test SNR = %d dB ...', snr_pilot);

    for v = 1:V_test

        % fresh channel — rng continues from training stream
        [h_dc_te, ~, ~, ~, ~, ~, ~] = generate_channel_H_LIS(1, M, Np, fs, fc, 1, K);

        for j = 1:J_test
            for k = 1:K

                h_k     = h_dc_te(:, 1, k);

                % Clean label — no label noise at test time
                Y_te_cell{s}(tt, :) = single([real(h_k); imag(h_k)]');

                % Input
                y_noisy = awgn(h_k' * eye(M), snr_pilot, 'measured');
                y_2D    = reshape(y_noisy, [sqrtM, sqrtM]);
                X_te_cell{s}(:,:,1,tt) = single(real(y_2D));
                X_te_cell{s}(:,:,2,tt) = single(imag(y_2D));

                tt = tt + 1;
            end
        end
    end
    fprintf(' done (%d samples)\n', tt-1);
end

X_test_snr10 = X_te_cell{1};   Y_test_snr10 = Y_te_cell{1};
X_test_snr20 = X_te_cell{2};   Y_test_snr20 = Y_te_cell{2};
X_test_snr30 = X_te_cell{3};   Y_test_snr30 = Y_te_cell{3};

fprintf('Test set done in %.1f sec\n\n', toc);

% =========================================================================
% SAVE
% =========================================================================
save_path = 'DC_dataset_V5000.mat';
save(save_path, ...
    'X_train', 'Y_train', ...
    'X_val',   'Y_val',   ...
    'X_test_snr10', 'Y_test_snr10', ...
    'X_test_snr20', 'Y_test_snr20', ...
    'X_test_snr30', 'Y_test_snr30', ...
    'M', 'K', 'Np', 'fc', ...
    'V', 'U', 'T',  ...
    'SNR_list','SNRh_list', ...
    '-v7.3');

fprintf('Saved: %s\n', save_path);
fprintf('Train input:  [8 x 8 x 2 x %d]\n', size(X_train, 4));
fprintf('Train labels: [%d x 128]\n', size(Y_train, 1));
