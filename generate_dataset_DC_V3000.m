% =========================================================================
% TEST DATASET GENERATION — Full SNR Range (-10 to 50 dB)
% =========================================================================
% Paper: "Deep Channel Learning for Large Intelligent Surfaces Aided
%         mm-Wave Massive MIMO Systems", Elbir et al., 2020
%
% PURPOSE:
%   Generate test dataset covering SNR = -10 to 50 dB in 5 dB steps
%   to reproduce paper Fig. 3 exactly.
%
% KEY DESIGN DECISIONS:
%   (1) rng continues from training stream — NO rng() call here
%       Run this script immediately after generate_dataset_DC_V3000.m
%       in the SAME MATLAB session so rng state is past V=3000.
%       This guarantees test channels are statistically identical
%       to training but never seen during training.
%
%   (2) Labels are CLEAN (no label noise)
%       Training used noisy labels for regularization.
%       Test uses clean h_k as ground truth — this is what
%       paper Fig. 3 measures NMSE against.
%
%   (3) J=100 Monte Carlo trials per SNR level (paper Section IV)
%       Total per SNR = J * K = 100 * 8 = 800 samples
%
% SYSTEM PARAMETERS — must match training exactly:
%   M=64, K=8, Np=10, fc=60GHz, fs=4GHz
%
% OUTPUT: DC_test_fig3.mat
%   One X/Y pair per SNR level, named X_test_snr_<value>
%   e.g. X_test_snr_m10 (SNR=-10), X_test_snr_0, X_test_snr_10 ...
%   (MATLAB variable names cannot start with '-' so negatives use 'm')
% =========================================================================

% =========================================================================
% SYSTEM PARAMETERS  (must match training)
% =========================================================================

if gpuDeviceCount("available") > 0
    g = gpuDevice;
    fprintf('🚀 GPU ACTIVATED: %s (%.1f GB)\n', g.Name, g.TotalMemory/1e9);
else
    fprintf('⚠️ NO GPU DETECTED. Training will run on CPU and be much slower.\n');
end

M  = 64;
K  = 8;
Np = 10;
fc = 60e9;
fs = 4e9;

sqrtM = sqrt(M);   % = 8

X_pilot = eye(M);  % identity pilot matrix

% =========================================================================
% TEST SNR RANGE — matches paper Fig. 3 x-axis exactly
% =========================================================================
SNR_test_list = -10:10:50;   % [-10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
n_snr         = length(SNR_test_list);   % = 13

J      = 100;          % Monte Carlo trials per SNR (paper Section IV)
T_snr  = J * K;        % = 800 samples per SNR level

fprintf('=========================================\n');
fprintf('  TEST DATASET — FULL SNR RANGE\n');
fprintf('  Reproduces paper Fig. 3\n');
fprintf('=========================================\n');
fprintf('SNR range: %d to %d dB (step 5 dB)\n', ...
         SNR_test_list(1), SNR_test_list(end));
fprintf('SNR levels: %d\n', n_snr);
fprintf('J=%d trials x K=%d users = %d samples/SNR\n', J, K, T_snr);
fprintf('Total samples: %d\n', T_snr * n_snr);
fprintf('=========================================\n\n');
fprintf('NOTE: Run this in the SAME MATLAB session as\n');
fprintf('generate_dataset_DC_V3000.m to continue rng stream.\n\n');

% =========================================================================
% PRE-ALLOCATE — one cell per SNR level
% =========================================================================
X_cell = cell(1, n_snr);
Y_cell = cell(1, n_snr);
for s = 1:n_snr
    % Input:  [sqrtM x sqrtM x 2 x T_snr] = [8 x 8 x 2 x 800]
    % Label:  [T_snr x 2*M]               = [800 x 128]
    X_cell{s} = zeros(sqrtM, sqrtM, 2, T_snr, 'single');
    Y_cell{s} = zeros(T_snr, 2*M, 'single');
end

% =========================================================================
% MAIN LOOP
% =========================================================================
tic;

for s = 1:n_snr

    snr_pilot = SNR_test_list(s);
    t = 1;

    fprintf('SNR = %+3d dB ... ', snr_pilot);

    for j = 1:J   % J=100 Monte Carlo trials

        % Generate K fresh channels — rng continues from training stream
        [h_dc, ~, ~, ~, ~, ~, ~] = generate_channel_H_LIS(1, M, Np, fs, fc, 1, K);

        for k = 1:K

            h_k = (1/sqrt(10^0.3)).* h_dc(:, 1, k);  % [M x 1] clean complex channel
            %h_k = h_dc(:, 1, k);

            % ----------------------------------------------------------
            % LABEL — clean channel (no label noise at test time)
            % z_DC = [Re{h_k}; Im{h_k}]  as [1 x 2M] row
            % ----------------------------------------------------------
            Y_cell{s}(t, :) = single([real(h_k); imag(h_k)]');

            % ----------------------------------------------------------
            % INPUT — eq(4): y_D = h_k^H * X + noise
            % X = I_M  =>  y_D = h_k' + noise  [1 x M]
            % Reshape to [8 x 8], store Re and Im separately
            % ----------------------------------------------------------
            y_noisy = awgn(h_k' * X_pilot, snr_pilot, 'measured');
            y_2D    = reshape(y_noisy, [sqrtM, sqrtM]);

            X_cell{s}(:,:,1,t) = single(real(y_2D));
            X_cell{s}(:,:,2,t) = single(imag(y_2D));

            t = t + 1;

        end   % k
    end   % j

    fprintf('done (%d samples)\n', t-1);

end   % s

fprintf('\nTotal time: %.1f sec\n\n', toc);

% =========================================================================
% SAVE — one named variable per SNR level
% =========================================================================
% Variable naming: negative SNR uses 'm' prefix (MATLAB limitation)
%   SNR=-10 -> X_test_snr_m10 / Y_test_snr_m10
%   SNR=  0 -> X_test_snr_0   / Y_test_snr_0
%   SNR= 10 -> X_test_snr_10  / Y_test_snr_10
%   SNR= 50 -> X_test_snr_50  / Y_test_snr_50

save_vars = {};   % will collect variable names for save()

for s = 1:n_snr
    snr_val = SNR_test_list(s);

    % Build variable name
    if snr_val < 0
        vname_X = sprintf('X_test_snr_m%d', abs(snr_val));
        vname_Y = sprintf('Y_test_snr_m%d', abs(snr_val));
    else
        vname_X = sprintf('X_test_snr_%d',  snr_val);
        vname_Y = sprintf('Y_test_snr_%d',  snr_val);
    end

    % Assign to workspace dynamically
    eval(sprintf('%s = X_cell{%d};', vname_X, s));
    eval(sprintf('%s = Y_cell{%d};', vname_Y, s));

    save_vars{end+1} = vname_X;
    save_vars{end+1} = vname_Y;
end

% Save metadata variables too
SNR_test_list_save = SNR_test_list;
save_vars{end+1} = 'SNR_test_list_save';
save_vars{end+1} = 'M';
save_vars{end+1} = 'K';
save_vars{end+1} = 'Np';
save_vars{end+1} = 'fc';
save_vars{end+1} = 'J';
save_vars{end+1} = 'T_snr';

save('DC_test5000_fig3.mat', save_vars{:}, '-v7.3');

fprintf('Saved: DC_test_fig3.mat\n');
fprintf('Variables saved:\n');
for i = 1:2:length(save_vars)-6   % print X/Y pairs only
    fprintf('  %s  /  %s\n', save_vars{i}, save_vars{i+1});
end

% =========================================================================
% PYTHON LOADING GUIDE
% =========================================================================
fprintf('\n--- Python loading ---\n');
fprintf('import h5py, numpy as np\n\n');
fprintf('f = h5py.File("DC_test_fig3.mat", "r")\n');
fprintf('SNR_list = list(range(-10, 51, 5))  # -10 to 50 step 5\n\n');
fprintf('X_test_dict = {}\n');
fprintf('Y_test_dict = {}\n');
fprintf('for snr in SNR_list:\n');
fprintf('    key = f"snr_m{abs(snr)}" if snr < 0 else f"snr_{snr}"\n');
fprintf('    mat_key_X = f"X_test_{key}"\n');
fprintf('    mat_key_Y = f"Y_test_{key}"\n');
fprintf('    X_test_dict[snr] = np.array(f[mat_key_X]).transpose(3,0,1,2)  # (N,2,8,8)\n');
fprintf('    Y_test_dict[snr] = np.array(f[mat_key_Y]).T                    # (N,128)\n');
