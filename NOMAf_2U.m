%% Clear
clc; clear; close all;

%% Load datasets
data1 = load('withoutpathlosspred.mat');   % hd1 - stronger user (without path loss)
data2 = load('pathlosspredicted.mat');     % hd2 - weaker user (with path loss)

%% Load hd1 (stronger user - without path loss, SNR = 0 dB)
h_cnn_hd1  = data1.h_cnn_snr_0;
h_ls_hd1   = data1.h_ls_snr_0;
h_true_hd1 = data1.h_true_snr_0;

%% Load hd2 (weaker user - with path loss, SNR = -10 dB)
h_cnn_hd2  = data2.h_cnn_snr_m10;
h_ls_hd2   = data2.h_ls_snr_m10;
h_true_hd2 = data2.h_true_snr_m10;

%% Reconstruct complex channels - hd1 (stronger)
hd1_cnn  = h_cnn_hd1(:,  1:64) + 1j*h_cnn_hd1(:,  65:128);
hd1_ls   = h_ls_hd1(:,   1:64) + 1j*h_ls_hd1(:,   65:128);
hd1_true = h_true_hd1(:, 1:64) + 1j*h_true_hd1(:, 65:128);

%% Reconstruct complex channels - hd2 (weaker)
hd2_cnn  = h_cnn_hd2(:,  1:64) + 1j*h_cnn_hd2(:,  65:128);
hd2_ls   = h_ls_hd2(:,   1:64) + 1j*h_ls_hd2(:,   65:128);
hd2_true = h_true_hd2(:, 1:64) + 1j*h_true_hd2(:, 65:128);

%% Compute channel gains
g1_cnn  = abs(hd1_cnn).^2;
g1_ls   = abs(hd1_ls).^2;
g1_true = abs(hd1_true).^2;

g2_cnn  = abs(hd2_cnn).^2;
g2_ls   = abs(hd2_ls).^2;
g2_true = abs(hd2_true).^2;

%% Sanity check - hd1 should be stronger than hd2
fprintf('Mean gain hd1 (stronger, 0 dB):   %.6f\n', mean(g1_true(:)));
fprintf('Mean gain hd2 (weaker,  -10 dB):  %.6f\n', mean(g2_true(:)));
if mean(g1_true(:)) > mean(g2_true(:))
    fprintf('OK: hd1 is stronger than hd2\n\n');
else
    fprintf('WARNING: hd1 is NOT stronger than hd2 — check user assignment!\n\n');
end

%% Normalize gains to True channel power (fair comparison)
g1_cnn = g1_cnn * (mean(g1_true(:)) / mean(g1_cnn(:)));
g1_ls  = g1_ls  * (mean(g1_true(:)) / mean(g1_ls(:)));
g2_cnn = g2_cnn * (mean(g2_true(:)) / mean(g2_cnn(:)));
g2_ls  = g2_ls  * (mean(g2_true(:)) / mean(g2_ls(:)));

%% MSE check (verify CNN is better than LS)
mse_cnn_hd1 = mean((g1_cnn(:) - g1_true(:)).^2);
mse_ls_hd1  = mean((g1_ls(:)  - g1_true(:)).^2);
mse_cnn_hd2 = mean((g2_cnn(:) - g2_true(:)).^2);
mse_ls_hd2  = mean((g2_ls(:)  - g2_true(:)).^2);

fprintf('hd1 | Gain MSE CNN: %.6f | Gain MSE LS: %.6f', mse_cnn_hd1, mse_ls_hd1);
if mse_cnn_hd1 < mse_ls_hd1
    fprintf(' -> CNN BETTER\n');
else
    fprintf(' -> WARNING: CNN WORSE\n');
end

fprintf('hd2 | Gain MSE CNN: %.6f | Gain MSE LS: %.6f', mse_cnn_hd2, mse_ls_hd2);
if mse_cnn_hd2 < mse_ls_hd2
    fprintf(' -> CNN BETTER\n');
else
    fprintf(' -> WARNING: CNN WORSE\n');
end

%% NOMA Power allocation
% Stronger user (hd1) gets less power
% Weaker user   (hd2) gets more power
a1 = 0.2;   % stronger user (hd1)
a2 = 0.8;   % weaker user   (hd2)

%% Normalized noise
BW  = 1;
Fdb = -90;
no  = (10.^((Fdb)/10)) * BW;


%% SNR range
Pt = -30:5:10;
pt = 10.^(Pt/10);
p  = length(Pt);

%% Pre-allocate
Rsum_cnn  = zeros(1, p);
Rsum_ls   = zeros(1, p);
Rsum_true = zeros(1, p);

%% Sum-rate computation
for u = 1:p

    % User 2 (weaker, hd2): NO SIC — User 1 signal acts as interference
    gamma_2_cnn  = (a2 * pt(u) * g2_cnn)  ./ (a1 * pt(u) * g2_cnn  + no);
    gamma_2_ls   = (a2 * pt(u) * g2_ls)   ./ (a1 * pt(u) * g2_ls   + no);
    gamma_2_true = (a2 * pt(u) * g2_true) ./ (a1 * pt(u) * g2_true + no);

    rate_2_cnn  = log2(1 + gamma_2_cnn);
    rate_2_ls   = log2(1 + gamma_2_ls);
    rate_2_true = log2(1 + gamma_2_true);

    % User 1 (stronger, hd1): SIC applied — noise only remains
    gamma_1_cnn  = (a1 * pt(u) * g1_cnn)  ./ no;
    gamma_1_ls   = (a1 * pt(u) * g1_ls)   ./ no;
    gamma_1_true = (a1 * pt(u) * g1_true) ./ no;

    rate_1_cnn  = log2(1 + gamma_1_cnn);
    rate_1_ls   = log2(1 + gamma_1_ls);
    rate_1_true = log2(1 + gamma_1_true);

    % Sum rate = User 1 rate + User 2 rate
    Rsum_cnn(u)  = mean(rate_1_cnn,  'all') + mean(rate_2_cnn,  'all');
    Rsum_ls(u)   = mean(rate_1_ls,   'all') + mean(rate_2_ls,   'all');
    Rsum_true(u) = mean(rate_1_true, 'all') + mean(rate_2_true, 'all');
     Rsum_OMA_true(u) =1/2* mean(rate_1_true, 'all') +1/2* mean(rate_2_true, 'all');


end

%% Plot
figure;
hold on;
plot(Pt, Rsum_true, 'b:',  'LineWidth', 2);
plot(Pt, Rsum_cnn,  'k-',  'LineWidth', 2);
plot(Pt, Rsum_ls,   'r--', 'LineWidth', 2);
plot(Pt,  Rsum_OMA_true,   'black', 'LineWidth', 2);

hold off;
grid on;
xlabel('SNR (dB)');
ylabel('Achievable Sum Rate (bps/Hz)');
title('NOMA Sum Rate — hd1: Stronger (No Path Loss, 0 dB) | hd2: Weaker (With Path Loss, -10 dB)');
legend('True Channel', 'CNN Estimation', 'Least Squares', 'Location', 'northwest');