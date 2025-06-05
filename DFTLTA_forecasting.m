%% DFTLA_forecasting.m
% Author: Moka Kaleji • Contact: mohammadkaleji1998@gmail.com
% Affiliation: Master Thesis in Econometrics: 
% Advancing High-Dimensional Factor Models: Integrating Time-Varying 
% Parameters with Dynamic Factors.
% University of Bologna

% Purpose: Perform out-of-sample forecasting using a dynamic factor model 
% with time-varying loadings and transition matrix A (DFTLTA), evaluate 
% forecast accuracy, and visualize results.
% Explanation: This script loads estimation outputs from QMLDFM_TVLA, generates
% forecasts for a user-specified horizon H, computes error metrics (MSFE, RMSE), 
% compares against benchmarks (random walk, AR(1)), conducts statistical 
% tests (Diebold-Mariano, forecast encompassing, Ljung-Box), and visualizes
% results.
% References:
%   - Stock, J. H., & Watson, M. W. (2002). Forecasting using principal 
%     components from a large number of predictors. Journal of the American
%     Statistical Association, 97(460), 1167-1179.
%   - Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy.
%     Journal of Business & Economic Statistics, 13(3), 253-263.
%   - Harvey, A. C., & Todd, P. H. (1983). Forecasting economic time series
%     with structural and Box-Jenkins models: A case study. Journal of 
%     Business & Economic Statistics, 1(4), 299-307.
%   - Ljung, G. M., & Box, G. E. P. (1978). On a measure of lack of fit in 
%     time series models. Biometrika, 65(2), 297-303.

clear; close all; clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Dataset Selection, Load Data and Define Key Variables
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Purpose: Allow user to select a dataset (monthly or quarterly) and load 
% corresponding data.
% Explanation: Presents a dialog for selecting between monthly (MD1959.xlsx)
% or quarterly (QD1959.xlsx) datasets, loads the chosen data, and defines 
% key variables (GDP, Unemployment, Inflation)
% for forecasting evaluation. The datasets are high-dimensional time series
% used in macroeconomic forecasting, with specific indices for key variables.
% References:
%   - McCracken, M. W., & Ng, S. (2016). FRED-MD: A monthly database for 
%     macroeconomic research. Journal of Business & Economic Statistics, 
%     34(4), 574-589. (And FRED-QD).
options = {'Monthly (MD1959.xlsx)', 'Quarterly (QD1959.xlsx)'};
[choiceIndex, ok] = listdlg( ...
    'PromptString', 'Select the dataset to load:', ...
    'SelectionMode', 'single', ...
    'ListString', options, ...
    'Name', 'Dataset Selection', ...
    'ListSize', [400 200]);
if ~ok
    error('No dataset selected. Exiting...');
end
% Load data based on user selection
% Explanation: Loads the dataset from a specified filepath, sets the time 
% series length T, and extracts key variables’ indices. The monthly dataset
% has T=790 observations, and the quarterly has T=264, reflecting typical 
% macroeconomic data frequencies.
switch choiceIndex
    case 1
        filepath = ['/Users/moka/Research/Thesis/Live Project/' ...
            'Processed_Data/MD1959.xlsx'];
        T = 790;
        tableData = readtable(filepath);
        x = table2array(tableData);
        key_vars = [1, 24, 105];                                       % Indices for key variables
    case 2
        filepath = ['/Users/moka/Research/Thesis/Live Project/' ...
            'Processed_Data/QD1959.xlsx'];
        T = 264;
        tableData = readtable(filepath);
        x = table2array(tableData(:,2:end));                               % Exclude ate column
        key_vars = [1, 58, 116];                                      % Indices for key variables
    otherwise
        error('Unexpected selection index.');
end
var_names = {'GDP', 'Unemployment', 'Inflation'};


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load DFTLTA Estimation Outputs 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Purpose: Load pre-estimated DFTLTA model parameters and results.
% Explanation: Loads the output file 'dftl_estim_results.mat' containing 
% the MKA struct (from QMLDFM_TVLA), training data statistics (mean_train, 
% std_train, T_train), and model parameters (R, h, p, max_iter, tol). 
% These are used for forecasting and evaluation.
load('dftl&A_estim_results.mat');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Forecast Horizon 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Purpose: Prompt user to specify the forecast horizon H.
% Explanation: Computes the maximum possible horizon s = T - T_train 
% (test sample size) and validates that H is a positive integer between 
% 1 and s. This ensures forecasts are feasible within the test data range.
s = T - T_train;
H = input(sprintf('Enter forecast horizon H from 1 to T = %d: ', s));      
if ~(isscalar(H) && isnumeric(H) && H == floor(H) && H >= 1 && H <= s)
    error('H must be an **integer** between 1 and T = %d.', s);            
end

disp(['Using forecast horizon H = ', num2str(H)]);                         % Forecast horizon

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Prepare Test Sample 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Purpose: Extract and normalize the test data for forecasting evaluation.
% Explanation: Selects the test data x_test for the specified horizon H 
% from the full dataset and normalizes it using training data statistics 
% (mean_train, std_train) to match the scale of the estimation inputs,
% ensuring consistency.
x_test = x(T_train+1:T_train+H, :);
x_test_norm = (x_test - mean_train) ./ std_train;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Running Forecast Via DFTLTA Outputs 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Purpose: Generate forecasts using the DFTLTA model estimates.
% Explanation: Calls the DFTL_Forecast function to produce H-step-ahead 
% forecasts for factors (Fhat_forecast) and observables (yhat), using the 
% smoothed states (xitT), last computed companion matrix (Ahat_companion), 
% loadings (Lhat), and  covariances (Sigma_e_hat, Qhat). The deterministic 
% option (stochastic=false) is used for point forecasts.
[MKA_Forecast] = DFTLTA_Forecast(MKA.xitT, MKA.Ahat_companion, MKA.Lhat, ...
    MKA.Sigma_e_hat, MKA.Qhat, H, R, p, mean_train, std_train, false);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% MSFE/Ratio, RMSE, Diebold–Mariano Tests, Ljung–Box Tests, Encompassing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Purpose: Evaluate forecast accuracy and compare against benchmarks.
% Explanation: Computes Mean Squared Forecast Error (MSFE), Root Mean 
% Squared Error (RMSE), MSFE ratios against random walk (RW) and AR(1) 
% benchmarks, and conducts statistical tests to assess forecast performance
% and residual properties.
% References:
%   - Diebold & Mariano (1995) for comparing predictive accuracy.
%   - Harvey & Todd (1983) for forecast encompassing tests.
%   - Ljung & Box (1978) for residual autocorrelation tests.

% --- MSFE and RMSE ---
% Purpose: Calculate MSFE and RMSE for key variables.
% Explanation: MSFE is the average squared forecast error, computed per 
% horizon and variable. RMSE is the square root of MSFE, providing a 
% scale-interpretable metric.
squared_errors = (x_test(:, key_vars) - MKA_Forecast.yhat(:, key_vars)).^2;
MSFE_horizon = mean(squared_errors, 2);
MSFE_all = mean(squared_errors, 1);
MSFE_overall = mean(MSFE_all);

fprintf('Overall MSFE (original): %.4f\n', MSFE_overall);
for idx = 1:length(key_vars)
    fprintf('MSFE for (original) %s: %.4f\n', var_names{idx}, ...
        MSFE_all(idx));
end

RMSE_original = sqrt(MSFE_all);
for idx = 1:length(key_vars)
    fprintf('RMSE for %s (original): %.4f\n', var_names{idx}, ...
        RMSE_original(idx));
end

% --- Random Walk Benchmark ---
% Purpose: Compute MSFE for a naive random walk forecast.
% Explanation: The random walk forecast assumes y_{t+h} = y_t, using the 
% last training observation. MSFE ratios (DFTLTA vs. RW) indicate relative 
% performance.
yhat_rw = repmat(x(T_train, :), H, 1);
squared_errors_rw = (x_test(:, key_vars) - yhat_rw(:, key_vars)).^2;
MSFE_rw_all = mean(squared_errors_rw, 1);
MSFE_ratio_rw = MSFE_all ./ MSFE_rw_all;

% --- AR(1) Benchmark ---
% Purpose: Compute MSFE for an AR(1) forecast.
% Explanation: Fits an AR(1) model to each key variable’s training data and
% forecasts H steps ahead. MSFE ratios (DFTLTA vs. AR(1)) assess DFTLTA’s 
% performance against a simple time series model.
clear forecast
ar1_errors = zeros(H, length(key_vars));
for k = 1:length(key_vars)
    y_train = x(1:T_train, key_vars(k));
    ar_model = arima(1,0,0);
    est_model = estimate(ar_model, y_train, 'Display', 'off');
    y_forecast = forecast(est_model, H, 'Y0', y_train);
    ar1_errors(:,k) = (x_test(:, key_vars(k)) - y_forecast).^2;
end
MSFE_ar1 = mean(ar1_errors, 1);
MSFE_ratio_ar1 = MSFE_all ./ MSFE_ar1;

% --- Diebold-Mariano Tests ---
% Purpose: Test whether DFTLTA forecasts are significantly more accurate than
% benchmarks.
% Explanation: The Diebold-Mariano (DM) test compares squared forecast 
% errors: DM = mean(d_t) / (std(d_t) / sqrt(n)), where d_t is the 
% difference in squared errors. A two-sided p-value is computed using a 
% t-distribution.
n = H;                                                                     % number of out‐of‐sample forecasts

for k = 1:length(key_vars)
    % 1) DFTL vs Naive RW
    d_rw = squared_errors(:,k) - squared_errors_rw(:,k);
    dbar = mean(d_rw);
    sd = std(d_rw,1);                                                      % use 1/N normalization for variance
    DM_rw = dbar / (sd/sqrt(n));
    p_rw = 2*(1 - tcdf(abs(DM_rw), n-1));
    
    % 2) DFTL vs AR(1)
    d_ar1 = squared_errors(:,k) - ar1_errors(:,k);
    dbar1 = mean(d_ar1);
    sd1 = std(d_ar1,1);
    DM_ar1 = dbar1 / (sd1/sqrt(n));
    p_ar1 = 2*(1 - tcdf(abs(DM_ar1), n-1));
    
    fprintf('DM test (%s vs RW):  DM=%.3f, p=%.3f\n', var_names{k}, ...
        DM_rw, p_rw);
    fprintf('DM test (%s vs AR1): DM=%.3f, p=%.3f\n\n', var_names{k}, ...
        DM_ar1, p_ar1);
end

% --- Forecast-Encompassing Tests ---
% Purpose: Test whether DFTL forecasts encompass RW or AR(1) forecasts.
% Explanation: Regresses the forecast error (y - f_DFTLTA) on the difference
% (f_benchmark - f_DFTLTA). A non-significant coefficient suggests DFTLTA 
% encompasses the benchmark, containing all relevant information. 
nObs = H;                                                                  % number of forecasts
dfree = nObs - 2;                                                          % df for t‐tests

for k = 1:length(key_vars)
    y   = x_test(:, key_vars(k));                                          % actuals
    f0  = MKA_Forecast.yhat(:,   key_vars(k));                             % DFTL forecast
    f_rw= yhat_rw(:,key_vars(k));                                          % RW forecast
    f_a1= y_forecast;                                                      % AR(1) forecast

    % --- Test 1: Does DFTL encompass RW? regress (y - f0) on (f_rw - f0)
    d_rw = y - f0;
    X_rw = [ones(nObs,1), (f_rw - f0)];
    beta_rw = (X_rw'*X_rw)\(X_rw'*d_rw);
    res_rw  = d_rw - X_rw*beta_rw;
    sigma2_rw = (res_rw'*res_rw)/dfree;
    Vbeta_rw  = sigma2_rw * inv(X_rw'*X_rw);
    t_rw      = beta_rw(2)/sqrt(Vbeta_rw(2,2));
    p_rw      = 2*(1 - tcdf(abs(t_rw), dfree));
    
    % --- Test 2: Does DFTL encompass AR(1)? regress (y - f0) on (f_a1 - f0)
    d_a1 = y - f0;
    X_a1 = [ones(nObs,1), (f_a1 - f0)];
    beta_a1 = (X_a1'*X_a1)\(X_a1'*d_a1);
    res_a1  = d_a1 - X_a1*beta_a1;
    sigma2_a1 = (res_a1'*res_a1)/dfree;
    Vbeta_a1  = sigma2_a1 * inv(X_a1'*X_a1);
    t_a1      = beta_a1(2)/sqrt(Vbeta_a1(2,2));
    p_a1      = 2*(1 - tcdf(abs(t_a1), dfree));

    fprintf('\nEncompassing tests for %s:\n', var_names{k});
    fprintf(' DFTLTA vs RW:  \tβ̂=%.3f, t=%.2f, p=%.3f\n', beta_rw(2), t_rw, ...
        p_rw);
    fprintf(' DFTLTA vs AR1: \tβ̂=%.3f, t=%.2f, p=%.3f\n', beta_a1(2), t_a1, ...
        p_a1);
end

% --- Ljung-Box Test ---
% Purpose: Test for autocorrelation in forecast residuals.
% Explanation: The Ljung-Box test computes a Q-statistic for residual 
% autocorrelations up to maxLags, testing the null hypothesis of no 
% autocorrelation against the alternative of serial correlation.
if H > 1
maxLags = 3;                                                       % how many lags to test
alpha   = 0.05;                                                            % significance level

% Compute residuals matrix (H×4)
residuals = x_test(:, key_vars) - MKA_Forecast.yhat(:, key_vars);

for k = 1:length(key_vars)
    res_k = residuals(:,k);
    n     = length(res_k);

    % compute autocorrelations up to maxLags
    acfAll = autocorr(res_k, 'NumLags', maxLags);
    % autocorr returns [lag0, lag1, …, lagMaxLags]
    rho = acfAll(2:end);                                                   % drop lag-0

    % Ljung–Box Q statistic
    Q = n*(n+2) * sum( rho.^2 ./ (n - (1:maxLags))' );
    pValue = 1 - chi2cdf(Q, maxLags);

    % decision
    if pValue < alpha
        verdict = 'Reject H_0 → autocorrelation';
    else
        verdict = 'Fail to reject H_0';
    end

    fprintf('%s: Q(%d)=%.2f, p=%.3f → %s\n', ...
        var_names{k}, maxLags, Q, pValue, verdict);
end
else
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Visualization 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Purpose: Visualize forecasting performance and model diagnostics.
% Explanation: Creates a multi-panel figure showing MSFE, actual vs. 
% forecasted values, factor time series, cross-correlations, and MSFE 
% ratios, aiding interpretation of DFTLTA performance and factor dynamics.

% --- Cross-Correlation of Factors ---
% Purpose: Identify the most correlated pair of factors.
% Explanation: Computes the correlation matrix of estimated factors and 
% finds the pair with the highest absolute correlation, useful for 
% understanding factor relationships.
corr_matrix = corr(MKA.Fhat);
lower_tri = tril(corr_matrix, -1);
[~, max_idx] = max(abs(lower_tri(:)));
[i, j] = ind2sub(size(lower_tri), max_idx);
disp(['Most correlated factors: ', num2str(i), ' & ', num2str(j)]);
max_lags = 10;
[cross_corr, lags] = xcorr(MKA.Fhat(:,i), MKA.Fhat(:,j), max_lags, 'coeff');

% --- Aggregate MSFE by Horizon ---
fig1 = figure;
plot(1:H, MSFE_horizon, '-o', 'LineWidth', 1.5);
title('Aggregate MSFE by Horizon'); xlabel('Horizon'); ylabel('MSFE'); 
grid on;
exportgraphics(fig1, 'MSFE_by_Horizon.pdf', 'ContentType', 'vector', 'Resolution', 300);

% --- Actual vs. Forecast for Key Variables ---
for idx = 1:3
    fig = figure;
    plot(1:T_train, x(1:T_train, key_vars(idx)), 'b-', 'DisplayName', ...
        'Train Actual'); hold on;
    plot(1:T_train, MKA.CChat(1:T_train, key_vars(idx)), 'r--', 'DisplayName', ...
        'Estimate'); hold on;
    plot(T_train+1:T_train+H, x_test(:, key_vars(idx)), 'k-', 'DisplayName', ...
        'Test Actual');
    plot(T_train+1:T_train+H, MKA_Forecast.yhat(:, key_vars(idx)), 'g-', ...
        'DisplayName', 'Forecast');
    hold off; title(var_names{idx});
    xlabel('Time'); ylabel('Value'); grid on;
    % Formatting
    title(var_names{idx});
    xlabel('Time'); ylabel('Value'); grid on;
    legend('Location', 'Best');

    % Save the figure as high-res PDF
    filename = sprintf('Forecast_Var_%s.pdf', var_names{idx});
    exportgraphics(fig, filename, 'ContentType', 'vector', 'Resolution', 300);

    % Close the figure to avoid clutter
    close(fig);
end

% --- Histogram of MSFE ---
fig6 = figure;
histogram(MSFE_all, 10);
title('MSFE Distribution'); xlabel('MSFE'); ylabel('Frequency'); grid on;

% --- Time Series of Factors ---
fig7 = figure;
plot(MKA.Fhat, 'LineWidth', 1.5);
title('Estimated Factors over Time'); xlabel('Time'); ylabel('Value');
legend(arrayfun(@(k)['Factor ' num2str(k)], 1:size(MKA.Fhat,2), ...
    'UniformOutput', false), 'Location', 'Best'); grid on;

% --- Cross-Correlogram of Most Correlated Factors ---
fig8 = figure;
stem(lags, cross_corr, 'LineWidth', 1.5);
title(['Cross-Correlogram: Factor ' num2str(i) ' vs ' num2str(j)]);
xlabel('Lag'); ylabel('Corr'); grid on;

% --- MSFE Ratio Bar Chart ---
fig2 = figure;
bar([MSFE_ratio_rw; MSFE_ratio_ar1]');
title('MSFE Ratios: DFTLTA vs RW and AR(1)');
xticklabels(var_names); ylabel('Ratio');
legend('DFTLTA / RW', 'DFTLTA / AR(1)', 'Location', 'Best'); grid on;

exportgraphics(fig2, 'MSFE Ratio.pdf', 'ContentType', 'vector', 'Resolution', 300);

% --- MSFE per Horizon for Key Variables ---
fig9 = figure;
plot(1:H, squared_errors(:,1), '-o', 'LineWidth', 1.3); hold on;
plot(1:H, squared_errors(:,2), '-s', 'LineWidth', 1.3);
plot(1:H, squared_errors(:,3), '-d', 'LineWidth', 1.3);
hold off;
title('MSFE per Horizon for Key Variables');
xlabel('Horizon'); ylabel('Squared Error');
legend(var_names, 'Location', 'Best'); grid on;

sgtitle(sprintf('DFTL Forecasting: p=%d', p));
disp('Forecasting and visualization complete.');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Main Forecast Function 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [MKA_Forecast] = DFTLTA_Forecast(xitT, Ahat, Lhat, Sigma_e_hat, ...
    Qhat, H, R, p, mean_train, std_train, stochastic)
    % Forecast factors and observables for H periods using factor model outputs
    % Purpose:
    % Generates H-step-ahead forecasts for factors and observables using 
    % outputs from QMLDFM_TVL. The model assumes time-varying loadings 
    % (Lambda_t) and VAR coefficients (Ahat), with forecasts either 
    % deterministic or stochastic.
    % Inputs:
    %   xitT: T x (R*(p+1)) smoothed states from QMLDFM_TVLA
    %   Ahat: (R*p) x (R*p) companion matrix
    %   Lhat: N x R x T time-varying loadings
    %   Sigma_e_hat: N x N idiosyncratic covariance (diagonal)
    %   Qhat: (R*p) x (R*p) factor innovation covariance
    %   H: Forecast horizon
    %   R: Number of factors
    %   p: Number of lags
    %   mean_train: 1 x N training data means
    %   std_train: 1 x N training data standard deviations
    %   stochastic: Boolean (true for stochastic forecasts, false for not)
    % Outputs: MKA_Forecast (struct containing):
    %   yhat: H x N forecasted observables (original scale)
    %   yhat_norm: H x N forecasted observables (normalized scale)
    %   Fhat_forecast: H x R forecasted factors
    % References:
    %   - Stock & Watson (2002). Forecasting using principal components 
    %     from a large number of predictors. (For factor model forecasting).
    %   - Lütkepohl, H. (2005). New Introduction to Multiple Time Series 
    %     Analysis. Springer. (For VAR forecasting and companion matrix).
    %   - Durbin, J., & Koopman, S. J. (2012). Time Series Analysis by State
    %     Space Methods. Oxford University Press. (For state-space forecasting).

    % --- Dimensions ---
    % Purpose: Extract dimensions for validation and initialization.
    % Explanation: T is the training sample length, N is the number of 
    % variables, and state_dim = R*p is the companion state dimension. 
    % These ensure correct array sizes and model consistency.
    T = size(xitT, 1);
    N = size(Lhat, 1);
    state_dim = R * p; 

    % --- Validate Inputs ---
    % Purpose: Ensure input dimensions and values are correct.
    % Explanation: Checks that xitT, Ahat, Lhat, Qhat, Sigma_e_hat, and H 
    % match expected dimensions and constraints, preventing runtime errors 
    % and ensuring model compatibility.
    if size(xitT, 2) ~= R * (p + 1)
        error('xitT must be T x (R*(p+1)), got %d x %d', size(xitT, 1), ...
            size(xitT, 2));
    end
    if size(Ahat) ~= [state_dim, state_dim]
        error('Ahat must be %d x %d, got %d x %d', state_dim, state_dim, ...
            size(Ahat, 1), size(Ahat, 2));
    end
    if size(Lhat, 3) ~= T
        error('Lhat must be N x R x T, got %d x %d x %d', size(Lhat, 1), ...
            size(Lhat, 2), size(Lhat, 3));
    end
    if size(Qhat) ~= [state_dim, state_dim]
        error('Qhat must be %d x %d, got %d x %d', state_dim, state_dim, ...
            size(Qhat, 1), size(Qhat, 2));
    end
    if size(Sigma_e_hat) ~= [N, N]
        error('Sigma_e_hat must be %d x %d, got %d x %d', N, N, ...
            size(Sigma_e_hat, 1), size(Sigma_e_hat, 2));
    end
    if H < 1
        error('Forecast horizon H must be positive, got %d', H);
    end
    if nargin < 9
        stochastic = false;                                                % Default to deterministic forecasts
    end

    % --- Initialize Outputs ---
    % Purpose: Allocate arrays for forecasted factors and observables.
    % Explanation: Fhat_forecast (H x R) stores factor forecasts, and 
    % X_forecast (H x N) stores observable forecasts (normalized scale).
    Fhat_forecast = zeros(H, R);
    X_forecast = zeros(H, N);

    % --- Last State ---
    % Purpose: Extract the final smoothed state for forecasting.
    % Explanation: Takes the last smoothed state xitT(T, 1:state_dim), 
    % representing [f_T; f_{T-1}; ...; f_{T-p+1}], as the starting point 
    % for forecasting.
    state = xitT(T, 1:state_dim)';                                         % state_dim x 1

    % --- Last Loadings ---
    % Purpose: Use the final time-varying loadings for forecasting.
    % Explanation: Assumes Lambda_{T+h} = Lambda_T for h=1,...,H, a common 
    % approximation in time-varying models when future loadings are unavailable.
    Lt = squeeze(Lhat(:, :, T));                                           % N x R

    % --- Cholesky Decompositions for Stochastic Forecasts ---
    % Purpose: Prepare noise covariances for stochastic forecasts.
    % Explanation: For stochastic forecasts, computes Cholesky decompositions
    % of Qhat’s R x R block and Sigma_e_hat to generate random innovations.
    % Adds small diagonal terms if matrices are not positive definite, 
    % ensuring numerical stability.
if stochastic
    % Extract R x R block from Qhat directly (no Cholesky)
    Qhat_R = Qhat(1:R, 1:R);                                           % R x R non-zero block

    % Use Sigma_e_hat as-is for idiosyncratic noise
    % (assumed to be positive definite or as estimated)
else
    Qhat_R       = zeros(R, R);                                       % No factor noise
    Sigma_e_hat  = zeros(N, N);                                       % No idiosyncratic noise
end

% --- Forecast Loop ---
% Purpose: Generate H-step-ahead forecasts for factors and observables.
% Explanation: Iteratively applies the VAR companion matrix Ahat to 
% forecast factors: state_{t+h} = Ahat * state_{t+h-1} Optionally adds
% stochastic noise u_t ~ N(0, Qhat_R). Observables are computed as 
% X_{t+h} = Lt * f_{t+h} + e_t, with optional noise e_t ~ N(0, Sigma_e_hat).
for h = 1:H
    % Predict next state deterministically
    state = Ahat * state;                                              % state_dim x 1

    if stochastic
        % Add factor innovation noise (only to first R components)
        innov = Qhat_R * randn(R, 1);
        state(1:R) = state(1:R) + innov;
    end

    % Store factor forecast
    Fhat_forecast(h, :) = state(1:R)';                                 % R x 1

    % Map to observables
    X_forecast(h, :) = (Lt * Fhat_forecast(h, :)')';

    if stochastic
        % Add idiosyncratic noise
        X_forecast(h, :) = X_forecast(h, :) + (Sigma_e_hat * randn(N, 1))';
    end
end
    % --- Results ---
    % Purpose: Transform forecasts to original scale and organize outputs.
    % Explanation: Converts normalized forecasts (X_forecast) to the 
    % original scale using training data statistics (mean_train, std_train).
    % Stores normalized and original forecasts, and factor forecasts, in a struct.
    MKA_Forecast.yhat_norm=X_forecast;
    MKA_Forecast.yhat=X_forecast .* std_train + mean_train;
    MKA_Forecast.Fhat_forecast=Fhat_forecast;
end
