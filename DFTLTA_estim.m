%% DFTLTA_estim.m
% Author: Moka Kaleji • Contact: mohammadkaleji1998@gmail.com
% Affiliation: Master Thesis in Econometrics: 
% Advancing High-Dimensional Factor Models: Integrating Time-Varying 
% Parameters with Dynamic Factors.
% University of Bologna

clear; close all; clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% QML Dynamic Factor Model With Time-varying Loadings & A
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main Script for Quasi Maximum Likelihood Dynamic Factor Model with 
% Time-Varying Parameters
% Purpose:
% This script facilitates the estimation of a dynamic factor model with 
% time-varying loadings, and transition matrices.
% It begins with dataset selection, allowing the user to choose between 
% monthly or quarterly data, specify the training sample size, and standardize
% the data. The processed data is then passed to the QMLDFM_TVLA function for 
% model estimation.
% Workflow:
%   1. Dataset selection and loading
%   2. Training sample size specification
%   3. Data standardization
%   4. Model estimation using DFTLA.m
% Dependencies:
%   - DFTLA.m
%   - lsfm.m
%   - MK_VAR.m
%   - MK_ols.m
%   - MATLAB Statistics and Machine Learning Toolbox (for listdlg, readtable)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Dataset Selection, Frequency, Training Sample Size, and Standardization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Purpose:
% Allow the user to select the dataset periodicity (monthly or quarterly), 
% load the corresponding data, specify the training sample size, and 
% standardize the data for numerical stability in model estimation.
% Explanation:
% The dynamic factor model requires a multivariate time series dataset. 
% This section provides a user-friendly interface to choose between pre-processed
% monthly or quarterly datasets, ensuring flexibility in periodicity. 
% The training sample size (T_train) is specified to focus on a subset of 
% the data, which is useful for in-sample estimation and out-of-sample 
% forecasting. Standardization (zero mean, unit variance) is applied to prevent
% numerical issues and ensure consistent scaling across variables, a common
% practice in high-dimensional time series modeling.

% --- Present Available Periodicity Options and Capture User Choice ---
% Purpose: Display a dialog for the user to select dataset periodicity.
% Explanation: The listdlg function provides a graphical interface to choose
% between monthly ('MD1959.xlsx') and quarterly ('QD1959.xlsx') datasets. 
% The selection is validated to ensure a choice is made, halting execution 
% if cancelled to prevent undefined behavior.
options = {'Monthly (MD1959.xlsx)', 'Quarterly (QD1959.xlsx)'};
[choiceIndex, ok] = listdlg('PromptString','Select dataset:',...
                             'SelectionMode','single',...
                             'ListString',options,...
                             'Name','Dataset Selection',...
                             'ListSize',[400 200]);
if ~ok
    error('Dataset selection cancelled. Exiting script.');
end
% --- Load Data Based on Frequency ---
% Purpose: Load the selected dataset from an Excel file and extract the time
% series data.
% Explanation: The filepath is constructed based on the user's choice, 
% pointing to pre-processed datasets stored in a specific directory. The data
% is read into a table using readtable, then converted to a numeric array. 
% For quarterly data, the first column (date index) is excluded, as it is not
% part of the time series. The dimensions T (time points) and N (variables)
% are extracted for subsequent processing.
switch choiceIndex
    case 1                                                                 % Monthly frequency
        filepath = ['/Users/moka/Research/Thesis/Live Project/' ...
            'Processed_Data/MD1959.xlsx'];
        raw = readtable(filepath);
        x = table2array(raw);                                              % Include all series
        T = size(x,1);
    case 2                                                                 % Quarterly frequency
        filepath = ['/Users/moka/Research/Thesis/Live Project/' ...
            'Processed_Data/QD1959.xlsx'];
        raw = readtable(filepath);
        x = table2array(raw(:,2:end));                                     % Drop date index
        T = size(x,1);
    otherwise
        error('Unexpected selection index.');
end
[N_obs, N] = size(x);
% --- Define Training Sample Size ---
% Purpose: Prompt the user to specify the number of observations for the 
% training sample.
% Explanation: The training sample size (T_train) determines the subset of 
% data used for model estimation, allowing the remaining observations for 
% out-of-sample validation or forecasting. A default value of 225 is suggested,
% but the user can input any integer between 1 and T-1. The input is validated
% to ensure it is positive and less than the total number of observations, 
% preventing invalid training periods.
defaultTrain = '225';
prompt = sprintf(['Dataset has %d observations. Enter training size ' ...
    '(T_train):'], T);
userInput = inputdlg(prompt, 'Training Horizon', [3 100], {defaultTrain});
if isempty(userInput)
    error('No training size provided. Exiting.');
end
T_train = str2double(userInput{1});
assert(T_train>0 && T_train<T, 'T_train must be integer in (0, %d)', T);
% --- Standardization ---
% Purpose: Standardize the training data to zero mean and unit variance.
% Explanation: Standardization is critical for numerical stability in 
% high-dimensional factor models, as variables with different scales can lead
% to ill-conditioned matrices or biased factor estimates. The training data
% (first T_train observations) is centered by subtracting the mean and scaled
% by dividing by the standard deviation, computed across the training sample.
% This ensures all variables contribute equally to the factor structure and
% prevents numerical overflow in the EM algorithm.
x_train = x(1:T_train, :);
mean_train = mean(x_train);
std_train  = std(x_train);
x_train_norm = (x_train - mean_train) ./ std_train;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Model Estimation with DFTLAQ
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Purpose:
% Estimate the dynamic factor model using the standardized training data.
% Explanation:
% The QMLDFM_TVLA function is called with the standardized training data 
% x_train_norm, along with user-specified or default parameters for the 
% number of factors (R), bandwidths (h, h_A), VAR lag order (p), and EM 
% algorithm settings (max_iter, tol). The model estimates time-varying 
% loadings, A, and factors, producing outputs for analysis and 
% forecasting. Here, we set example parameters, but these can be adjusted
% based on the dataset or research objectives.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Setting Model Parameters & Running DFM with TVL&A estimation 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- Set Model Parameters ---
% Explanation: Define parameters for QMLDFM_TVLA. These are illustrative 
% values and should be tuned based on the dataset's characteristics 
% (e.g., number of variables, time series dynamics). R=5 assumes a moderate
% number of factors; h=0.1 and h_A=0.1 balance smoothness and flexibility;
% p=2 allows for lagged dynamics; max_iter=1000 and tol=1e-6 ensure convergence.
R = 6;                                                                     % Number of factors
h = 0.2718116211;                                                                   % Bandwidth for LSFM
p = 1;                                                                     % VAR lag order
max_iter = 1000;                                                           % Maximum EM iterations
tol = 1e-6;                                                                % Convergence tolerance
h_A = 0.2718116211;                                                                 % Bandwidth for Ahat
 
% --- Run QMLDFM_TVLA ---
% Explanation: Call the main estimation function with the standardized 
% training data and parameters. Outputs include common components, factors,
% loadings, covariances, and the log-likelihood, which can be used for model
% evaluation, forecasting, or diagnostics.
[MKA] = QMLDFM_TVLA(x_train_norm, R, h, p, mean_train, std_train, max_iter, tol, h_A);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Save Results for Forecasting and Further Analysis 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
save('dftl&A_estim_results.mat', 'R', 'h', 'MKA', 'p', ...
     'x_train', 'x_train_norm', 'mean_train', 'std_train', 'T_train', 'N', ...
     'max_iter', 'tol');
disp('DFTL&A estimation complete. Results saved.');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Main Function 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [MKA] = QMLDFM_TVLA(X, R, h, p, mean_train, std_train, max_iter, tol, h_A)
    % Quasi Maximum Likelihood estimation of a dynamic factor model with 
    % time-varying loadings and Ahat
    % Purpose:
    % Estimates a dynamic factor model where observed data X_t is driven 
    % by latent factors f_t
    % with time-varying factor loadings (Lambda_t) and time-varying VAR 
    % transition matrices (A_t). The model is:
    %     X_t = Lambda_t * f_t + e_t,  e_t ~ N(0, Sigma_e)
    %     f_t = A_t,1 * f_{t-1} + ... + A_t,p * f_{t-p} + u_t,  u_t ~ N(0, Q)
    % Unlike qmldfm_tvl, Qhat is time-invariant, focusing computational 
    % resources on time-varying Lambda_t and A_t. Uses the
    % Expectation-Maximization (EM) algorithm with Kalman filtering and
    % smoothing to estimate parameters and latent states.
    % Inputs:
    %   X: T x N matrix of observed data
    %   R: Number of factors
    %   h: Bandwidth for initial LSFM estimation
    %   p: Number of lags for factor dynamics (VAR(p))
    %   max_iter: Maximum EM iterations
    %   tol: Convergence tolerance
    %   h_A: Bandwidth for time-varying Ahat estimation
    % Outputs: MKA (struct containing):
    %   MKA.CChat: T x N common components
    %   MKA.Fhat: T x R smoothed factors
    %   MKA.xitT: T x (R*(p+1)) smoothed states
    %   MKA.Lhat: N x R x T time-varying loadings
    %   MKA.Sigma_e_hat: N x N idiosyncratic covariance
    %   MKA.Ahat: T x p cell array of R x R VAR coefficient matrices
    %   MKA.Ahat_companion: (R*p) x (R*p) companion matrix from Ahat at t=T
    %   MKA.Qhat: (R*p) x (R*p) process noise covariance
    %   MKA.logL: Log-likelihood
    %   MKA.Rhat: N x N prediction error covariance
    %   MKA.PtT: (R*(p+1)) x (R*(p+1)) x T smoothed state covariance
    %   MKA.eigvals_Ahat: Eigenvalues of Ahat companion matrices
% References:
%   - Barigozzi, Matteo & Luciani, Matteo. (2024). Quasi Maximum Likelihood
%     Estimation and Inference of Large Approximate Dynamic Factor Models
%     via the EM algorithm. Finance and Economics Discussion Series. 
%     1-135. 10.17016/FEDS.2024.086. 
%   - Hafner, Christian & Motta, Giovanni & Sachs, Rainer. (2011). 
%     Locally stationary factor models: Identification and nonparametric
%     estimation. Econometric Theory. 27. 1279-1319. 10.1017/S0266466611000053. 
%   - Dahlhaus, R. (1996). Asymptotic statistical inference for 
%     nonstationary processes with evolutionary spectra.
%   - Stock, J. H., & Watson, M. W. (2002). Forecasting using principal 
%     components from a large number of predictors. Journal of the American
%     Statistical Association, 97(460), 1167-1179.
%   - Doz, C., Giannone, D., & Reichlin, L. (2011). A two-step estimator for
%     large approximate dynamic factor models based on Kalman filtering. 
%     Journal of Econometrics, 164(1), 188-205.
%   - Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood
%     from incomplete data via the EM algorithm. Journal of the Royal 
%     Statistical Society: Series B, 39(1), 1-38.
%   - Durbin, J., & Koopman, S. J. (2012). Time Series Analysis by State Space
%     Methods. Oxford University Press.
%   - Barigozzi, Matteo. "Quasi Maximum Likelihood Estimation of High-Dimensional
%     Factor Models." Oxford Research Encyclopedia of Economics and Finance.
%     21 Aug. 2024; Accessed 12 May. 2025.


    % --- Dimensions ---
  % Purpose: Extract dimensions.
  % Explanation: The input data X is a T x N matrix where T is the number
  % of time points and N is the number of observed variables. 
    [T, N] = size(X);

    % --- Validate Inputs ---
    % Purpose: Set default bandwidth for time-varying Ahat estimation if 
    % not provided.
    % Explanation: The bandwidth h_A controls the smoothness of Ahat estimates
    % in the kernel-weighted least squares step, balancing flexibility 
    % (capturing time variation) and stability (avoiding overfitting). 
    % A default value of 0.1 is chosen based on empirical performance in 
    % time-varying parameter models.
    if nargin < 7
        h_A = 0.1;                                                         % Default bandwidth
    end

    % --- Initial Estimates via Locally Stationary Factor Model (LSFM) ---
    % Purpose: Obtain initial estimates of time-varying loadings, factors, 
    % and idiosyncratic covariance using a kernel-based static factor model.
    % Explanation: The lsfm function employs kernel-weighted principal 
    % component analysis (PCA) to estimate Lambda_t (N x R), f_t (R x 1), 
    % and Sigma_t (N x N) at each time t. This captures time-varying 
    % relationships via local covariance structures, providing a robust 
    % starting point for the EM algorithm. Sigma_e_hat is initialized as 
    % a diagonal matrix to enforce sparsity and identifiability.
    [~, Fhat_initial, Lhat, Sigmahat] = lsfm(X, R, h);
    Fhat = Fhat_initial; % T x R
    Lhat = permute(Lhat, [1, 2, 3]); % N x R x T
    Sigma_e_hat = Sigmahat;

    % --- VAR(p) Initialization ---
    % Purpose: Initialize the time-varying VAR(p) coefficients Ahat and 
    % process noise covariance Qhat.
    % Explanation: The MK_VAR function estimates a time-invariant VAR(p) 
    % model on the initial factors Fhat_initial to obtain starting values 
    % for Ahat_t,i (R x R matrices for lags i=1,...,p), replicated across 
    % all t. Qhat is initialized as a time-invariant (R*p) x (R*p) matrix, 
    % with the top-left R x R block set to the covariance of factor 
    % differences, assuming stationarity in the noise process. The state 
    % dimensions (state_dim, state_dim_aug) reflect the companion form and 
    % augmented state for Kalman filtering.
    [~, ~, AL0] = MK_VAR(Fhat_initial, p, 0);                              % R x R x p
    state_dim = R * p;                                                     % Companion state dimension: e.g., 10 for R=5, p=2
    state_dim_aug = R * (p + 1);                                           % Augmented state for xitT: e.g., 15
    Ahat = cell(T, p);                                                     % T x p cell array
    for t = 1:T
        for lag = 1:p
            Ahat{t, lag} = AL0(:, :, lag);                                 % R x R
        end
    end
    % Initialize Qhat using VAR(p) residuals
    Qhat = zeros(state_dim, state_dim);
    % Fit VAR(p) and compute residuals
    % Prepare lagged data
    Y = Fhat_initial(p+1:end, :);                                          % (T-p)-by-R, dependent variable
    qq = zeros(T-p, R*p);                                                  % Regressors
    for lag = 1:p
        qq(:, (lag-1)*R+1:lag*R) = Fhat_initial(p+1-lag:end-lag, :);
    end
                                                                           % Estimate VAR coefficients
    A = (qq' * qq) \ (qq' * Y);                                            % R*p-by-R
    % Compute residuals
    residuals = Y - qq * A;                                                % (T-p)-by-R

    % Compute covariance of residuals
    Qhat(1:R, 1:R) = cov(residuals);                                       % R-by-R covariance

    % Ensure positive definiteness
    Qhat(1:R, 1:R) = (Qhat(1:R, 1:R) + Qhat(1:R, 1:R)') / 2;               % Ensure symmetry
    [V, D] = eig(Qhat(1:R, 1:R));
    D = max(real(diag(D)), 1e-6);                                          % Ensure positive eigenvalues
    Qhat(1:R, 1:R) = V * diag(D) * V';                                     % Reconstruct positive definite matrix

    % --- Precompute Kernel Weights for Ahat ---
    % Purpose: Compute Gaussian kernel weights for smoothing time-varying 
    % Ahat estimates.
    % Explanation: Time-varying Ahat_t is estimated using kernel-weighted 
    % least squares, with weights determined by a Gaussian kernel 
    % K((u_s - u_t)/h_A), where u_t = t/T normalizes time to [0,1]. The 
    % bandwidth h_A controls the smoothing window, and weights 
    % are normalized to sum to 1 for each t, ensuring proper weighted 
    % averaging in the M-step. This approach captures smooth temporal 
    % variation in Ahat_t.
    u = (1:T)' / T;
    weights = cell(T, 1);
    for t = 1:T
        z = (u - u(t)) / h_A;
        weights{t} = (1 / sqrt(2*pi)) * exp(-0.5 * z.^2);
        weights{t} = weights{t} / sum(weights{t});
    end

    %%%%%%%%%%%%%%%%
    % EM Algorithm %
    %%%%%%%%%%%%%%%%
    % Purpose: Iteratively estimate latent states and model parameters 
    % using the EM algorithm.
    % Explanation: The EM algorithm alternates between the E-step 
    % (estimating latent states f_t via Kalman filtering and smoothing) and
    % the M-step (updating parameters Ahat_t, Qhat, and Sigma_e_hat). 
    % Convergence is assessed by the relative change in log-likelihood,
    % maximizing the expected complete-data log-likelihood. This approach 
    % handles incomplete data (latent factors) effectively in dynamic factor 
    % models.
    decrease_count=0;
    logL_prev = -Inf;
    for iter = 1:max_iter

        %%%%%%%%%%
        % E-Step %
        %%%%%%%%%%
        % Kalman Filter and Smoother
        % Purpose: Estimate latent states and their covariances given 
        % current parameters.
        % Explanation: The E-step uses Kalman filtering to compute filtered
        % states (f_{t|t}) and covariances (P_{t|t}) forward in time, 
        % followed by Kalman smoothing to compute smoothed states (f_{t|T})
        % and covariances (P_{t|T}) using all observations. 
        % The log-likelihood is accumulated to monitor convergence, 
        % critical for EM convergence assessment.
        logL = 0;
        xitT = zeros(T, state_dim_aug); % T x (R*(p+1))
        Fhat_filt_aug = zeros(T, state_dim_aug);
        P_filt_aug = zeros(state_dim_aug, state_dim_aug, T);
        Fhat_pred_aug = zeros(T, state_dim_aug);
        P_pred_aug = zeros(state_dim_aug, state_dim_aug, T);
        Fhat_smooth_aug = zeros(T, state_dim_aug);
        P_smooth_aug = zeros(state_dim_aug, state_dim_aug, T);

        % --- Initial State ---
        % Purpose: Initialize the state and covariance at t=1.
        % Explanation: The initial state is set using the first p+1 factor
        % estimates from
        % LSFM, arranged in the augmented state vector 
        % [f_t; f_{t-1}; ...; f_{t-p}]. The initial covariance is set to the
        % identity matrix, reflecting moderate uncertainty compared to a 
        % diffuse prior (e.g., 1e6 * eye), assuming reasonable initial 
        % conditions for simplicity.
        for pp = 1:p+1
            if pp <= T
                Fhat_filt_aug(1, (pp-1)*R+1:pp*R) = Fhat_initial(pp, :);
            end
        end
        P_filt_aug(:, :, 1) = eye(state_dim_aug);

        %%%%%%%%%%%%%%%%%
        % Kalman Filter %
        %%%%%%%%%%%%%%%%%
        % Purpose: Compute filtered states and covariances forward in time.
        % Explanation: The Kalman filter alternates between prediction 
        % (using the state transition model) and update 
        % (incorporating observations). The state vector is augmented to 
        % include lagged factors, and the transition matrix A_aug_t is
        % time-varying. The log-likelihood is computed for t > p to account
        % for initial lags, ensuring valid likelihood contributions.
        for t = 1:T
            % --- Time-Varying Transition Matrix ---
            % Purpose: Construct the augmented transition matrix A_aug_t.
            % Explanation: A_aug_t is an (R*(p+1)) x (R*(p+1)) matrix in 
            % companion form, with the first R rows containing Ahat_t,i for
            % lags i=1,...,p, and subdiagonal blocks as identity matrices to
            % shift lagged states. This enables modeling VAR(p) dynamics in
            % a first-order state-space form, crucial for Kalman filtering.
            A_aug_t = zeros(state_dim_aug, state_dim_aug);                 % e.g., 15 x 15
            for lag = 1:p
                A_aug_t(1:R, (lag-1)*R + 1 : lag*R) = Ahat{t, lag};        % R x R blocks
            end
            for i = 1:p
                A_aug_t(R*i+1:R*(i+1), R*(i-1)+1:R*i) = eye(R);            % Subdiagonal identities
            end

            % --- Process Noise ---
            % Purpose: Define the process noise covariance for the 
            % augmented state.
            % Explanation: Q_aug is an (R*(p+1)) x (R*(p+1)) matrix with 
            % Qhat’s top-left R x R block representing noise for current 
            % factors, and zeros elsewhere, as lagged factors are 
            % deterministic shifts in the companion form. Qhat is 
            % time-invariant, simplifying the model compared to qmldfm_tvl.
            Q_aug = zeros(state_dim_aug, state_dim_aug);
            Q_aug(1:R, 1:R) = Qhat(1:R, 1:R);

            % --- Prediction ---
            % Purpose: Predict the state and covariance at t given t-1.
            % Explanation: The prediction step computes:
            %   f_{t|t-1} = A_t * f_{t-1|t-1}
            %   P_{t|t-1} = A_t * P_{t-1|t-1} * A_t' + Q
            % For t=1, the initial state and covariance are used directly,
            % ensuring a proper starting point for the filter.
            if t > 1
                Fhat_pred_aug(t, :) = A_aug_t * Fhat_filt_aug(t-1, :)';
                P_pred_aug(:, :, t) = A_aug_t * P_filt_aug(:, :, t-1) *...
                    A_aug_t' + Q_aug;
            else
                Fhat_pred_aug(t, :) = Fhat_filt_aug(t, :);
                P_pred_aug(:, :, t) = P_filt_aug(:, :, t);
            end

            % --- Observation Matrix ---
            % Purpose: Define the observation matrix H_t mapping states to
            % observations.
            % Explanation: H_t = [Lambda_t, 0] maps the augmented state 
            % (current and lagged factors) to observations, where Lambda_t
            % is N x R, and zeros account for lagged factors not directly
            % affecting X_t. This structure aligns with the factor model’s
            % observation equation.
            Lt = squeeze(Lhat(:, :, t));                                   % N x R
            L = [Lt, zeros(N, state_dim_aug - R)];                         % N x (R*(p+1))

            % --- Update ---
            % Purpose: Update the state and covariance using the observation X_t.
            % Explanation: The update step computes:
            %   v_t = X_t - L * f_{t|t-1} (innovation)
            %   H = L * P_{t|t-1} * L' + Sigma_e (innovation covariance)
            %   K_t = P_{t|t-1} * L' * S_t^{-1} (Kalman gain)
            %   f_{t|t} = f_{t|t-1} + K_t * v_t
            %   P_{t|t} = P_{t|t-1} - K_t * L * P_{t|t-1}
            y_pred = L * Fhat_pred_aug(t, :)';
            v_t = X(t, :)' - y_pred;
            H = L * P_pred_aug(:, :, t) * L' + Sigma_e_hat;                % Conditional variance of the Observation
            Hinv = inv(H);
            logL = logL + 0.5 * (-log(det(H)) - v_t' * Hinv * v_t);

            K = P_pred_aug(:, :, t) * L' * Hinv;
            Fhat_filt_aug(t, :) = Fhat_pred_aug(t, :) + (K * v_t)';
            P_filt_aug(:, :, t) = P_pred_aug(:, :, t) - K * L * ...
                P_pred_aug(:, :, t);
        end

        % --- Debug: Check Filtered States ---
        % Purpose: Ensure filtered states are non-zero to detect potential 
        % numerical issues.
        % Explanation: If Fhat_filt_aug is all zeros, it indicates a failure
        % in the filtering process (e.g., numerical instability or incorrect
        % initialization). A warning is issued to alert the user, 
        % facilitating debugging and ensuring robust execution.
        if all(Fhat_filt_aug(:) == 0)
            warning('Fhat_filt_aug is all zeros at iteration %d', iter);
        end

        %%%%%%%%%%%%%%%%%%%
        % Kalman Smoother %
        %%%%%%%%%%%%%%%%%%%
        % Purpose: Compute smoothed states and covariances using all 
        % observations.
        % Explanation: The smoother refines filtered estimates by incorporating
        % future observations:
        %   J_t = P_{t|t} * A_{t+1}' * P_{t+1|t}^{-1} (smoothing gain)
        %   f_{t|T} = f_{t|t} + J_t * (f_{t+1|T} - f_{t+1|t})
        %   P_{t|T} = P_{t|t} + J_t * (P_{t+1|T} - P_{t+1|t}) * J_t'
        % A pseudo-inverse is used for P_{t+1|t}^{-1} to handle potential 
        % singularity, ensuring numerical stability in high-dimensional settings.
        Fhat_smooth_aug(T, :) = Fhat_filt_aug(T, :);
        P_smooth_aug(:, :, T) = P_filt_aug(:, :, T);
        for t = T-1:-1:1
            % Construct A_aug_t_next for t+1
            A_aug_t_next = zeros(state_dim_aug, state_dim_aug);
            for lag = 1:p
                A_aug_t_next(1:R, (lag-1)*R + 1 : lag*R) = Ahat{t+1, lag};
            end
            for i = 1:p
                A_aug_t_next(R*i+1:R*(i+1), R*(i-1)+1:R*i) = eye(R);
            end

            % Use pseudo-inverse for stability
            J = P_filt_aug(:, :, t) * A_aug_t_next' * pinv(P_pred_aug ...
                (:, :, t+1));
            Fhat_smooth_aug(t, :) = Fhat_filt_aug(t, :) + (J * ...
                (Fhat_smooth_aug(t+1, :) - Fhat_pred_aug(t+1, :))')';
            P_smooth_aug(:, :, t) = P_filt_aug(:, :, t) + J * ...
                (P_smooth_aug(:, :, t+1) - P_pred_aug(:, :, t+1)) * J';
        end

        % --- Debug: Check Smoothed States ---
        % Purpose: Ensure smoothed states are non-zero to detect potential
        % issues. Explanation: A zero xitT indicates a smoothing failure
        % (e.g., due to numerical errors or degenerate covariances), 
        % prompting a warning to guide debugging and ensure reliable state 
        % estimates.
        if all(Fhat_smooth_aug(:) == 0)
            warning('xitT is all zeros at iteration %d', iter);
        end
        xitT = Fhat_smooth_aug;                                            % T x (R*(p+1))
        Fhat = xitT(:, 1:R);                                               % T x R

        %%%%%%%%%%
        % M-Step %
        %%%%%%%%%%
        % Purpose: Update model parameters to maximize the expected 
        % log-likelihood.
        % Explanation: Using smoothed states, update Ahat_t, Qhat, and 
        % Sigma_e_hat via kernel-weighted least squares and covariance 
        % estimation. Lambda_t is not updated, retaining LSFM estimates, 
        % which simplifies the M-step but assumes initial loadings are 
        % sufficiently accurate. This approach reduces computational burden
        % but may limit model flexibility compared to qmldfm_tvl.
        Ahat_new = cell(T, p);
        max_eigvals_iter = zeros(T, 1);                                    % Track max eigenvalues for summary
        for t = 1:T
            % --- Update Ahat_t ---
            % Purpose: Estimate time-varying VAR coefficients Ahat_t.
            % Explanation: Ahat_t is estimated by kernel-weighted least squares:
            %   Ahat_t = (sum w_{t,s} * f_s * f_{s-1}') *
            %   (sum w_{t,s} * f_{s-1} * f_{s-1}')^{-1}
            % Stability is enforced by ensuring eigenvalues of the companion
            % matrix have magnitude <= 0.99, critical for valid VAR dynamics
            % and stationarity.
            sum_FtFt_lags = zeros(R, state_dim);                           % R x (R*p)
            sum_Ft_lagsFt_lags = zeros(state_dim, state_dim);              % (R*p) x (R*p)
            for s = p+1:T
                w = weights{t}(s);
                Ft = xitT(s, 1:R)';                                        % R x 1
                Ft_lags = xitT(s-1, 1:state_dim)';                         % (R*p) x 1
                sum_FtFt_lags = sum_FtFt_lags + w * (Ft * Ft_lags');
                sum_Ft_lagsFt_lags = sum_Ft_lagsFt_lags + w * (Ft_lags * Ft_lags');
            end

            %sum_Ft_lagsFt_lags = sum_Ft_lagsFt_lags + 1e-6 * eye(state_dim);
            % Note: Regularization (e.g., adding 1e-6 * eye) is commented 
            % out, relying on data properties for invertibility of 
            % sum_Ft_lagsFt_lags, which may risk singularity in small 
            % samples or noisy data.
            Ahat_t = sum_FtFt_lags / sum_Ft_lagsFt_lags;                   % R x (R*p)
            Ahat_comp_t = zeros(state_dim, state_dim);
            Ahat_comp_t(1:R, :) = Ahat_t;
            for i = 1:p-1
                Ahat_comp_t(R*i+1:R*(i+1), R*(i-1)+1:R*i) = eye(R);
            end
            eigvals = eig(Ahat_comp_t);
            max_eig = max(abs(eigvals));
            max_eigvals_iter(t) = max_eig;                                 % Store for summary

            if any(abs(eigvals) >= 1)
                % Project Ahat_t to ensure stability
                % Explanation: If any eigenvalue exceeds 0.99 in magnitude,
                % the companion matrix is adjusted by capping real 
                % eigenvalues between -0.99 and 0.99, ensuring stationarity
                % of the VAR process. This projection preserves the matrix 
                % structure while enforcing dynamic stability.
                [V, D] = eig(Ahat_comp_t);
                D = diag(min(max(real(diag(D)), -0.99), 0.99));
                Ahat_comp_t = real(V * D * inv(V));
                Ahat_t = Ahat_comp_t(1:R, :);
                eigvals = eig(Ahat_comp_t);
                max_eig = max(abs(eigvals));
                fprintf(['Iteration %d, t=%d: Max |eigenvalue| after ' ...
                    'enforcement: %.6f\n'], iter, t, max_eig);
            end
          
            eigvals_Ahat(t, :) = eigvals.';                                % Store eigenvalues

            % Store in cell array
            for lag = 1:p
                Ahat_new{t, lag} = Ahat_t(:, (lag-1)*R + 1 : lag*R);       % R x R
            end
        end
        Ahat = Ahat_new;

        % --- Print Eigenvalue Summary ---
        % Purpose: Provide a summary of eigenvalue stability for the iteration.
        % Explanation: The mean and maximum of the maximum eigenvalue 
        % magnitudes across time points are printed, offering insight into 
        % the stability of the VAR dynamics. This diagnostic aids in assessing
        % model reliability and debugging potential instability issues.
        fprintf(['Iteration %d: Mean max |eigenvalue| = %.6f, Max max ' ...
            '|eigenvalue| = %.6f\n'], ...
            iter, mean(max_eigvals_iter), max(max_eigvals_iter));

        % --- Update Qhat ---
        % Purpose: Estimate time-invariant process noise covariance Qhat.
        % Explanation: Qhat’s top-left R x R block is updated as the average
        % covariance of residuals u_t = f_t - sum(A_t,i * f_{t-i}), computed
        % across t=p+1,...,T. The remaining elements of Qhat are zero, 
        % reflecting the companion form structure. The time-invariant 
        % assumption simplifies estimation but may miss temporal changes in
        % noise variance.
        EF = zeros(R, R);
        EF1 = zeros(R, state_dim);
        for t = 2:T
            Ft = xitT(t, 1:R)';                                            % Smoothed factors
            Ft_lags = xitT(t-1, 1:state_dim)';                             % Lagged states
            PtT_t = P_smooth_aug(1:R, 1:R, t);                             % Smoothed covariance
            PtT_cross = P_smooth_aug(1:R, 1:state_dim, t);                 % Cross-covariance
            EF = EF + Ft * Ft' + PtT_t;
            EF1 = EF1 + Ft * Ft_lags' + PtT_cross;
        end
        Qhat_new = (EF - Ahat_t(1:R, 1:state_dim) * EF1') / (T - 1);       % Use last Ahat_t or adjust as needed
        Qhat(1:R, 1:R) = (Qhat_new + Qhat_new') / 2;

        % --- Update Sigma_e_hat ---
        % Purpose: Estimate idiosyncratic noise covariance Sigma_e_hat.
        % Explanation: Sigma_e_hat is the average covariance of observation
        % residuals e_t = X_t - Lambda_t * f_t, enforced as diagonal to 
        % reduce parameters and ensure identifiability in the factor model.
        % This assumption aligns with standard practice in large-scale 
        % factor models to manage computational complexity.
        % Sigma_e_hat = sum(eta_t * eta_t' + Lambda_t * P_t * Lambda_t') / T
        Sigma_e_new = zeros(N, N);
        for t = 1:T
            Lt = squeeze(Lhat(:, :, t));
            resid = X(t, :)' - Lt * xitT(t, 1:R)';
            PtT_t = P_smooth_aug(1:R, 1:R, t);                           % Covariance of factors at t
            Sigma_e_new = Sigma_e_new + resid * resid'+ Lt * PtT_t * Lt';
        end
        Sigma_e_hat = diag(diag(Sigma_e_new / T));

        % --- Convergence Check ---
        % Purpose: Assess convergence based on relative log-likelihood change.
        % Explanation: Convergence is reached when:
        %   |logL - logL_prev| / (|logL| + |logL_prev| + 1e-3) / 2 < tol
        % The small constant 1e-3 prevents division by zero, ensuring robust
        % convergence checking. This criterion is standard in EM algorithms
        % for monitoring likelihood improvement.
        if iter > 1
            % did it decrease this iteration?
                if logL < logL_prev
                    decrease_count = decrease_count + 1;
                    fprintf('Iteration %d: logL Decreased (count = %d)\n', iter, ...
                        decrease_count);
                    if decrease_count == 2
                        fprintf(['Log-likelihood Decreased twice: stopping at ' ...
                            'iter %d\n'], iter);
                        break;
                    end
                end
        
            % check normal convergence criterion
            rel_change = abs(logL - logL_prev) / (abs(logL) + abs ...
                (logL_prev) + 1e-3) / 2;
            fprintf('Iteration %d: logL = %.4f, rel_change = %.4e\n', ...
                iter, logL, rel_change);
            if rel_change < tol
                fprintf('Converged at iteration %d (rel_change < tol)\n', ...
                    iter);
                break;
            end
        end
    
    % update for next iter
    logL_prev = logL;
    end

    % --- Common Components ---
    % Purpose: Calculate the common components C_t = Lambda_t * f_t.
    % Explanation: These represent the portion of X_t explained by the 
    % latent factors, useful for forecasting, decomposition, or evaluating 
    % model fit. The common components are central to factor models for 
    % summarizing high-dimensional data.
    CChat = zeros(T, N);
    for t = 1:T
        Lt = squeeze(Lhat(:, :, t));
        CChat(t, :) = (Lt * xitT(t, 1:R)')';
    end

    % --- Construct Companion Matrix from Ahat{T, :} ---
    % Purpose: Form the companion matrix for the final time point’s VAR 
    % coefficients.
    % Explanation: The companion matrix represents the VAR(p) dynamics in
    % first-order form, used for stability analysis, forecasting, or dynamic
    % simulations. It encapsulates the time-varying Ahat at t=T, providing
    % a snapshot of the final dynamics.
    Ahat_companion = zeros(state_dim, state_dim);
    for lag = 1:p
        Ahat_companion(1:R, (lag-1)*R + 1 : lag*R) = Ahat{T, lag};
    end
    for i = 1:p-1
        Ahat_companion(R*i+1:R*(i+1), R*(i-1)+1:R*i) = eye(R);
    end

    % --- Set PtT ---
    % Purpose: Store the smoothed state covariances for output.
    % Explanation: PtT contains the smoothed covariances P_{t|T} for the
    % augmented state, useful for uncertainty quantification, confidence 
    % intervals, or inference about latent factors.
    PtT = P_smooth_aug;                                                    % (R*(p+1)) x (R*(p+1)) x T
    logL = logL_prev;

    % --- Residuals ---%
    resid = zeros(T, N);
    for t = 1:T
        Lt = squeeze(Lhat(:, :, t));
        resid = X(t, :)' - Lt * xitT(t, 1:R)';
    end

    % --- Results ---
    % Purpose: Organize outputs into a struct for convenient access.
    % Explanation: The MKA struct consolidates all model estimates and 
    % diagnostics, including common components, factors, loadings, 
    % covariances, log-likelihood, and eigenvalues of Ahat companion 
    % matrices. This structure facilitates further analysis, forecasting, 
    % or reporting, aligning with standard practices in econometric modeling.
    MKA.CChat=mean_train+CChat.*std_train;
    MKA.Fhat=Fhat;
    MKA.xitT=xitT; 
    MKA.Lhat=Lhat; 
    MKA.Sigma_e_hat=Sigma_e_hat;
    MKA.Ahat=Ahat; 
    MKA.Ahat_companion=Ahat_companion; 
    MKA.Qhat=Qhat; 
    MKA.logL=logL;
    MKA.PtT=PtT;
    MKA.eigvals_Ahat=eigvals_Ahat;
    MKA.Residuals=resid;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Supporting Functions 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%
% LSFM %
%%%%%%%%
% Locally Stationary Factor Model (LSFM)
% Purpose:
% Estimates a static factor model at each time point using 
% kernel-weighted PCA to provide initial estimates of time-varying loadings
% (Lambda_t), factors (f_t), and idiosyncratic covariance (Sigma_t).
% Reference:
%   - Dahlhaus, R. (1996)
%   - Motta G, Hafner CM, von Sachs R. (2011)

function [CChat, Fhat, Lhat, Sigma_e_hat] = lsfm(X, R, h)
% Mathematical Formulation:
% For each t, compute a local covariance matrix:
% Sigma_t = sum(w_{t,s} * X_s * X_s'),  w_{t,s} = K((u_s - u_t)/h) / sum(K)
% where K is a Gaussian kernel, u_t = t/T, and h is the bandwidth. 
% Loadings Lambda_t are the top R eigenvectors of Sigma_t, and factors 
% are f_t = (Lambda_t' * Lambda_t)^{-1} * Lambda_t' * X_t.
% Inputs:
%   X: T x N data matrix
%   R: Number of factors
%   h: Bandwidth for kernel smoothing
% Outputs:
%   CChat: T x N common components
%   Fhat: T x R factors
%   Lhat: N x R x T loadings
%   Sigmahat: N x N x T covariance matrices

    % --- Dimensions ---
    [T, N] = size(X);


    % --- Initialize Outputs ---
    CChat = zeros(T, N);
    Fhat = zeros(T, R);
    Lhat = zeros(N, R, T);
    Sigmahat = zeros(N, N, T);

    % --- Compute Kernel Weights and Local Covariance ---
    % Explanation: For each t, compute weights based on temporal proximity 
    % using a Gaussian kernel. The local covariance Sigma_t is a weighted 
    % sum of outer products X_s * X_s', capturing local data structure.
    u = (1:T)' / T;                                                        % Normalized time index
    for t = 1:T                                                            
        u_t = u(t);
        z = (u - u_t) / h;
        weights = (1/sqrt(2*pi)) * exp(-0.5 * z.^2);                       % Gaussian kernel
        weights = weights / sum(weights);                                  % Normalize
        for i = 1:N
            for j = 1:i
                cross_prod = X(:,i) .* X(:,j);
                Sigmahat(i,j,t) = sum(weights .* cross_prod);
                Sigmahat(j,i,t) = Sigmahat(i,j,t);                         % Ensure symmetry
            end
        end
    end

    % --- Eigenvalue Decomposition and Factor Estimation ---
    % Explanation: For each t, compute the top R eigenvectors of Sigma_t to
    % obtain Lambda_t. Factors are estimated by projecting X_t onto Lambda_t,
    % and common components are C_t = Lambda_t * f_t. The sign of loadings 
    % is adjusted for consistency.
    opts.disp = 0;                                                         % Suppress eigs output
    for t = 1:T
        Sigma_t = squeeze(Sigmahat(:,:,t));
        Sigma_t = (Sigma_t + Sigma_t')/2;                                  % Ensure symmetry
        [V,D] = eig(Sigma_t);
        D = max(real(diag(D)),0);                                          % Ensure non-negative eigenvalues
        Sigma_t = V * diag(D) * V';                                        % Reconstruct positive semi-definite matrix
        % Compute eigenvectors and eigenvalues
        [A, D] = eigs(Sigma_t, R, 'largestabs', opts);                     % Top R eigenvectors and eigenvalues
        eigenvalues = diag(D);                                             % Extract eigenvalues as a vector
        sqrt_eigenvalues = sqrt(eigenvalues);                              % Square root of eigenvalues
        
        % Adjust sign of eigenvectors for consistency
        sign_adjust = diag(sign(A(1,:)));
        A_adjusted = A * sign_adjust;

        E = X - squeeze( sum( Lhat .* reshape(Fhat',1,R,T), 2 ) )';        % T×N residuals

        % estimate idiosyncratic variances directly
        sigma_e_vec = mean( E.^2, 1 )';                                    % N×1 vector of idio variances
        Sigma_e_hat = diag( sigma_e_vec );
        
        % Initialize loadings and factors per user's request
        Lhat(:,:,t) = A_adjusted .* sqrt_eigenvalues';                     % Lhat = A * sqrt(D), scaling each column
        A_scaled = A_adjusted ./ sqrt_eigenvalues';                        % For Fhat = X * A / sqrt(D)
        Fhat(t,:) = X(t,:) * A_scaled;                                     % Factors scaled inversely
        CChat(t,:) = Fhat(t,:) * Lhat(:,:,t)';                             % Common component
    end
end

%%%%%%%%%%
% MK_VAR %
%%%%%%%%%%
% Purpose: Estimate a VAR(p) model on factor time series.
% Explanation: Constructs lagged regressors and applies OLS (via MK_ols) to
% estimate VAR coefficients, supporting deterministic terms (constant, trend).
% Reference: 
% - Lütkepohl (2005) for VAR estimation.
% - Matteo Barigozzi & Matteo Luciani, 2024.
function [A, u, AL] = MK_VAR(y, k, det)
% Vector Autoregressive (VAR) Model Estimation
% Purpose:
% Estimates a VAR(k) model on a multivariate time series y_t:
%     y_t = A_1 * y_{t-1} + ... + A_k * y_{t-k} + u_t
%   where A_i are coefficient matrices, and u_t is the residual.
% Inputs:
%   y: T x R time series matrix
%   k: Number of lags
%   det: Deterministic terms (0: none, 1: constant, 2: trend, 3: both)
% Outputs:
%   A: Coefficient matrix (including deterministic terms)
%   u: T-k x R residuals
%   AL: R x R x k coefficient matrices

    % --- Dimensions and Data Preparation ---
    [T, R] = size(y);
    yy = y(k+1:T,:);                                                       % Dependent variable (t=k+1,...,T)
    xx = NaN(T-k, R*k);                                                    % Lagged regressors
    for ii = 1:R
        for jj = 1:k
            xx(:, k*(ii-1)+jj) = y(k+1-jj:T-jj, ii);                       % Construct lags
        end
    end
    % --- Deterministic Terms ---
    % Explanation: Include constant or trend as specified by det.
    if det == 0
        ll = 0;
    elseif det == 3
        ll = 2;
    else
        ll = 1;
    end
    % --- OLS Estimation ---
    % Explanation: Estimate coefficients for each variable using OLS via ML_ols.
    A = NaN(R*k + ll, R);                                                  % Coefficient matrix
    u = NaN*yy;                                                            % Residuals
    for ii = 1:R
        [A(:,ii), u(:,ii)] = MK_ols(yy(:,ii), xx, det);
    end
    % --- Reshape Coefficients ---
    % Explanation: Extract VAR coefficients, excluding deterministic terms,
    % and reshape into R x R x k array for compatibility with DFTLTATQ.
    At = A;
    if det == 3
        At(1:2,:) = [];
    elseif det == 1 || det == 2
        At(1,:) = [];
    end
    AL = NaN(R, R, k);
    for kk = 1:k
        AL(:,:,kk) = At(1+kk-1:k:end,:)';
    end
end

%%%%%%%%%%
% MK_ols %
%%%%%%%%%%
% Purpose: Perform OLS regression with optional deterministic terms.
% Explanation: Estimates coefficients beta via OLS, supporting constant, 
% trend, or both.
% Reference: 
% - Lütkepohl (2005) for OLS in time series.
% - Matteo Barigozzi & Matteo Luciani, 2024.
function [beta, u] = MK_ols(y, x, det)
% Ordinary Least Squares (OLS) Regression
% Purpose:
% Estimates coefficients beta in the linear model y = x * beta + u,
% optionally including deterministic terms (constant, trend).
% Inputs:
%   y: T x 1 dependent variable
%   x: T x k regressor matrix
%   det: Deterministic terms (0: none, 1: constant, 2: trend, 3: both)
% Outputs:
%   beta: k+ll x 1 coefficient vector
%   u: T x 1 residuals

    % --- Dimensions ---
    T = size(x,1);

    % --- Augment Regressors with Deterministic Terms ---
    % Explanation: Add constant and/or trend as specified by det.
    cons = ones(T,1);
    trend = (1:T)';
    if det == 1
        x = [cons x];
    elseif det == 2
        x = [trend x];
    elseif det == 3
        x = [cons trend x];
    end
    % --- OLS Estimation ---
    % Explanation: Solve beta = (x' * x)^{-1} * x' * y and compute residuals
    % u = y - x * beta.
    k = size(x,2);
    xx = eye(k) / (x'*x);
    beta = xx * x' * y;
    u = y - x * beta;
end
