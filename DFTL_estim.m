%% DFTL_estim.m
% Author: Moka Kaleji • Contact: mohammadkaleji1998@gmail.com
% Affiliation: Master Thesis in Econometrics: 
% Advancing High-Dimensional Factor Models: Integrating Time-Varying 
% Loadings and Transition Matrix with Dynamic Factors.
% University of Bologna

clear; close all; clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Quasi Maximum Likelihood Dynamic Factor Model With Time-varying Loadings
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main Script for Quasi Maximum Likelihood Dynamic Factor Model with 
% Time-Varying Loadings
% Purpose:
% This script facilitates the estimation of a dynamic factor model with 
% time-varying loadings.
% It begins with dataset selection, allowing the user to choose between 
% monthly or quarterly data, specify the training sample size, and standardize
% the data. The processed data is then passed to the DFTL function for 
% model estimation.
% Workflow:
%   1. Dataset selection and loading
%   2. Training sample size specification
%   3. Data standardization
%   4. Model estimation using DFTL.m
% Dependencies:
%   - DFTL.m
%   - lsfm.m
%   - MK_VAR.m
%   - MK_ols.m
%   - MK_VAR_companion_matrix
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
%% Running DFM with TVL Estimation 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Purpose: Execute the Quasi Maximum Likelihood estimation of a dynamic 
% factor model with time-varying loadings (TVL) on normalized training data.
% Explanation: This section sets model parameters and runs the QMLDFM_TVL function to
% estimate a dynamic factor model where observed data X_t is driven by latent
% factors f_t with time-varying loadings Lambda_t but time-invariant VAR
% coefficients A. The model is:
% X_t = Lambda_t * f_t + e_t,  e_t ~ N(0, Sigma_e)
% f_t = A_1 * f_{t-1} + ... + A_p * f_{t-p} + u_t,  u_t ~ N(0, Q)
% Parameters are chosen to balance model complexity and computational feasibility.
R=8;                                                                       % Number of factors
h=0.08;                                                                     % Bandwidth for LSFM: Controls smoothness of initial time-varying loadings.
p=4;                                                                       % VAR lags: Captures factor dynamics.
max_iter=1000;                                                             % Maximum EM iterations
tol=1e-6;                                                                  % Convergence tolerance

% Run QMLDFM_TVL to estimate the model on normalized training data.
% Explanation: x_train_norm is assumed to be a T x N matrix of standardized
% data (zero mean, unit variance), which improves numerical stability and
% estimation accuracy in factor models.
[MK] = QMLDFM_TVL(x_train_norm, R, h, p, max_iter, tol);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Save Results for Forecasting and Further Analysis 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Purpose: Save model parameters and estimation results for subsequent 
% analysis or forecasting.
% Explanation: The results are stored in a .mat file, including model 
% parameters (R, h, p, max_iter, tol), estimated outputs (MK struct), and data
% characteristics (x_train, x_train_norm, mean_train, std_train, T_train, N).
% This enables reproducibility and downstream tasks like forecasting or 
% model evaluation.
save('dftl_estim_results.mat', 'R', 'h', 'MK', 'p', ...
     'x_train', 'x_train_norm', 'mean_train', 'std_train', 'T_train', 'N', ...
     'max_iter', 'tol');
disp('DFTL estimation complete. Results saved.');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Main Function 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [MK] = QMLDFM_TVL(X, R, h, p, max_iter, tol)
    % Quasi Maximum Likelihood estimation of a dynamic factor model with 
    % time-varying loadings
    % Purpose:
    % Estimates a dynamic factor model where observed data X_t is driven 
    % by latent factors f_t with time-varying loadings (Lambda_t) and 
    % time-invariant VAR coefficients (A). The model is:
    %     X_t = Lambda_t * f_t + e_t,  e_t ~ N(0, Sigma_e)
    %     f_t = A_1 * f_{t-1} + ... + A_p * f_{t-p} + u_t,  u_t ~ N(0, Q)
    % Uses the Expectation-Maximization (EM) algorithm with Kalman 
    % filtering and smoothing to estimate parameters and latent states. 
    % Inputs:
    %   X: T x N matrix of observed data
    %   R: Number of factors
    %   h: Bandwidth for initial LSFM estimation
    %   p: Number of lags for factor dynamics (VAR(p))
    %   max_iter: Maximum EM iterations
    %   tol: Convergence tolerance
    % Outputs: MK (struct containing):
    %   MK.CChat: T x N common components
    %   MK.Fhat: T x R smoothed factors
    %   MK.xitT: T x (R*(p+1)) smoothed states
    %   MK.Lhat: N x R x T time-varying loadings
    %   MK.Sigma_e_hat: N x N idiosyncratic covariance
    %   MK.Ahat: (R*p) x (R*p) companion matrix
    %   MK.Qhat: (R*p) x (R*p) process noise covariance
    %   MK.logL: Log-likelihood
    %   MK.Rhat: N x N prediction error covariance
    %   MK.PtT: (R*(p+1)) x (R*(p+1)) x T smoothed state covariance
    %   MK.eigvals_Ahat: Eigenvalues of Ahat companion matrix
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
    Sigma_e_hat = diag(mean(diag(squeeze(mean(Sigmahat, 3)))) * ones(N, 1));

    % --- VAR(p) Initialization ---
    % Purpose: Initialize the VAR(p) coefficients Ahat and process noise
    % covariance Qhat.
    % Explanation: The MK_VAR function estimates a VAR(p) model on the 
    % initial factors Fhat_initial to obtain Ahat in companion form 
    % (R*p x R*p) for p > 1, or directly as AL0 (R x R) for p = 1. 
    % Qhat is initialized as a (R*p) x (R*p) matrix, with the top-left R x R
    % block set to the covariance of factor differences, assuming stationarity.
    [~, ~, AL0] = MK_VAR(Fhat_initial, p, 0);
    state_dim = R * p; % Companion state dimension
    state_dim_aug = R * (p + 1); % Augmented state for xitT
    if p > 1
    Ahat = MK_VAR_companion_matrix(AL0);
    else
    Ahat = AL0;
    end
 
    Qhat = zeros(state_dim, state_dim);
    Qhat(1:R, 1:R)=cov(Fhat_initial(p+1:end, :) - Fhat_initial(p:end-1,:));

    %%%%%%%%%%%%%%%%
    % EM Algorithm %
    %%%%%%%%%%%%%%%%
    % Purpose: Iteratively estimate latent states and model parameters using
    % the EM algorithm.
    % Explanation: The EM algorithm alternates between the E-step 
    % (estimating latent states f_t via Kalman filtering and smoothing) and
    % the M-step (updating Ahat, Qhat, and Sigma_e_hat). Lambda_t is fixed 
    % from LSFM, simplifying the M-step. Convergence is assessed by the 
    % relative change in log-likelihood, maximizing the expected complete-data
    % log-likelihood.
    logL_prev = -Inf;
    for iter = 1:max_iter
        % --- E-Step: Kalman Filter and Smoother ---
        % Purpose: Estimate latent states and their covariances given 
        % current parameters.
        % Explanation: The E-step uses Kalman filtering to compute filtered
        % states (f_{t|t}) and covariances (P_{t|t}) forward in time, 
        % followed by Kalman smoothing to compute smoothed states (f_{t|T})
        % and covariances (P_{t|T}). The log-likelihood is accumulated to 
        % monitor convergence.
        logL = 0;
        xitT = zeros(T, state_dim_aug); % T x 9
        Fhat_filt_aug = zeros(T, state_dim_aug);
        P_filt_aug = zeros(state_dim_aug, state_dim_aug, T);
        Fhat_pred_aug = zeros(T, state_dim_aug);
        P_pred_aug = zeros(state_dim_aug, state_dim_aug, T);

        % --- Augmented Transition Matrix ---
        % Purpose: Construct the augmented transition matrix A_aug.
        % Explanation: A_aug is an (R*(p+1)) x (R*(p+1)) matrix in companion
        % form, with the first R rows containing Ahat’s first R rows, and 
        % subdiagonal blocks as identity matrices to shift lagged states.
        % This enables VAR(p) dynamics in a first-order state-space form, 
        % with Ahat time-invariant unlike QMLDFM_TVLA.
        A_aug = zeros(state_dim_aug, state_dim_aug); % 9 x 9
        A_aug(1:R, 1:state_dim) = Ahat(1:R, :); % 3 x 6
        for i = 1:p
            A_aug(R*i+1:R*(i+1), R*(i-1)+1:R*i) = eye(R);                  % Subdiagonal identities
        end

        % --- Process Noise ---
        % Purpose: Define the process noise covariance for the augmented 
        % state.
        % Explanation: Q_aug is an (R*(p+1)) x (R*(p+1)) matrix with Qhat’s
        % top-left R x R block representing noise for current factors, and
        % zeros elsewhere, as lagged factors are deterministic shifts in 
        % the companion form.
        Q_aug = zeros(state_dim_aug, state_dim_aug);
        Q_aug(1:R, 1:R) = Qhat(1:R, 1:R);

        % --- Initial State ---
        % Purpose: Initialize the state and covariance at t=1.
        % Explanation: The initial state is set using the first p+1 factor
        % estimates from LSFM, arranged in the augmented state vector 
        % [f_t; f_{t-1}; ...; f_{t-p}]. The initial covariance is set to the
        % identity matrix, reflecting moderate uncertainty compared to a 
        % diffuse prior, assuming reasonable initial conditions.
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
        % (incorporating observations). The log-likelihood is computed for
        % t > p to account for initial lags, ensuring valid contributions.
        for t = 1:T
            % --- Prediction ---
            % Purpose: Predict the state and covariance at t given t-1.
            % Explanation: The prediction step computes:
            %   f_{t|t-1} = A_aug * f_{t-1|t-1}
            %   P_{t|t-1} = A_aug * P_{t-1|t-1} * A_aug' + Q_aug
            % For t=1, the initial state and covariance are used directly.
            if t > 1
                Fhat_pred_aug(t, :) = A_aug * Fhat_filt_aug(t-1, :)';
                P_pred_aug(:, :, t) = A_aug * P_filt_aug(:, :, t-1) * ...
                    A_aug' + Q_aug;
            else
                Fhat_pred_aug(t, :) = Fhat_filt_aug(t, :);
                P_pred_aug(:, :, t) = P_filt_aug(:, :, t);
            end

            % --- Observation Matrix ---
            % Purpose: Define the observation matrix H_t mapping states to
            % observations.
            % Explanation: H_t = [Lambda_t, 0] maps the augmented state to
            % observations, where Lambda_t is N x R, and zeros account for
            % lagged factors not directly affecting X_t.
            Lt = squeeze(Lhat(:, :, t)); % N x R
            H = [Lt, zeros(N, state_dim_aug - R)]; 

            % --- Update ---
            % Purpose: Update the state and covariance using the observation X_t.
            % Explanation: The update step computes:
            %   v_t = X_t - H_t * f_{t|t-1} (innovation)
            %   S_t = H_t * P_{t|t-1} * H_t' + Sigma_e (innovation covariance)
            %   K_t = P_{t|t-1} * H_t' * S_t^{-1} (Kalman gain)
            %   f_{t|t} = f_{t|t-1} + K_t * v_t
            %   P_{t|t} = P_{t|t-1} - K_t * H_t * P_{t|t-1}
            % Cholesky decomposition ensures numerical stability for 
            % det(S_t) and S_t^{-1}.
            y_pred = H * Fhat_pred_aug(t, :)';
            v_t = X(t, :)' - y_pred;
            S_t = H * P_pred_aug(:, :, t) * H' + Sigma_e_hat;
            if t > p
                try
                    L = chol(S_t, 'lower');
                    log_det_S_t = 2 * sum(log(diag(L)));
                    inv_S_v_t = L' \ (L \ v_t);
                    logL = logL + 0.5 * (-N * log(2*pi) - log_det_S_t - v_t' * inv_S_v_t);
                catch
                    warning('S_t not positive definite at t = %d', t);
                end
            end
            K = P_pred_aug(:, :, t) * H' / S_t;
            Fhat_filt_aug(t, :) = Fhat_pred_aug(t, :) + (K * v_t)';
            P_filt_aug(:, :, t) = P_pred_aug(:, :, t) - K * H * ...
                P_pred_aug(:, :, t);
        end

        % --- Debug: Check Filtered States ---
        % Purpose: Ensure filtered states are non-zero to detect issues.
        % Explanation: A zero Fhat_filt_aug indicates a filtering failure 
        % (e.g., numerical instability), prompting a warning to guide debugging.
        if all(Fhat_filt_aug(:) == 0)
            warning('Fhat_filt_aug is all zeros at iteration %d', iter);
        end

        %%%%%%%%%%%%%%%%%%%
        % Kalman Smoother %
        %%%%%%%%%%%%%%%%%%%
        % Purpose: Compute smoothed states and covariances using all observations.
        % Explanation: The smoother refines filtered estimates:
        %   J_t = P_{t|t} * A_aug' * P_{t+1|t}^{-1} (smoothing gain)
        %   f_{t|T} = f_{t|t} + J_t * (f_{t+1|T} - f_{t+1|t})
        %   P_{t|T} = P_{t|t} + J_t * (P_{t+1|T} - P_{t+1|t}) * J_t'
        % A pseudo-inverse is used for P_{t+1|t}^{-1} to handle potential singularity.
        P_smooth_aug = zeros(state_dim_aug, state_dim_aug, T);
        xitT(T, :) = Fhat_filt_aug(T, :);
        P_smooth_aug(:, :, T) = P_filt_aug(:, :, T);
        for t = T-1:-1:1
            J = P_filt_aug(:, :, t) * A_aug' * pinv(P_pred_aug(:, :, t+1));
            xitT(t, :) = Fhat_filt_aug(t, :) + (xitT(t+1, :) - ...
                Fhat_pred_aug(t+1, :)) * J';
            P_smooth_aug(:, :, t) = P_filt_aug(:, :, t) + J * (P_smooth_aug ...
                (:, :, t+1) - P_pred_aug(:, :, t+1)) * J';
        end

        % --- Debug: Check Smoothed States ---
        % Purpose: Ensure smoothed states are non-zero to detect issues.
        % Explanation: A zero xitT indicates a smoothing failure, prompting
        % a warning.
        if all(xitT(:) == 0)
            warning('xitT is all zeros at iteration %d', iter);
        end
        Fhat = xitT(:, 1:R);

        %%%%%%%%%%
        % M-Step %
        %%%%%%%%%%
        % Purpose: Update model parameters to maximize the expected 
        % log-likelihood.
        % Explanation: Using smoothed states, update Ahat, Qhat, and 
        % Sigma_e_hat via least squares and covariance estimation. Lambda_t
        % is not updated, retaining LSFM estimates.
        sum_FtFt_lags = zeros(R, state_dim); 
        sum_Ft_lagsFt_lags = zeros(state_dim, state_dim); 
        for t = p+1:T
            Ft = xitT(t, 1:R)'; 
            Ft_lags = xitT(t-1, 1:state_dim)'; 
            sum_FtFt_lags = sum_FtFt_lags + Ft * Ft_lags'; 
            sum_Ft_lagsFt_lags = sum_Ft_lagsFt_lags + Ft_lags * Ft_lags'; 
        end
        % --- Update Ahat ---
        % Purpose: Estimate time-invariant VAR coefficients Ahat.
        % Explanation: Ahat is updated by least squares:
        %   Ahat = (sum f_t * f_{t-1}') * (sum f_{t-1} * f_{t-1}')^{-1}
        % Stability is not explicitly enforced, unlike QMLDFM_TVLA, 
        % assuming data-driven stability. Eigenvalues are tracked for diagnostics.
        Ahat_new = sum_FtFt_lags / sum_Ft_lagsFt_lags;
            eigvals = eig(Ahat);
            max_eig = max(abs(eigvals));
            max_eigvals_iter(t) = max_eig; % Store for summary
            eigvals_Ahat(t,:) = eigvals.'; % Store eigenvalues
   
        Ahat(1:R, :) = Ahat_new; 

        % --- Print Eigenvalue Summary ---
        % Purpose: Provide eigenvalue stability diagnostics.
        % Explanation: Reports mean and maximum eigenvalue magnitudes, 
        % aiding in assessing VAR stability.
        fprintf('Iteration %d: Mean max |eigenvalue| = %.6f, Max max |eigenvalue| = %.6f\n', ...
            iter, mean(max_eigvals_iter), max(max_eigvals_iter));

        % --- Update Qhat ---
        % Purpose: Estimate process noise covariance Qhat.
        % Explanation: Qhat’s top-left R x R block is the average covariance
        % of residuals u_t = f_t - Ahat * f_{t-1}, computed across t=p+1,...,T.
        Qhat_new = zeros(R, R);
        for t = p+1:T
            resid = xitT(t, 1:R)' - Ahat(1:R, :) * xitT(t-1, 1:state_dim)'; 
            Qhat_new = Qhat_new + resid * resid';
        end
        Qhat(1:R, 1:R) = Qhat_new / (T);
        
        % --- Update Sigma_e_hat ---
        % Purpose: Estimate idiosyncratic noise covariance Sigma_e_hat.
        % Explanation: Sigma_e_hat is the average covariance of residuals 
        % e_t = X_t - Lambda_t * f_t, enforced as diagonal for identifiability.
        Sigma_e_new = zeros(N, N);
        for t = 1:T
            Lt = squeeze(Lhat(:, :, t));
            resid = X(t, :)' - Lt * xitT(t, 1:R)';
            Sigma_e_new = Sigma_e_new + resid * resid';
        end
        Sigma_e_hat = diag(diag(Sigma_e_new / T));

        % --- Convergence Check ---
        % Purpose: Assess convergence based on relative log-likelihood change.
        % Explanation: Convergence is reached when:
        %   |logL - logL_prev| / (|logL| + |logL_prev| + 1e-3) / 2 < tol
        % The constant 1e-3 prevents division by zero.
        if iter > 1
            rel_change = abs(logL - logL_prev) / (abs(logL) + abs(logL_prev) ...
                + 1e-3) / 2;
            fprintf('Iteration %d: logL = %.4f, rel_change = %.4e\n', iter, ...
                logL, rel_change);
            if rel_change < tol
                fprintf('Converged at iteration %d\n', iter);
                break;
            end
        end
        logL_prev = logL;
    end

    % --- Common Components ---
    % Purpose: Calculate common components C_t = Lambda_t * f_t.
    % Explanation: These represent the portion of X_t explained by the 
    % factors, useful for forecasting or decomposition.
    CChat = zeros(T, N);
    for t = 1:T
        Lt = squeeze(Lhat(:, :, t));
        CChat(t, :) = (Lt * xitT(t, 1:R)')';
    end

    % --- Compute Rhat (Prediction Error Covariance) ---
    % Purpose: Estimate the covariance of prediction errors.
    % Explanation: Rhat includes idiosyncratic errors and factor uncertainty:
    %   Rhat = sum(eta_t * eta_t' + Lambda_t * P_t * Lambda_t') / (T-cc)
    % The choice of cc=0 may assume all observations are used; typically, cc=p.
    yy = X'; % N x T
    xx = xitT(:, 1:R)'; % R x T (factors only)
    Rhat = zeros(N, N);
    cc = 0; % Assuming cc = 0, adjust if different
    for tt = cc+1:T
        Lt = squeeze(Lhat(:, :, tt)); % N x R
        eta = yy(:, tt) - Lt * xx(:, tt); % N x 1
        PtT_tt = P_smooth_aug(1:R, 1:R, tt); % Covariance of factors at t
        Rhat = Rhat + eta * eta' + Lt * PtT_tt * Lt';
    end
    Rhat = diag(diag(Rhat / (T - cc))); % Enforce diagonal structure

    % --- Set PtT ---
    % Purpose: Store smoothed state covariances for output.
    % Explanation: PtT contains P_{t|T} for the augmented state, useful for
    % uncertainty quantification.
    PtT = P_smooth_aug; % (R*(p+1)) x (R*(p+1)) x T
    logL = logL_prev;

    % --- Results ---
    % Purpose: Organize outputs into a struct.
    % Explanation: The MK struct consolidates estimates and diagnostics for
    % further analysis.
    MK.CChat=CChat;
    MK.Fhat=Fhat;
    MK.xitT=xitT;
    MK.Lhat=Lhat; 
    MK.Sigma_e_hat=Sigma_e_hat;
    MK.Ahat=Ahat;
    MK.Qhat=Qhat;
    MK.Rhat=Rhat;
    MK.PtT=PtT;
    MK.logL=logL;
    MK.eigvals_Ahat=eigvals_Ahat;
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
function [CChat, Fhat, Lhat, Sigmahat] = lsfm(X, R, h)
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
                Sigmahat(j,i,t) = Sigmahat(i,j,t);
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
    % and reshape into R x R x k array for compatibility with DFTL.
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
% Explanation: Estimates coefficients beta via OLS, supporting constant, trend, or both.
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MK_VAR_companion_matrix %
%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Purpose: Construct the companion matrix for a VAR(p) model.
% Explanation: Converts VAR coefficients into a first-order companion 
% matrix for state-space representation.
% Reference: 
% - Lütkepohl (2005) for companion matrix in VAR models.
% - Matteo Barigozzi & Matteo Luciani, 2024.           
function PHI = MK_VAR_companion_matrix(A)
    s = size(A);
    if length(s) == 3
        R = s(1);
        p = s(3);
        state_dim = R * p;
        PHI = zeros(state_dim, state_dim);
        for i = 1:p
            PHI(1:R, (i-1)*R+1:i*R) = A(:,:,i);
        end
        for i = 1:p-1
            PHI(R*i+1:R*(i+1), R*(i-1)+1:R*i) = eye(R);
        end
    else
        error('Input A must be an R x R x p array');
    end
end
