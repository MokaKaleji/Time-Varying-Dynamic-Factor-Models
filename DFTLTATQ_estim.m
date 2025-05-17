%% DFTLTAQ_estim.m
% Author: Moka Kaleji • Contact: mohammadkaleji1998@gmail.com
% Affiliation: Master Thesis in Econometrics: 
% Advancing High-Dimensional Factor Models: Integrating Time-Varying 
% Loadings and Transition Matrix with Dynamic Factors.
% University of Bologna

clear; close all; clc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% QML Dynamic Factor Model With Time-varying Loadings, A, and Q
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main Script for Quasi Maximum Likelihood Dynamic Factor Model with 
% Time-Varying Parameters
% Purpose:
% This script facilitates the estimation of a dynamic factor model with 
% time-varying loadings, transition matrices, and process noise covariances.
% It begins with dataset selection, allowing the user to choose between 
% monthly or quarterly data, specify the training sample size, and standardize
% the data. The processed data is then passed to the DFTLAQ function for 
% model estimation.
% Workflow:
%   1. Dataset selection and loading
%   2. Training sample size specification
%   3. Data standardization
%   4. Model estimation using DFTLAQ.m
% Dependencies:
%   - DFTLAQ.m
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
% This section provides a user-friendly interface to choose between 
% pre-processed monthly or quarterly datasets, ensuring flexibility in 
% periodicity. The training sample size (T_train) is specified to focus on
% a subset of the data, which is useful for in-sample estimation and 
% out-of-sample forecasting. Standardization (zero mean, unit variance) is
% applied to prevent numerical issues and ensure consistent scaling across 
% variables, a common practice in high-dimensional time series modeling.

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
% The QMLDFM_TVLAQ function is called with the standardized training data 
% x_train_norm, along with user-specified or default parameters for the 
% number of factors (R), bandwidths (h, h_A), VAR lag order (p), and EM 
% algorithm settings (max_iter, tol). The model estimates time-varying 
% loadings, factors, and covariances, producing outputs for analysis and 
% forecasting. Here, we set example parameters, but these can be adjusted
% based on the dataset or research objectives.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Setting Model Parameters & Running DFM with TVL&A&Q estimation 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- Set Model Parameters ---
% Explanation: Define parameters for QMLDFM_TVLAQ. These are illustrative 
% values and should be tuned based on the dataset's characteristics 
% (e.g., number of variables, time series dynamics). R=5 assumes a moderate
% number of factors; h=0.1 and h_A=0.1 balance smoothness and flexibility;
% p=2 allows for lagged dynamics; max_iter=1000 and tol=1e-6 ensure convergence.
R = 2;                                                                     % Number of factors
h = 0.13;                                                                  % Bandwidth for LSFM
p = 4;                                                                     % VAR lag order
max_iter = 1000;                                                           % Maximum EM iterations
tol = 1e-4;                                                                % Convergence tolerance
h_A = 0.2;                                                                 % Bandwidth for Ahat and Qhat

% --- Run QMLDFM_TVLAQ ---
% Explanation: Call the main estimation function with the standardized 
% training data and parameters. Outputs include common components, factors,
% loadings, covariances, and the log-likelihood, which can be used for model
% evaluation, forecasting, or diagnostics.
[MKAQ] = QMLDFM_TVLAQ(x_train_norm, R, h, p, max_iter, tol, h_A);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Save Results for Forecasting and Further Analysis 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
save('dftl&A&Q_estim_results.mat', 'R', 'h', 'MKAQ', 'p', ...
     'x_train', 'x_train_norm', 'mean_train', 'std_train', 'T_train', 'N', ...
     'max_iter', 'tol');
disp('DFTL&A&Q estimation complete. Results saved.');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Main Function 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [MKAQ] = QMLDFM_TVLAQ(X, R, h, p, max_iter, tol, h_A)
% Quasi Maximum Likelihood estimation of a dynamic factor model with 
% time-varying parameters
% Purpose:
% Estimates a dynamic factor model where observed data X_t is driven 
% by latent factors f_t with time-varying loadings (Lambda_t), 
% transition matrices (A_t), and process noise covariances (Q_t). 
% The model is:
%     X_t = Lambda_t * f_t + e_t,  e_t ~ N(0, Sigma_e)
%     f_t = A_t,1 * f_{t-1} + ... + A_t,p * f_{t-p} + u_t,  u_t ~ N(0, Q_t)
% Uses the Expectation-Maximization (EM) algorithm with Kalman 
% filtering/smoothing to iteratively estimate parameters and latent states.
% Inputs:
%   X: T x N data matrix of observed time series
%   R: Number of latent factors
%   h: Bandwidth for kernel smoothing in initial locally stationory factor model (LSFM)
%   p: Order of the VAR(p) process for factor dynamics
%   max_iter: Maximum number of EM iterations
%   tol: Convergence tolerance for relative log-likelihood change
%   h_A: Bandwidth for kernel smoothing of time-varying Ahat and Qhat
% Outputs:
%   MKAQ.CChat: T x N common components (Lambda_t * f_t)
%   MKAQ.Fhat: T x R smoothed factors
%   MKAQ.xitT: T x (R*(p+1)) smoothed augmented states
%   MKAQ.Lhat: N x R x T time-varying factor loadings
%   MKAQ.Sigma_e_hat: N x N diagonal idiosyncratic covariance
%   MKAQ.Ahat: T x p cell array of R x R VAR coefficient matrices
%   MKAQ.Ahat_companion: (R*p) x (R*p) companion matrix at t=T
%   MKAQ.Qhat: R x R x T time-varying process noise covariance
%   MKAQ.logL: Final log-likelihood
%   MKAQ.Rhat: N x N diagonal prediction error covariance
%   MKAQ.PtT: (R*(p+1)) x (R*(p+1)) x T smoothed state covariance
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

  % --- Default Bandwidth for Kernel Smoothing ---
  % Purpose: Set default bandwidth for time-varying parameter estimation 
  % if not provided.
  % Explanation: The bandwidth h_A controls the smoothness of Ahat and 
  % Qhat estimates.
  % A smaller h_A allows more time variation but may lead to overfitting;
  % a larger h_A smooths estimates, enhancing stability. Default is 0.1,
  % balancing flexibility and robustness.
  if nargin < 7
      h_A = 0.1;                                                           % Default bandwidth for Ahat and Qhat
  end

  % --- Initial Estimates via Locally Stationary Factor Model (LSFM) ---
  % Purpose: Obtain initial estimates of time-varying loadings, factors,
  % and idiosyncratic covariance using a kernel-based static factor model.
  % Explanation: The lsfm function applies kernel-weighted principal 
  % component analysis (PCA) to estimate Lambda_t (N x R), f_t (R x 1),
  % and Sigma_t (N x N) at each time t. This provides a starting point 
  % for the EM algorithm, leveraging local covariance structures to 
  % capture time-varying relationships. The bandwidth h controls the 
  % smoothness of these initial estimates.
  [~, Fhat_initial, Lhat, Sigmahat] = lsfm(X, R, h);
  Fhat = Fhat_initial;                                                     % T x R initial factors
  Lhat = permute(Lhat, [1, 2, 3]);                                         % Reshape to N x R x T for consistency
  % Initialize Sigma_e_hat as a diagonal matrix using the average 
  % idiosyncratic variance across time, ensuring positive definiteness 
  % and simplicity.
  Sigma_e_hat = diag(mean(diag(squeeze(mean(Sigmahat, 3)))) * ones(N, 1));

  % --- VAR(p) Initialization for Factor Dynamics ---
  % Purpose: Initialize the time-varying VAR(p) coefficients Ahat and 
  % process noise Qhat.
  % Explanation: The ML_VAR function estimates a time-invariant VAR(p) 
  % model on the initial factors Fhat_initial to obtain starting values 
  % for Ahat_t,i (R x R matrices for lags i=1,...,p). These are replicated
  % across all t to initialize Ahat. Qhat is initialized as the covariance
  % of factor differences, approximating the process noise variance.
  [~, ~, AL0] = MK_VAR(Fhat_initial, p, 0);                                % R x R x p initial VAR coefficients
  state_dim = R * p;                                                       % Dimension of companion state vector
  state_dim_aug = R * (p + 1);                                             % Augmented state dimension including current factors
  Ahat = cell(T, p);                                                       % T x p cell array for time-varying coefficients
  for t = 1:T
      for lag = 1:p
          Ahat{t, lag} = AL0(:, :, lag);                                   % Initialize Ahat as constant across time
      end
  end
  % Estimate initial Qhat from the covariance of factor differences
  Qhat_initial = cov(Fhat_initial(p+1:end, :) - Fhat_initial(p:end-1, :)); % R x R
  Qhat = repmat(Qhat_initial, [1, 1, T]);                                  % R x R x T, constant across time initially

  % --- Precompute Kernel Weights for Time-Varying Parameters ---
  % Purpose: Compute Gaussian kernel weights for smoothing Ahat and Qhat 
  % estimates.
  % Explanation: Time-varying parameters are estimated using 
  % kernel-weighted averages over time, with weights determined by a 
  % Gaussian kernel K((u_s - u_t)/h_A). Here, u_t = t/T normalizes time 
  % to [0,1], and h_A controls the smoothing window. Weights are normalized
  % to sum to 1 for each t, ensuring proper averaging.
  u = (1:T)' / T;                                                          % Normalized time index
  weights = cell(T, 1);                                                  
  for t = 1:T
      z = (u - u(t)) / h_A;                                                % Scaled time differences
      weights{t} = (1 / sqrt(2*pi)) * exp(-0.5 * z.^2);                    % Gaussian kernel
      weights{t} = weights{t} / sum(weights{t});                           % Normalize weights
  end

  % --- EM Algorithm ---
  % Purpose: Iteratively estimate latent states and model parameters 
  % using the EM algorithm.
  % Explanation: The EM algorithm alternates between the E-step 
  % (estimating latent states via Kalman filtering/smoothing) and the 
  % M-step (updating parameters Lambda_t, Ahat_t, Qhat_t, and Sigma_e_hat).
  % Convergence is assessed by the relative change in log-likelihood.
  logL_prev = -Inf;                                                        % Initialize previous log-likelihood
  for iter = 1:max_iter
  % ### E-Step: Kalman Filter and Smoother ###
  % Purpose: Estimate latent states and their covariances given 
  % current parameters.
  % Explanation: The Kalman filter computes filtered states (f_{t|t})
  % and covariances (P_{t|t}) forward in time, while the smoother refines
  % these to smoothed states (f_{t|T}) and covariances (P_{t|T}) using all
  % observations. The log-likelihood is accumulated for convergence checking.
  logL = 0;                                                                % Initialize log-likelihood for this iteration
  xitT = zeros(T, state_dim_aug);                                          % T x (R*(p+1)) smoothed states
  Fhat_filt_aug = zeros(T, state_dim_aug);                                 % Filtered augmented states
  P_filt_aug = zeros(state_dim_aug, state_dim_aug, T);                     % Filtered covariances
  Fhat_pred_aug = zeros(T, state_dim_aug);                                 % Predicted states
  P_pred_aug = zeros(state_dim_aug, state_dim_aug, T);                     % Predicted covariances
  Fhat_smooth_aug = zeros(T, state_dim_aug);                               % Temporary smoothed states
  P_smooth_aug = zeros(state_dim_aug, state_dim_aug, T);                   % Smoothed covariances

  % --- Initial State for Kalman Filter ---
  % Purpose: Initialize the state and covariance at t=1 with a diffuse prior.
  % Explanation: The initial state is set using the first p+1 factor 
  % estimates from LSFM, arranged in the augmented state vector 
  % [f_t; f_{t-1}; ...; f_{t-p}]. A large diagonal covariance (I) 
  % reflects high initial uncertainty, ensuring robustness to 
  % misspecification of initial conditions.
  for pp = 1:p+1
          if pp <= T
              Fhat_filt_aug(1, (pp-1)*R+1:pp*R) = Fhat_initial(pp, :);
          end
  end
  P_filt_aug(:, :, 1) = eye(state_dim_aug);                                % Diffuse prior covariance


  % --- Kalman Filter (Forward Pass) ---
  % Purpose: Compute filtered states and covariances forward in time.
  % Explanation: The Kalman filter alternates between prediction 
  % (using the state transition model) and update (incorporating observations).
  % The state vector is augmented to include lagged factors, and the transition
  % matrix A_aug_t is time-varying. The log-likelihood is computed for t > p
  % to avoid issues with initial lags.
  for t = 1:T
          % Construct time-varying augmented transition matrix
          % Explanation: A_aug_t is an (R*(p+1)) x (R*(p+1)) matrix in 
          % companion form, where the first R rows contain Ahat_t,i for 
          % lags i=1,...,p, and subdiagonal blocks are identity matrices
          % to shift lagged states.
          A_aug_t = zeros(state_dim_aug, state_dim_aug);
          for lag = 1:p
              A_aug_t(1:R, (lag-1)*R + 1 : lag*R) = Ahat{t, lag};
          end
          for i = 1:p
              A_aug_t(R*i+1:R*(i+1), R*(i-1)+1:R*i) = eye(R);
          end

          % Time-varying process noise covariance
          % Explanation: Q_aug is an (R*(p+1)) x (R*(p+1)) matrix with 
          % Qhat_t in the top-left R x R block, representing noise for 
          % current factors, and zeros elsewhere.
          Q_aug = zeros(state_dim_aug, state_dim_aug);
          Q_aug(1:R, 1:R) = Qhat(:, :, t);

          % Prediction Step
          % Explanation: Predict the state and covariance at t given t-1:
          %   f_{t|t-1} = A_t * f_{t-1|t-1}
          %   P_{t|t-1} = A_t * P_{t-1|t-1} * A_t' + Q_t
          % For t=1, use the initial state and covariance.
          if t > 1
              Fhat_pred_aug(t, :) = A_aug_t * Fhat_filt_aug(t-1, :)';
              P_pred_aug(:, :, t) = A_aug_t * P_filt_aug(:, :, t-1) * ...
                  A_aug_t' + Q_aug;
          else
              Fhat_pred_aug(t, :) = Fhat_filt_aug(t, :);
              P_pred_aug(:, :, t) = P_filt_aug(:, :, t);
          end

          % Observation Matrix
          % Explanation: H_t = [Lambda_t, 0] maps the augmented state 
          % (current and lagged factors) to observations, where Lambda_t
          % is N x R and the zeros account for lagged factors not directly
          % affecting X_t.
          Lt = squeeze(Lhat(:, :, t));                                     % N x R
          H = [Lt, zeros(N, state_dim_aug - R)];                           % N x (R*(p+1))

          % Update Step
          % Explanation: Update the state and covariance using the observation X_t:
          %   v_t = X_t - H_t * f_{t|t-1} (innovation)
          %   S_t = H_t * P_{t|t-1} * H_t' + Sigma_e (innovation covariance)
          %   K_t = P_{t|t-1} * H_t' * S_t^{-1} (Kalman gain)
          %   f_{t|t} = f_{t|t-1} + K_t * v_t
          %   P_{t|t} = P_{t|t-1} - K_t * H_t * P_{t|t-1}
          y_pred = H * Fhat_pred_aug(t, :)';
          v_t = X(t, :)' - y_pred;                                         % Innovation
          S_t = H * P_pred_aug(:, :, t) * H' + Sigma_e_hat;                % Innovation covariance
          if t > p
          % Compute log-likelihood for t > p to avoid initial lag effects
          % Explanation: The log-likelihood increment is:
          % log p(X_t | X_{1:t-1}) = -N/2 * log(2*pi) - 1/2 * log(det(S_t))
          % - 1/2 * v_t' * S_t^{-1} * v_t. Cholesky decomposition ensures
          % numerical stability for det(S_t) and S_t^{-1}.
              try
                  L = chol(S_t, 'lower');                                  % Cholesky decomposition
                  log_det_S_t = 2 * sum(log(diag(L)));                     % log(det(S_t)) = 2 * sum(log(diag(L)))
                  inv_S_v_t = L' \ (L \ v_t);                              % Solve S_t^{-1} * v_t
                  logL = logL + 0.5 * (-N * log(2*pi) - log_det_S_t - ...
                      v_t' * inv_S_v_t);
              catch
                  warning('S_t not positive definite at t = %d', t);
              end
          end
          K = P_pred_aug(:, :, t) * H' / S_t;                              % Kalman gain
          Fhat_filt_aug(t, :) = Fhat_pred_aug(t, :) + (K * v_t)';
          P_filt_aug(:, :, t) = P_pred_aug(:, :, t) - K * H * ...
              P_pred_aug(:, :, t);
   end

   % --- Kalman Smoother (Backward Pass) ---
   % Purpose: Compute smoothed states and covariances using all observations.
   % Explanation: The smoother refines filtered estimates by incorporating 
   % future observations:
   %   J_t = P_{t|t} * A_{t+1}' * P_{t+1|t}^{-1} (smoothing gain)
   %   f_{t|T} = f_{t|t} + J_t * (f_{t+1|T} - f_{t+1|t})
   %   P_{t|T} = P_{t|t} + J_t * (P_{t+1|T} - P_{t+1|t}) * J_t'
   % A pseudo-inverse is used for P_{t+1|t}^{-1} to handle potential singularity.
   Fhat_smooth_aug(T, :) = Fhat_filt_aug(T, :);                            % Initialize at t=T
   P_smooth_aug(:, :, T) = P_filt_aug(:, :, T);
   for t = T-1:-1:1
          % Construct A_aug_t_next for smoothing
          A_aug_t_next = zeros(state_dim_aug, state_dim_aug);
          for lag = 1:p
              A_aug_t_next(1:R, (lag-1)*R + 1 : lag*R) = Ahat{t+1, lag};
          end
          for i = 1:p
              A_aug_t_next(R*i+1:R*(i+1), R*(i-1)+1:R*i) = eye(R);
          end

          J = P_filt_aug(:, :, t) * A_aug_t_next' * pinv(P_pred_aug...
              (:, :, t+1));
          Fhat_smooth_aug(t, :) = Fhat_filt_aug(t, :) + (J * ...
              (Fhat_smooth_aug(t+1, :) - Fhat_pred_aug(t+1, :))')';
          P_smooth_aug(:, :, t) = P_filt_aug(:, :, t) + J * (P_smooth_aug ...
              (:, :, t+1) - P_pred_aug(:, :, t+1)) * J';
   end

   % Store smoothed states and factors
   xitT = Fhat_smooth_aug;                                                 % T x (R*(p+1)) smoothed states
   Fhat = xitT(:, 1:R);                                                    % T x R smoothed factors

   % --- Compute Residuals for Qhat Update ---
   % Purpose: Calculate residuals for estimating time-varying process 
   % noise covariance.
   % Explanation: Residuals are computed as u_t = f_t - sum(A_t,i * f_{t-i}),
   % where f_t and f_{t-i} are smoothed estimates. 
   % These residuals drive the Qhat update.
   resid = cell(T, 1);
   for t = p+1:T
          Ahat_t = zeros(R, state_dim);
          for lag = 1:p
              Ahat_t(:, (lag-1)*R + 1 : lag*R) = Ahat{t, lag};
          end
          resid{t} = xitT(t, 1:R)' - Ahat_t * xitT(t-1, 1:state_dim)';     % R x 1
   end

   % ### M-Step: Parameter Updates ###
   % Purpose: Update model parameters to maximize the expected log-likelihood.
   % Explanation: Using smoothed states, update Lambda_t, Ahat_t, Qhat_t, 
   % and Sigma_e_hat via kernel-weighted least squares and covariance estimation.

   % --- Update Ahat ---
   % Purpose: Estimate time-varying VAR coefficients Ahat_t.
   % Explanation: Ahat_t is estimated by kernel-weighted least squares:
   % Ahat_t = (sum w_{t,s} * f_s * f_{s-1}') * (sum w_{t,s} * f_{s-1} * 
   % f_{s-1}')^{-1} Stability is enforced by ensuring eigenvalues of the
   % companion matrix have magnitude <= 0.99.
   Ahat_new = cell(T, p);
   max_eigvals_iter = zeros(T, 1);                                         % Track max eigenvalues for summary
   for t = 1:T
          sum_FtFt_lags = zeros(R, state_dim);
          sum_Ft_lagsFt_lags = zeros(state_dim, state_dim);
          for s = p+1:T
              w = weights{t}(s);
              Ft = xitT(s, 1:R)';
              Ft_lags = xitT(s-1, 1:state_dim)';
              sum_FtFt_lags = sum_FtFt_lags + w * (Ft * Ft_lags');
              sum_Ft_lagsFt_lags = sum_Ft_lagsFt_lags + w * ( ...
                  Ft_lags * Ft_lags');
          end
          Ahat_t = sum_FtFt_lags / sum_Ft_lagsFt_lags;                     % Least squares solution
          % Eigenvalue monitoring and stability enforcement
          Ahat_comp_t = zeros(state_dim, state_dim);
          Ahat_comp_t(1:R, :) = Ahat_t;
          for i = 1:p-1
              Ahat_comp_t(R*i+1:R*(i+1), R*(i-1)+1:R*i) = eye(R);
          end
          eigvals = eig(Ahat_comp_t);
          max_eig = max(abs(eigvals));
          max_eigvals_iter(t) = max_eig; % Store for summary

          if any(abs(eigvals) > 0.99)
              % Enforce stability by projecting eigenvalues
              [V, D] = eig(Ahat_comp_t);
              D = diag(min(max(real(diag(D)), -0.99), 0.99));
              Ahat_comp_t = real(V * D * inv(V));
              Ahat_t = Ahat_comp_t(1:R, :);
              eigvals = eig(Ahat_comp_t);
              max_eig = max(abs(eigvals));
              fprintf(['Iteration %d, t=%d: Max |eigenvalue| after ' ...
                  'enforcement: %.6f\n'], iter, t, max_eig);
         end
          
          eigvals_Ahat(t, :) = eigvals.';                                  % Store eigenvalues
          for lag = 1:p
              Ahat_new{t, lag} = Ahat_t(:, (lag-1)*R + 1 : lag*R);
          end
    end
      Ahat = Ahat_new;

    % Print eigenvalue summary for the iteration
    fprintf(['Iteration %d: Mean max |eigenvalue| = %.6f, Max max ' ...
          '|eigenvalue| = %.6f\n'], iter, mean(max_eigvals_iter), ...
          max(max_eigvals_iter));

    % --- Update Qhat ---
    % Purpose: Estimate time-varying process noise covariance Qhat_t.
    % Explanation: Qhat_t is the kernel-weighted average of residual covariances:
    %   Qhat_t = sum(w_{t,s} * u_s * u_s') / sum(w_{t,s})
    % where u_s are the residuals from the factor dynamics.
    for t = 1:T
          sum_resid_sq = zeros(R, R);
          sum_w = 0;
          for s = p+1:T
              w = weights{t}(s);
              if ~isempty(resid{s})
                  sum_resid_sq = sum_resid_sq + w * (resid{s} * resid{s}');
                  sum_w = sum_w + w;
              end
          end
          if sum_w > 0
              Qhat(:, :, t) = sum_resid_sq / sum_w;
          else
              Qhat(:, :, t) = Qhat_initial;                                % Fallback to initial estimate
          end
    end

    % --- Update Sigma_e_hat ---
    % Purpose: Estimate idiosyncratic noise covariance Sigma_e_hat.
    % Explanation: Sigma_e_hat is the average covariance of observation 
    % residuals: e_t = X_t - Lambda_t * f_t
    % Enforce a diagonal structure to reduce parameters and ensure identifiability.
    Sigma_e_new = zeros(N, N);
    for t = 1:T
          Lt = squeeze(Lhat(:, :, t));
          resid_obs = X(t, :)' - Lt * xitT(t, 1:R)';
          Sigma_e_new = Sigma_e_new + resid_obs * resid_obs';
    end
    Sigma_e_hat = diag(diag(Sigma_e_new / T));

    % --- Convergence Check ---
    % Purpose: Assess convergence based on relative log-likelihood change.
    % Explanation: Convergence is reached when:
    %   |logL - logL_prev| / (|logL| + |logL_prev| + 1e-3) / 2 < tol
    % The small constant 1e-3 prevents division by zero.
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

    % --- Compute Common Components ---
    % Purpose: Calculate the common components C_t = Lambda_t * f_t.
    % Explanation: These represent the portion of X_t explained by the 
    % factors, used for forecasting and decomposition.
    CChat = zeros(T, N);
    for t = 1:T
        Lt = squeeze(Lhat(:, :, t));
        CChat(t, :) = (Lt * xitT(t, 1:R)')';
    end

    % --- Construct Companion Matrix at t=T ---
    % Purpose: Form the companion matrix for the final time point’s VAR 
    % coefficients.
    % Explanation: The companion matrix represents the VAR(p) dynamics in 
    % first-order form, used for stability analysis and forecasting.
    Ahat_companion = zeros(state_dim, state_dim);
    for lag = 1:p
        Ahat_companion(1:R, (lag-1)*R + 1 : lag*R) = Ahat{T, lag};
    end
    for i = 1:p-1
        Ahat_companion(R*i+1:R*(i+1), R*(i-1)+1:R*i) = eye(R);
    end

    % --- Construct Augmented Matrix at t=T ---
    % Purpose: Form the augmented matrix for the final time point’s process
    % noise covariance
    % Explanation: The augmented matrix represents the VAR(p) dynamics in 
    % first-order form, used for stability analysis and forecasting.
    Qhat_aug = zeros(state_dim, state_dim);
    for lag = 1:p
        Qhat_aug(1:R, 1:R) = Qhat(:, :, T);
    end
    for i = 1:p-1
        Qhat_aug(R*i+1:R*(i+1), R*(i-1)+1:R*i) = zeros;
    end

    % --- Compute Prediction Error Covariance (Rhat) ---
    % Purpose: Estimate the covariance of prediction errors.
    % Explanation: Rhat includes both idiosyncratic errors and uncertainty 
    % from factor estimates, computed as:
    %   Rhat = sum(eta_t * eta_t' + Lambda_t * P_t * Lambda_t') / T
    % where eta_t = X_t - Lambda_t * f_t, and P_t is the smoothed factor covariance.
    yy = X';                                                               % N x T
    xx = xitT(:, 1:R)';                                                    % R x T
    Rhat = zeros(N, N);
    cc = 0;
    for tt = cc+1:T
        Lt = squeeze(Lhat(:, :, tt));
        eta = yy(:, tt) - Lt * xx(:, tt);
        PtT_tt = P_smooth_aug(1:R, 1:R, tt);
        Rhat = Rhat + eta * eta' + Lt * PtT_tt * Lt';
    end
    Rhat = diag(diag(Rhat / (T - cc)));                                    % Enforce diagonal structure

    % --- Set Smoothed State Covariance ---
    % Purpose: Store the smoothed state covariances for output.
    PtT = P_smooth_aug;                                                    % (R*(p+1)) x (R*(p+1)) x T
    logL = logL_prev;

    % --- Results ---
    % Purpose: Organize outputs into a struct for convenient access.
    % Explanation: The MKAQ struct consolidates all model estimates and 
    % diagnostics, including common components, factors, loadings, covariances,
    % log-likelihood, and eigenvalues of Ahat companion matrices. This 
    % structure facilitates further analysis, forecasting, or reporting, 
    % aligning with standard practices in econometric modeling.
    MKAQ.CChat=CChat;
    MKAQ.Fhat=Fhat;
    MKAQ.xitT=xitT;
    MKAQ.Lhat=Lhat;
    MKAQ.Sigma_e_hat=Sigma_e_hat;
    MKAQ.Ahat=Ahat;
    MKAQ.Ahat_companion=Ahat_companion;
    MKAQ.Qhat=Qhat;
    MKAQ.Qhat_aug=Qhat_aug;
    MKAQ.logL=logL;
    MKAQ.Rhat=Rhat;
    MKAQ.PtT=PtT;
    MKAQ.eigvals_Ahat=eigvals_Ahat;
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