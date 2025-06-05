# Dynamic Factor Models with Time-Varying Components

**Author:** Moka Kaleji • Contact: mohammadkaleji1998@gmail.com

**Affiliation:** Master Thesis in Econometrics: 

Advancing High-Dimensional Factor Models: Integrating Time-Varying 
Parameters with Dynamic Factors, University of Bologna.

This repository contains MATLAB implementations of three advanced dynamic factor models designed to capture evolving structures in high-dimensional time series data. These models are particularly suited for macroeconomic or financial datasets where relationships among variables change over time.

The three models implemented are:

1. **DFTL** – Dynamic Factor model with Time-varying Loadings
2. **DFTLTA** – DFTL with Time-varying Transition Matrix (A)
3. **DFTLTATQ** – DFTLTA extended with a Time-varying State Covariance Matrix (Q)

Each model includes two scripts: one for estimation and one for forecasting. This README provides detailed guidance on each model’s purpose, methodology, inputs, and execution.

---

## 🔹 1. DFTL: Dynamic Factor model with Time-varying Loadings

### Description

The DFTL model estimates latent factors from a high-dimensional dataset using a factor model where **factor loadings vary over time**. This allows the relationship between observed variables and latent factors to evolve smoothly, capturing gradual structural changes.

### Files

* `DFTL_estim.m` – Estimates latent factors and time-varying loadings from input data.
* `DFTL_forecasting.m` – Generates out-of-sample forecasts using estimated components.

### Methodology

* Extracts latent factors via PCA or weighted PCA.
* Applies kernel smoothing or local weighting to estimate time-varying factor loadings.
* Assumes static AR dynamics in the factor process.

### Inputs

* A data matrix `Y` (dimensions: T × N), where T is time and N is the number of variables.
* Optional: tuning parameters for kernel smoothing.

### Outputs

* Estimated latent factors (`Fhat`)
* Time-varying loadings (`Lhat(t)` for each t)
* Forecasts for user-specified horizons

### Usage

```matlab
% Estimate model
DFTL_estim;

% Forecast future observations
DFTL_forecasting;
```

---

## 🔹 2. DFTLTA: DFTL with Time-varying Transition Matrix (A)

### Description

DFTLTA builds on DFTL by introducing **time-varying autoregressive (AR) coefficients** in the latent factor evolution equation. This captures evolving persistence or regime shifts in factor dynamics, making forecasts more adaptive to recent trends.

### Files

* `DFTLTA_estim.m` – Estimates factors, loadings, and time-varying AR coefficients.
* `DFTLTA_forecasting.m` – Forecasts future observations using dynamic factors and AR processes.

### Methodology

* Uses locally weighted regression or kernel smoothing to estimate time-varying AR coefficients.
* Models factor evolution as: `F_t = A(t) * F_{t-1} + noise`
* Jointly estimates time-varying loadings and AR parameters.

### Inputs

* Same data matrix `Y` as in DFTL.
* Additional bandwidth parameters for smoothing AR terms.

### Outputs

* Estimated latent factors (`Fhat`)
* Time-varying loadings (`Lhat(t)`)
* Time-varying AR matrices (`Ahat(t)`)
* Forecasts for selected horizons

### Usage

```matlab
% Estimate dynamic model with time-varying AR
DFTLTA_estim;

% Forecast using AR-evolving factors
DFTLTA_forecasting;
```

---

## 🔹 3. DFTLTATQ: DFTLTA with Time-varying State Covariance Matrix (Q)

### Description

DFTLTATQ is the most comprehensive of the three models. It allows **both autoregressive dynamics and the state disturbance covariance matrix** to vary over time. This captures time-varying uncertainty and volatility in the latent factor process, especially useful in financial or crisis-prone environments.

### Files

* `DFTLTATQ_estim.m` – Estimates the full time-varying system: loadings, AR, and Q matrices.
* `DFTLTATQ_forecasting.m` – Forecasts using all time-varying components for robust prediction.

### Methodology

* Extends the state-space representation to allow Q(t) to change over time.
* Uses nonparametric or local-likelihood methods to estimate Q(t).
* Builds on the DFTLTA structure for AR and loading estimation.

### Inputs

* Same base data matrix `Y`.
* Additional smoothing bandwidths for Q(t).

### Outputs

* Latent factors (`Fhat`)
* Time-varying loadings (`Lhat(t)`)
* Time-varying AR matrices (`Ahat(t)`)
* Time-varying state noise covariance (`Qhat(t)`)
* Forecasts with uncertainty bands (optional)

### Usage

```matlab
% Estimate full dynamic factor model
DFTLTATQ_estim;

% Generate forecasts with evolving uncertainty
DFTLTATQ_forecasting;
```

---

## 📦 Requirements

* MATLAB R2020a or newer
* Statistics and Machine Learning Toolbox (for regression, smoothing, filtering)
* All files should be placed in the same working directory
* Input dataset must be preprocessed (e.g., stationary, cleaned for NAs)

---

## 📁 Repository Structure

```
├── DFTL_estim.m            % Estimation for DFTL
├── DFTL_forecasting.m      % Forecasting for DFTL
├── DFTLTA_estim.m          % Estimation for DFTLTA
├── DFTLTA_forecasting.m    % Forecasting for DFTLTA
├── DFTLTATQ_estim.m        % Estimation for DFTLTATQ
├── DFTLTATQ_forecasting.m  % Forecasting for DFTLTATQ
```

---

## 📚 Citation & Credits

If you use this code in your work, please cite the relevant papers or acknowledge the original authors where applicable. For academic citation format, feel free to modify the example below:

```
@misc{dynamicfactors2025,
  title  = {Time-Varying Dynamic Factor Models},
  author = {Moka Kaleji},
  year   = {2025},
  note   = {GitHub repository: https://github.com/yourusername/dftl-models}
}
```

---

## 📬 Contact

For any questions or issues, please open an issue in this repository or contact the maintainer.

---

## 📝 License

This repository is licensed under the MIT License. See the LICENSE file for details.
