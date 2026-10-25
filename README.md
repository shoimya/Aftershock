<h1 align="center">Aftershock v1: Comparing Linear Regression and Gradient-Boosted Trees for Volatility Forecasting</h1>

<table align="center">
  <tr>
    <td align="center" width="50%">
      <strong>Shoimya Chowdhury</strong><br>
      <code>chowdh83@umd.edu</code>
    </td>
    <td align="center" width="50%">
      <strong>Madhura Gujar</strong><br>
      <code>madhuragujar16@gmail.com</code>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <strong>Yash Thakare</strong><br>
      <code>yashlthakare@gmail.com</code>
    </td>
    <td align="center" width="50%">
      <strong>Yashika Bhoge</strong><br>
      <code>yab1998am@gmail.com</code>
    </td>
  </tr>
</table>

**Research notebook:** [`aftershock_v1.ipynb`](aftershock_v1.ipynb)

## Abstract

Aftershock v1 investigates whether a more flexible machine learning model improves short-term volatility forecasting relative to a structured linear regression model. Using daily adjusted SPY prices, the project constructs historical volatility features and forecasts annualized volatility over the next five trading days. Three approaches are compared: a persistence baseline, a heterogeneous autoregressive (HAR) regression, and a LightGBM gradient-boosted tree model. Evaluation follows an expanding-window, annual walk-forward design with a gap that prevents future target information from entering training. The study focuses on the relationship between model complexity, feature design, and predictive performance, while identifying questions for further research.

**Keywords:** volatility forecasting, SPY, HAR, linear regression, LightGBM, walk-forward evaluation

## 1. Introduction

Volatility forecasting estimates the magnitude of future price fluctuations rather than their direction. Aftershock v1 examines this problem through a comparison of simple and more complex forecasting methods. Its central research question is whether the nonlinear relationships available to a gradient-boosted tree model provide an advantage over a linear model built around historical volatility at multiple time scales.

HAR combines daily, weekly, and monthly volatility measures through linear regression. LightGBM can represent nonlinear relationships and interactions through an ensemble of decision trees. Comparing these approaches provides a practical setting for studying whether additional flexibility improves forecasting when the available inputs are limited to recent returns and volatility.

The project has three objectives:

1. Construct features and a future volatility target using information aligned with the forecast date.
2. Compare persistence, linear regression, and gradient-boosted trees under a consistent chronological evaluation design.
3. Examine possible explanations for differences in performance and identify extensions that could test those explanations.

## 2. Dataset

The dataset consists of daily adjusted closing prices for SPY, downloaded through `yfinance` with `auto_adjust=True`. The recorded snapshot spans January 29, 1993, through October 2, 2026, and contains 8,477 prices. The cleaned price series contains no missing values.

After constructing the historical features and future target, incomplete observations are removed, leaving 8,450 complete modeling rows. The evaluation covers forecast dates from January 3, 2005, through September 25, 2026. The 2026 test period is partial. The final five price dates cannot be scored because their complete future five-day targets are unavailable in this snapshot.

This study uses a single instrument and daily observations. Its volatility measure is constructed from squared daily returns; it is not an intraday realized-volatility measure. Historical downloads may change if the data provider revises adjusted prices.

## 3. Methodology

### 3.1 Returns and prediction target

For adjusted closing price $P_t$, the daily log return is:

$$
r_t = \ln\left(\frac{P_t}{P_{t-1}}\right).
$$

The target is annualized volatility over exactly the next five trading returns:

$$
\sigma_{t,\mathrm{future}} = \sqrt{\frac{252}{5}\sum_{j=1}^{5}r_{t+j}^{2}}.
$$

The models predict its natural logarithm, $y_t = \ln(\sigma_{t,\mathrm{future}})$. Predictions are converted back to annualized percentage units using $100\exp(\hat{y}_t)$ before scoring.

### 3.2 Feature construction

Historical annualized volatility is calculated over windows of 1, 5, and 22 trading days:

$$
v_{t,k} = \sqrt{\frac{252}{k}\sum_{j=0}^{k-1}r_{t-j}^{2}}, \qquad k \in \{1,5,22\}.
$$

Each volatility input is floored at 0.0001 before its logarithm is taken. The resulting features are:

| Feature | Definition | Purpose |
|---|---|---|
| Daily log volatility | $\ln(\max(v_{t,1},0.0001))$ | Captures the most recent return magnitude |
| Weekly log volatility | $\ln(\max(v_{t,5},0.0001))$ | Summarizes recent short-term variation |
| Monthly log volatility | $\ln(\max(v_{t,22},0.0001))$ | Represents a longer recent volatility window |
| Daily log return | $r_t$ | Preserves the sign and magnitude of the latest return |

All inputs include information available through the forecast day's close. Forecasts therefore assume that this closing price has already been observed.

### 3.3 Forecasting models

**Naive persistence baseline.** The baseline repeats the latest five-day log-volatility value as the forecast. It establishes whether model training adds value beyond carrying forward a recent estimate.

**HAR regression.** The HAR model uses the three log-volatility features in a linear regression:

$$
\hat{y}_t = \beta_0 + \beta_1 x_{t,1} + \beta_5 x_{t,5} + \beta_{22} x_{t,22}.
$$

Here, $x_{t,k}$ denotes the corresponding floored log-volatility input. HAR is linear in these transformed features; with three predictors, its fitted relationship is a hyperplane rather than a single two-dimensional straight line.

**LightGBM.** The gradient-boosted tree model uses all four features. It combines multiple decision trees to represent nonlinear patterns and feature interactions. The recorded configuration uses 300 estimators, a learning rate of 0.03, 15 leaves, and a random seed of 42. These settings remain fixed throughout the evaluation and were not adjusted to improve the observed test scores.

Because LightGBM includes the daily return while HAR does not, the comparison reflects both model choice and a difference in feature sets. A controlled comparison using identical inputs is a natural extension.

### 3.4 Walk-forward evaluation and leakage prevention

Models are retrained annually using an expanding history of earlier observations, beginning with the 2005 test year and continuing through the partial 2026 period. Fitted model parameters remain fixed within each test year, while daily forecasts use newly available inputs.

A five-row gap is applied before each test period. The evaluation also checks that the latest training target ends before the first test date. This matters because a training row's target depends on five subsequent trading returns: separating rows by date alone is insufficient if their target windows extend into the test period.

Chronological evaluation preserves the order in which information becomes available and avoids a random train–test split that would mix earlier and later market conditions.

### 3.5 Evaluation metric

Forecast accuracy is assessed using mean absolute error (MAE) after conversion to annualized percentage units:

$$
\mathrm{MAE} = \frac{1}{N}\sum_{t=1}^{N}\left|100\sigma_{t,\mathrm{future}} - 100\exp(\hat{y}_t)\right|.
$$

Lower MAE indicates a smaller average absolute forecast error. The notebook also includes annual error comparisons and forecast plots to support examination of performance across time and during the February–May 2020 period. Numerical findings and rankings are omitted from this README.

## 4. Discussion: Why Might Linear Regression Outperform Trees?

The central observation motivating discussion is that LightGBM did not outperform HAR in this historical evaluation. Greater model complexity did not automatically translate into better forecasts. This observation raises questions about the information contained in the features, the stability of the relationships being learned, and the design of the comparison.

**Feature structure may favor HAR.** The inputs already summarize volatility across daily, weekly, and monthly windows. If the useful predictive relationship is adequately represented by a weighted combination of these measures, nonlinear splits may add little additional information. This is a possible explanation, not a relationship established by the current experiment.

**Flexibility may introduce estimation variability.** A tree ensemble can fit more detailed patterns than a linear regression, but some patterns may be specific to the training period. HAR's more constrained form may generalize better when additional complexity captures noise. The current comparison does not, by itself, demonstrate that LightGBM overfit.

**Changing market conditions may affect the models differently.** Annual retraining leaves each model unchanged within the test year. If the relationship between recent and future volatility shifts, a model trained on earlier conditions may adapt poorly. Testing shorter retraining intervals and rolling training windows could help evaluate this explanation.

**The configuration and objective also matter.** The study evaluates one fixed LightGBM configuration and one forecast horizon. It therefore cannot establish the best attainable performance of tree-based methods. In addition, modeling log volatility and evaluating errors after exponentiation introduces a distinction between the modeled scale and the scoring scale that deserves further examination.

These explanations remain hypotheses. The reported difference was small, and no statistical significance test was conducted. The comparison supports further investigation of when simpler models are sufficient; it does not establish that regression is universally superior to gradient-boosted trees.

## 5. Limitations

- **Limited scope:** The study considers one instrument, one five-day horizon, and one primary scoring metric.
- **Volatility proxy:** Squared daily returns provide a different measurement basis from intraday realized volatility.
- **Overlapping targets:** Consecutive five-day targets share returns, creating dependence that matters for statistical comparisons.
- **Unequal feature sets:** LightGBM receives an additional return feature, so the experiment does not isolate model architecture alone.
- **Fixed configuration:** The experiment does not include a separate chronological hyperparameter-selection procedure.
- **Uncertainty:** No significance test establishes whether the performance difference is robust to sampling variation.
- **Reproducibility:** Revised price histories and changed software versions may affect future runs.

## 6. Reproducibility

The analysis is implemented in `aftershock_v1.ipynb`. The recorded software environment is:

| Package | Version |
|---|---|
| pandas | 2.2.3 |
| numpy | 2.1.3 |
| scikit-learn | 1.6.1 |
| lightgbm | 4.6.0 |
| yfinance | 0.2.66 |

To reproduce the workflow, use a Python notebook environment with these package versions and the plotting dependencies imported by the notebook, then run the notebook from the first cell onward. Internet access is required for the price download. Preserve the data snapshot and software environment when comparing repeated runs. This README documents the supplied project report; it does not represent an independent rerun of the analysis.

## 7. Open Questions and Possible Extensions

1. **Does the difference persist with identical inputs?** Train HAR and LightGBM on the same feature sets, then add the daily return to both to separate feature effects from model effects.
2. **Is the performance difference statistically distinguishable?** Apply a forecast-comparison method or resampling procedure that accounts for dependence from overlapping targets.
3. **Can chronological tuning improve LightGBM?** Select hyperparameters using validation periods contained entirely within each training window, preserving the final test periods for evaluation.
4. **Would more frequent retraining help?** Compare annual retraining with monthly or quarterly updates and test expanding versus rolling training windows.
5. **Does the relationship hold across assets and horizons?** Extend the analysis to other instruments and prediction windows to assess how broadly the comparison applies.
6. **Would additional information improve forecasts?** Evaluate candidate inputs such as trading volume, price ranges, and market-wide indicators, ensuring each is available at the forecast time.
7. **How sensitive is the comparison to the target and scoring method?** Examine alternative volatility measures, additional error metrics, and the consequences of predicting on the log scale.
8. **Can the models complement each other?** Test whether a combination of HAR and LightGBM forecasts improves performance on untouched future evaluation periods.

These extensions leave the broader research question open: under which data, feature, and evaluation conditions does added model complexity provide a reliable forecasting advantage?
