
# Predicting Bankruptcy in US Public Companies with Machine Learning (1999–2018)

An early-warning system that uses annual financial statement data to flag US public companies at risk of bankruptcy, benchmarked against the classic Altman Z-score.

## Overview

Corporate failure rarely happens overnight. It is usually preceded by years of weakening liquidity, falling profitability and growing debt. This project tests whether multi-year financial statement data can identify those firms in advance, using supervised machine learning on a rare-event classification problem.

**Business objective:** build a classifier that uses financial ratios to identify companies at high risk of bankruptcy, so lenders, auditors, investors and risk managers can act before failure.

### Business questions

1. Which financial metrics are the strongest early predictors of bankruptcy?
2. Can active and bankrupt companies be told apart from their annual financials?
3. What financial warning signs precede bankruptcy?
4. How early can a model forecast distress?
5. What classification threshold balances false positives against false negatives?

## Dataset

- **File:** `american_bankruptcy.csv` (add your source link here)
- **Size:** 78,682 company-year rows, 8,971 unique companies, 1999–2018
- **Target:** `status_label` (`alive` / `failed`), with 73,462 alive rows and 5,220 failed rows (6.6%)
- **Features:** 18 raw financial variables (`X1`–`X18`), renamed to readable names during cleaning, e.g. `current_assets`, `net_income`, `total_assets`, `long_term_debt`, `market_value`, `retained_earnings`

## Methodology

### 1. Data cleaning
- No missing values and no duplicate rows found
- Removed one redundant column (`X16`, identical to `X9`)
- Renamed `X1`–`X18` to descriptive financial names
- Checked negative values; implausible negatives in fields such as sales, current assets and long-term debt were identified and converted to `NaN`

### 2. Exploratory data analysis
- **Class imbalance:** failed firms are only 6.6% of rows
- **Failed share over time:** rises from about 7.2% in 1999 to a peak of about 9.4% in 2003, then declines to about 1.3% by 2018
- **Size vs. sales:** net sales and total assets are strongly related (Spearman 0.94), and failed and alive firms overlap almost completely by size, so size alone does not separate them
- **Correlations:** size and balance-sheet variables are highly correlated; profit measures are the least tied to size

### 3. Feature engineering
36 model features in total, including:
- **Liquidity:** current ratio, quick ratio, receivables to assets
- **Leverage and solvency:** debt to assets, long-term debt to assets, liabilities to equity
- **Profitability and efficiency:** ROA, EBITDA / EBIT / gross margins, asset turnover
- **Altman Z-score components:** retained earnings / assets, EBIT / assets, market equity / liabilities
- **Temporal trends:** year-over-year growth in net income, liabilities and assets, computed per company

### 4. Modelling
Three models, plus the Altman Z-score (1968) as a non-trained benchmark:

| Model | Class-imbalance handling |
|---|---|
| Logistic Regression | `class_weight="balanced"` |
| Random Forest | `class_weight="balanced_subsample"` |
| Gradient Boosting (`HistGradientBoostingClassifier`) | class weight via `scale_pos_weight` |

Two prediction targets:

| Target | Definition | Positive share |
|---|---|---|
| `as_labelled` | Every year of a failed company = 1 | 6.6% |
| `within_2y` | Only the last 2 observed years of a failed company = 1 (early warning) | 1.5% |

**Validation:** 5-fold `StratifiedGroupKFold` grouped by company, so no firm appears in both train and test. All scores are out-of-fold. Because the classes are so imbalanced, PR-AUC is used alongside ROC-AUC, and classification thresholds are chosen at the best F1 on out-of-fold predictions.

## Results

| Target | Model | ROC-AUC | PR-AUC | Precision | Recall |
|---|---|---|---|---|---|
| As labelled | Logistic Regression | 0.649 | 0.121 | 0.125 | 0.405 |
| | **Random Forest** | **0.720** | **0.161** | 0.177 | 0.367 |
| | Gradient Boosting | 0.709 | 0.160 | 0.172 | 0.376 |
| | Altman Z (benchmark) | 0.620 | 0.091 | 0.107 | 0.469 |
| Within 2 years | Logistic Regression | 0.786 | 0.062 | 0.104 | 0.209 |
| | **Random Forest** | **0.851** | 0.110 | 0.148 | 0.307 |
| | Gradient Boosting | 0.848 | **0.114** | 0.140 | 0.328 |
| | Altman Z (benchmark) | 0.745 | 0.034 | 0.041 | 0.594 |

### Key takeaways
- **Beats the Altman Z-score:** tree models reach roughly 1.8x the benchmark's PR-AUC on the original target and about 3.4x on the 2-year target.
- **Near-term distress is easier to detect:** ROC-AUC rises from about 0.72 to about 0.85 on the 2-year target, meaning a soon-to-fail company is ranked above a healthy one about 85% of the time.
- **Stable results:** fold-to-fold standard deviations are small (ROC-AUC std 0.008–0.018 for the tree models).
- **Top predictors** (permutation importance on PR-AUC):
  1. `market_equity_to_liabilities` (first for both targets)
  2. `debt_to_assets`
  3. `net_income` / `roa`
  4. `market_value`
- **Warning signs:** high leverage, eroding equity value relative to obligations, and shrinking profitability.
- **Operating point:** at the best-F1 threshold, roughly 1 in 6 flagged company-years is a true failure and about a third of failures are caught.

## Limitations
- **Low precision:** at these thresholds most flags are false alarms, so the model suits screening and prioritisation rather than automated decisions.
- **Optimistic thresholds:** thresholds were tuned on the same out-of-fold predictions that are scored, so precision and recall are slightly optimistic.
- **Accounting features only:** no macroeconomic, industry or regulatory variables are included, so performance may degrade in unprecedented downturns.
- **Missing values:** the engineered dataset fills missing values and undefined ratios (from division by zero) with 0, which is a simplification.

## Recommendations and future work
- Score portfolios on a regular schedule and monitor for data drift, especially in `market_equity_to_liabilities`.
- Tune the decision threshold to risk appetite: lower it to maximise recall, raise it to maximise precision.
- Add macroeconomic and sector variables.
- Try sequence or panel-data models (e.g. LSTM) to capture how financial health deteriorates over time.

