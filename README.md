# climate_impact_gradboost

fat-tail-climate-risk/
│
├── README.md
├── requirements.txt
├── src/
│   └── quantile_xgb_climate_risk.py
└── results/
    ├── figures/
    └── tables/


## Climate Event Losses under Fat Tails

This project models economic losses from climate events using
monotone-constrained XGBoost and quantile regression (Q50/Q90/Q95).

Key ideas:
- Heavy-tailed loss distributions
- Quantile-based risk (tail focus, not mean)
- Scenario stress-testing
- Time-based train/validation/test split

The goal is not point prediction accuracy, but robust risk estimation
under uncertainty.
