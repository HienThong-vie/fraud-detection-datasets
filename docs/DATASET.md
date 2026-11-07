# PaySim dataset record

- Source: https://www.kaggle.com/datasets/ealaxi/paysim1
- Publisher: Edgar Lopez-Rojas
- Licence: CC BY-SA 4.0
- Format: one CSV containing synthetic mobile-money transactions
- Target: `isFraud`
- Time field: `step`, where one step represents one hour

The model excludes `isFlaggedFraud`, raw account identifiers, and balance-before/balance-after fields.
These controls reduce business-rule leakage, identity memorisation, and post-transaction leakage. Data is
split chronologically by `step`; resampling is used only to cap each already-separated period for a Mac.

Suggested citation: E. A. Lopez-Rojas, A. Elmir, and S. Axelsson, "PaySim: A financial mobile money
simulator for fraud detection," 28th European Modeling and Simulation Symposium, 2016.
