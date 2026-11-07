# Fraud Detection Datasets

One Git repository with eight independent dataset branches. `main` contains this navigation guide.
Each dataset branch contains its complete Python/PyCharm project, model checkpoints, configuration,
tests, evaluation charts and reports. The datasets are not merged into one CSV or classifier.

| Branch | Dataset | Source project |
|---|---|---|
| `baf` | Bank Account Fraud Base + Variants I–V | ai4cyber-fraud |
| `ieee` | IEEE-CIS Fraud Detection | ieee-fraud-ml |
| `paysim` | PaySim mobile money | paysim-fraud-ml |
| `handbook` | Fraud Detection Handbook simulator | fraud-handbook-ml |
| `sparkov` | Sparkov synthetic card transactions | sparkov-fraud-ml |
| `banksim` | BankSim payments | banksim-fraud-ml |
| `ulb` | ULB/Worldline credit-card transactions | ulb-creditcard-fraud-ml |
| `elliptic` | Elliptic Bitcoin AML | elliptic-bitcoin-aml-ml |

Open this repository in PyCharm. Choose a branch using the Git branch selector, or use:

```bash
git switch paysim
```

Then follow the README on that branch to set up its environment and run it. Save or commit work before
switching branches. Each branch keeps its own complete project; do not merge dataset branches into main.

For teammates working on multiple datasets at once, use separate worktrees to avoid mixing data and
Python environments between branches:

```bash
git worktree add ../fraud-paysim paysim
git worktree add ../fraud-ieee ieee
```

These are source snapshots of the existing boosted-model-comparison commits; the original repositories
and their full histories remain available. Each branch records its source commit in BRANCH_SOURCE.json.
Current checkpoints remain distinct from the exploratory balanced-experiment checkpoint. Existing
dataset source, licence, attribution and academic-assistance notes are retained in each project README.
Raw datasets and virtual environments are local inputs, not part of Git. Supply the matching dataset
on each branch before retraining. Do not share virtual environments across branches.

No GitHub remote is configured yet. Nothing in this repository has been pushed or merged.
