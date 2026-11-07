# BAF dataset provenance and licence record

## Acquisition record

| Field | Recorded value |
| --- | --- |
| Dataset | Bank Account Fraud Dataset Suite (NeurIPS 2022) |
| Kaggle identifier | `sgpjesus/bank-account-fraud-dataset-neurips-2022` |
| Kaggle version | 2 |
| Version date | 29 November 2023, 11:32:12 UTC |
| Version note | Remove relationship between index and label |
| Dataset licence | CC BY-NC-SA 4.0 |
| Dataset URL | <https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022> |
| Publisher project | <https://github.com/feedzai/bank-account-fraud> |
| Dataset paper | <https://arxiv.org/abs/2211.13358> |

The Kaggle API metadata was checked on 29 October 2026. The data licence above applies to the Kaggle
dataset distribution. Feedzai's GitHub repository contains a separate Apache 2.0 licence for repository
code; it must not be substituted for the dataset licence. The checked fields are preserved in
`docs/kaggle_dataset_metadata.json`.

Raw CSV files are not redistributed in this repository. `outputs/profiling/profile_manifest.json`
records the SHA-256 digest, byte size, schema, dtypes, missing-code counts, and profile for every local
BAF file used in the experiment.

The official datasheet defines `customer_age` as age rounded to decade and uses the groups below 50 and
50 or over for its age fairness study. It also states that every negative value in
`intended_balcon_amount` represents missing data. The preprocessing and fairness reports implement those
definitions directly.

## Required files

- `Base.csv`
- `Variant I.csv`
- `Variant II.csv`
- `Variant III.csv`
- `Variant IV.csv`
- `Variant V.csv`

## Variant relevance

| Dataset | Difference from Base | Project relevance |
| --- | --- | --- |
| Base | Reference sample without intentionally induced sampling bias | Main performance baseline and production prototype model |
| Variant I | Group-size disparity | Tests sensitivity to unequal protected-group representation |
| Variant II | Fraud-prevalence disparity | Tests performance and fairness under different group base rates |
| Variant III | Group-dependent class separability | Tests whether fraud is easier to identify for one group |
| Variant IV | Temporal prevalence disparity | Tests robustness when group fraud prevalence changes over time |
| Variant V | Temporal class-separability disparity | Tests robustness when predictive difficulty changes over time |

The variants are trained and reported separately. Combining them would hide the bias condition that each
variant is designed to test. Variants III and V contain the synthetic columns `x1` and `x2`, which induce
separability bias. They are recorded during profiling but excluded from predictive inputs to prevent an
artificial shortcut from dominating the fraud model.

## Canvas course archive

The task names the Basic Dataset archive from Canvas as `Basic_Datasets.zip`. On 29 October 2026, the team
supplied that required archive under its downloaded filename, `Assignment Datasets.zip`. The filename
difference is recorded explicitly instead of silently renaming the evidence. ZIP integrity testing passed,
and the archive-level acquisition record is:

| Task reference | Supplied file | Date recorded | Size (bytes) | SHA-256 | Storage |
| --- | --- | --- | ---: | --- | --- |
| `Basic_Datasets.zip` | `Assignment Datasets.zip` | 29 October 2026 | 4,754,112 | `c69ea56a433bcf7db86805b26b62450fad268f7ce36b7df7bfe0c29e3022a4a6` | Team member's local Downloads folder; not committed |

This Basic Dataset archive contains five classroom dataset groups: misinformation, malware-feature classification,
spam email, software vulnerabilities, and network intrusion. It contains no BAF Base or Variant CSVs.
Therefore, it is retained as evidence that the separate course archive was collected; it is not used as a
substitute for the selected BAF fraud dataset and is not an input to this project's training pipeline.
The machine-readable record is preserved in `docs/canvas_dataset_archive.json`. The archive itself remains
outside Git because no redistribution licence was included at its root and its origin is course-controlled.

## Citation

```bibtex
@article{jesusTurningTablesBiased2022,
  title   = {Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation},
  author  = {Jesus, S\'ergio and Pombal, Jos\'e and Alves, Duarte and Cruz, Andr\'e and
             Saleiro, Pedro and Ribeiro, Rita P. and Gama, Jo\~ao and Bizarro, Pedro},
  journal = {Advances in Neural Information Processing Systems},
  year    = {2022}
}
```

CC BY-NC-SA 4.0 requires attribution, non-commercial use, and compatible share-alike terms when covered
material is shared or adapted. Confirm any planned redistribution with the tutor or dataset owner.
