Run `python train.py --dataset <id>` or `python train.py --all` from the suite root. Models and reports
are written into `checkpoint/<id>/` and `evaluation/<id>/`. See the main README for experiment policy.

The default is now the [AP / recall / FPR v2 experiment](PRIORITY_SEARCH.md). It writes separate
models and charts under `evaluation/<id>/priority_search_v2/`, not over production checkpoints.
Use `--experiment supervised` to repeat v1. Do not launch a second primary queue while one is
already running; use the documented bounded supplemental queue if needed.
