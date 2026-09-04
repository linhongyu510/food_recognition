# Contributing

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

## Before opening a pull request

```bash
ruff check src tests scripts
pytest -q

# End-to-end check
python scripts/make_sample_data.py --output data/sample
food-recognition-train --config configs/smoke_test.yaml
```

CI runs the same steps on Python 3.10 / 3.11 / 3.12.

## Conventions

- **Test behaviour, not mocks.** Training tests run the real loop on generated
  data. If a test would still pass with a broken training step, it is not
  testing enough.
- **Assert against known values.** Metrics tests compare against hand-computed
  precision/recall/F1 rather than the implementation's own output.
- **Add a regression test for every bug fix**, with a comment explaining the
  original failure mode.
- **Validate config early.** New `TrainingConfig` fields belong in `validate()`
  so bad input fails before data loading.
- **Don't touch `experiments/legacy/` or `legacy/`.** These are preserved
  historical scripts.

## Benchmark numbers

Any accuracy figure added to the README must ship with the config file, commit
SHA and hardware used, plus the `metrics.json` it came from. Unsourced numbers
will be removed — the previous README's untraceable figures are why this rule
exists.
