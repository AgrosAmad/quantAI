# QVF benchmark records

This directory contains compact research artifacts exported from the local
experiments. Source code and experiment instructions are in
[experiments](../../experiments/README.md). Datasets, checkpoints, raw arrays,
source snapshots, live queues and large galleries remain local.

- [Original QVF reproduction](vanilla_reproduction/results.json): verified
  single-image and 50-image fits. These are not the current larger-data benchmark.
- [Z5](z5/summary.json): completed 32×32 spatial/circuit study.
- [Z6](z6/comparison.json): completed 256×256 quantum-design study.
- [Z7 historical snapshot](z7/STATUS_20261112_0411UTC.json): provisional first-two-seed
  routing results. Do not treat this dated snapshot as the final Z7 result.
- [Z8 complete-method comparison](z8/REPORT.md): all13 Z6/Z7 methods plus vanilla
  QVF at both resolutions. Missing measurements are explicitly marked pending.
  The matched protocol uses50,000 updates and three seeds per method/dataset.
- [Status when exported](z8/status_at_export.json) and
  [artifact source hashes](export_manifest.json).

The current training recipe is common within each dataset. Vanilla QVF retains
its64-value global code; spatial methods have336 values at32px and21,504 at256px.
The two datasets differ, so cross-panel score differences are not a controlled
resolution comparison. No quantum advantage or hardware execution is claimed.

Run `python -m experiments.qvf_project_z.coverage_export --root LOCAL_Z8_ROOT`
after the full queue finishes to refresh these static records before committing
new results. Exporting does not commit or push automatically.
