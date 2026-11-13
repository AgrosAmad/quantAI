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


- [Z7 completed comparison](z7/REPORT.md): all12 fresh runs and independent
  numerical/quantum checks finished; the older dated Z7 snapshot is historical.
- [Z9 full attention combination](z9/REPORT.md): adds per-scale attention to
  three quantum branches, comparing mean fusion, classical output attention,
  and coherent quantum scale routing. Both plots include all17 methods with
  vanilla QVF in red. Missing results remain pending.
- [Z9 export status](z9/status_at_export.json).

The Z9 queue waits for Z8, then trains18 new runs (three methods, two datasets,
three seeds) at50,000 updates. Its local5x5-token attention keeps per-query
retrieval bounded. The prior dense-attention5q reference has different encoding
and circuits, so that comparison does not isolate head count.

Refresh all current compact records with
`python -m experiments.qvf_project_z.combination_export --root LOCAL_Z9_ROOT`.
Exporting does not commit or push. Generated source snapshots and heavy data
remain local. The final comparison requires the Z8 and Z9 checks to pass.


- [Z10 larger-code vanilla QVF](z10/REPORT.md): adds256 global code entries
  with the original five-qubit QVF circuit and RGB readout unchanged.
  Both64-value and256-value QVF controls are red in the18-method figures.
- [Z10 export status](z10/status_at_export.json).

Z8 has completed all84 entries and independent metric/quantum checks.
Z10 preserves the active Z9 study and waits for its full pipeline, then trains
six fresh runs: two datasets, three seeds,50,000 updates each. A larger input
adds24,576 classical first-layer weights. This is a code-size control, not
an equal-storage comparison with the spatial methods.

Refresh all compact records with
`python -m experiments.qvf_project_z.code_budget_export --root LOCAL_Z10_ROOT`.
Missing measurements stay pending; raw arrays, checkpoints and source snapshots
stay local. Exporting does not automatically commit or push.
