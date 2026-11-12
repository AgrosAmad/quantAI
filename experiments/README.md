# QVF research experiments

This directory contains the audited QVF reproduction and the subsequent image
representation studies. The current objective is a useful quantum implicit
neural representation with explicit circuit costs and better image structure.
The work does not claim quantum advantage over classical image models.

- [Original QVF reproduction](qvf_repro/README.md): implementation audit and
  direct image fitting, including the authors' original amplitude/RGB mapping.
- [Project A](qvf_project_a/README.md): encoder/VAE, larger-data fitting, spatial
  features, token conditioning and DINO/RAE investigations.
- [Project Z](qvf_project_z/README.md): local multiscale quantum fields, circuit
  variations, coherent routing and the current complete-method benchmark.
- [Session handoff](qvf_repro/HANDOFF_CODEX.md): decisions, current jobs and limits.
- [Compact benchmark records](../reports/qvf_benchmarks/README.md): protocols,
  verified summaries and comparison plots suitable for version control.

The working environment is the existing `quantai` Conda environment: Python3.10,
PyTorch2.1.2+cu121, PennyLane0.34 and NumPy1.26.3. Do not upgrade dependencies
inside a frozen study. Use physical GPU0 only; GPU1 is reserved for the user.
Commands below assume the datasets and required earlier study artifacts already
exist locally, as documented by each experiment. Those large artifacts are not
distributed in Git.

```bash
CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 /home/agroz/anaconda3/envs/quantai/bin/python -m experiments.qvf_project_z.test_coverage

# Use a new output directory. Creation snapshots the current sources.
CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 /home/agroz/anaconda3/envs/quantai/bin/python -m experiments.qvf_project_z.coverage_study --root experiments/qvf_project_z/results/my_z8 --create

# Waits for the existing Z7 study before starting two GPU0 workers.
CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 /home/agroz/anaconda3/envs/quantai/bin/python -m experiments.qvf_project_z.coverage_run --root experiments/qvf_project_z/results/my_z8 --workers 2

# Refresh compact reports after the run; this does not commit or push.
CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 /home/agroz/anaconda3/envs/quantai/bin/python -m experiments.qvf_project_z.coverage_export --root experiments/qvf_project_z/results/my_z8
```

The current Z8 comparison contains all13 Z6/Z7 variants and vanilla QVF at both
32×32 and256×256, with4,744 images,50,000 updates and three seeds per panel.
It reuses39 verified or currently running256px trials and schedules45 fresh
ones. The vanilla baseline uses the original5q field and64-value global code,
under the same benchmark training/evaluation recipe as the other methods.

The CIFAR and Stanford Dogs panels use different datasets, latent budgets and
coordinate-query budgets. Compare methods within a panel. Full completion
requires checkpoint/stream/metric verification, quantum audits and a final report;
pending rows and synthetic smoke results are not quality evidence.
