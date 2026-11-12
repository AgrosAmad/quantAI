# Vanilla QVF image reconstruction

This directory implements the QVF image field with individually optimized image
codes, following the authors' released model. It contains no image encoder, VAE
sampling, diffusion model or added affine mapping from quantum measurements to
RGB. The current priority is to establish this baseline before extending it.

The November 8 audit and all nine fitting runs are complete. Vanilla QVF gives
sharp single-image fits and 50-image reconstruction accuracy within the paper's
reported range. The [interactive report](results/20261108_vanilla/report_final.html)
shows every fitted image alongside its original.

| Experiment | Measured reconstruction PSNR |
| --- | ---: |
| Three images fitted separately, seed 0 | 45.89 to 52.16 dB |
| Shared 50-image model, released-code conventions, three seeds | 29.76 ± 0.16 dB |
| Shared 50-image model, paper-text conventions, three seeds | 29.48 ± 0.42 dB |
| Paper Table 3, Gaussian + ReLU | 29.8 ± 1.1 dB |

Our collection error bars are sample standard deviations over seeds 0, 1 and 2.
The single-image range is over three different photographs, not repeated seeds.
All nine final checkpoints passed reload, image-ID, source-hash and raw-metric
checks. The largest reload difference was 1.55e-6 in RGB; trained outputs matched
the released PennyLane implementation within 8.35e-7 on the audited coordinates.
See [the verified results](results/20261108_vanilla/verified_results.json) and
[verification record](results/20261108_vanilla/verified_results_verification.json).

These successful fits show that severe blur is not an unavoidable property of
this QVF implementation. They do not isolate which encoder, readout or training
changes in Project A caused its loss of detail. That extension remains a separate
experiment; the baseline here uses the original image-fitting task.

The starting sources are the [QVF paper, version 2](https://arxiv.org/abs/2508.10900v2)
and the [released image implementation](https://github.com/shutengW/Quantum-Visual-Fields-with-Neural-Amplitude-Encoding/blob/cb647ef16aa20a8ca90514cef74a7b0927bb60a2/src/Images/QVF_for_image.py),
pinned at commit `cb647ef16aa20a8ca90514cef74a7b0927bb60a2`. The remote HEAD was
checked against this commit for the November 8 baseline. The model and simulator
in `qvf_fast.py` have not been changed for these new experiments.

## What is preserved and what is uncertain

The image field concatenates a 64-number image code with encoded pixel
coordinates. A three-hidden-layer ReLU MLP of width 128 produces 32 energies,
which become normalized amplitudes through softmax and a square root. The
five-qubit circuit has five layers of controlled-RY and RY gates. Its measurements
provide RGB predictions. Codes and model weights are optimized together.

The released code and paper specify different conventions. The runs keep these
two readings separate; neither uses Project A's additional output rescaling.

| Choice | Released-code conventions (`paper`) | Paper-text conventions (`paper_text`) |
| --- | --- | --- |
| RGB readout at five qubits | Z0 + Z3, Z1 + Z4, Z2 | First three raw Z expectations |
| Coordinate frequencies | Powers of two | Powers of two multiplied by pi |
| Mean image-code norm penalty weight | 0.0001 | 0.001 |
| Extra output shift or rescaling | None | None |

The `paper` preset uses the paper's model width, qubit count, depth, learning rate
and training duration, filling unspecified details from the release. It is not
the released demo configuration: that demo has four qubits, depth two and width
256. The `paper_text` preset additionally follows the paper on the three
conventions in the table.

The paper does not fully specify the exact image subset, image-code length,
positional-encoding depth, batch size or code learning rate. We use the first
50 CIFAR-10 dogs in official training-set order, 64-number codes, six coordinate
frequencies and the same initial learning rate for model and codes. The collection
runs use batches of 10. These choices are recorded rather than presented as
known details of the authors' experiments.

The Gaussian model has 48,950 model parameters. The 50 separately optimized
64-number codes add 3200 trainable values, for 52,150 in total. This is consistent
with the paper's rounded count of 52,000 if image codes are included. That is a
possible explanation of the apparent count mismatch in the older handoff, not
confirmation of the authors' counting convention.

## Implementation audit

`audit_vanilla.py` imports the official source from outside this repository. It
repairs four source expressions in memory, leaving the external checkout intact:

- Three rotation indices mistakenly address a two-dimensional parameter tensor
  with three indices. Each is corrected to index the qubit and circuit layer.
- The dataset reshapes the full class tensor as though it were the requested
  subset. It is corrected to reshape the already selected subset.

The original amplitude network, quantum gates, gate ordering, RGB mapping,
coordinate encoding and loss otherwise execute unchanged. Verification covers
20 circuit configurations, initial and generic angles, orthogonality, outputs and
angle gradients. Full five-qubit models also compare outputs for both RGB
readouts, all first-step parameter and code gradients, and 30 Adam updates for
both Gaussian and identity initialization.

The fresh audit passed. Worst circuit output difference was 1.28e-15 and worst
angle-gradient difference was 6.40e-14 in float64. Full-model training losses
agreed within 1.09e-6 relative error in float32. See
[the audit record](results/20261108_vanilla/audit.json).

This validates a faster exact simulator of the same circuit, not execution on
quantum hardware. `verify_vanilla.py` additionally reloads final trained
checkpoints and checks their predictions against the released PennyLane model.

## Direct image-fitting protocol

The [recorded protocol](results/20261108_vanilla/protocol.json) defines nine runs:

- Three single-image fits, using the first three dogs separately, seed 0,
  released-code conventions and 5000 optimizer updates per image.
- Three 50-image collection fits with released-code conventions, seeds 0, 1 and 2.
- Three fits of the same collection with paper-text conventions and the same seeds.

Each collection run lasts 5000 epochs with five batches per epoch, giving 25,000
optimizer updates. All 1024 pixels of each selected image contribute to each
update. Model and codes use Adam with initial learning rate 0.001. The paper's
plateau scheduler uses patience 50 epochs and multiplies the learning rate by 0.9.
The code-norm penalty remains present; it is part of the original auto-decoder
objective, not a VAE KL penalty. No perceptual or adversarial loss is added.

Evaluation uses the final checkpoint and all pixels of the fitted images.
Reported PSNR is computed per image from raw predictions, then averaged. Values
outside the RGB range are not clipped before the primary metric. PNGs clip only
for display; clipped metrics are recorded separately. Single-image tests check
capacity and implementation, not exact reproduction of the paper's underspecified
single-image experiment. The 50-image results are compared with Table 3's
Gaussian + ReLU result, 29.8 ± 1.1 dB.

## Commands and artifacts

Run from the repository root using the pinned environment. Do not upgrade its
packages. CIFAR-10 must already exist in `experiments/qvf_repro/data`; the loader
does not download it. GPU 0 is the only permitted GPU.

```bash
QVF_PY=/home/agroz/anaconda3/envs/quantai/bin/python
QVF_RUN=experiments/qvf_repro/results/my_vanilla_run
QVF_OFFICIAL=/tmp/qvf_official/repo/src/Images/QVF_for_image.py

$QVF_PY experiments/qvf_repro/audit_vanilla.py \
  --official "$QVF_OFFICIAL" --output "$QVF_RUN/audit.json"

$QVF_PY experiments/qvf_repro/sweep.py vanilla_single \
  --jobs 1 --device cpu --save-checkpoint --output-dir "$QVF_RUN"

CUDA_VISIBLE_DEVICES=0 $QVF_PY experiments/qvf_repro/sweep.py vanilla_collection \
  --jobs 1 --device cuda:0 --save-checkpoint --output-dir "$QVF_RUN"

CUDA_VISIBLE_DEVICES=0 $QVF_PY experiments/qvf_repro/sweep.py vanilla_paper_text \
  --jobs 1 --device cuda:0 --save-checkpoint --output-dir "$QVF_RUN"
```

The completed November 8 runs can be audited and rendered with a fresh report name:

```bash
QVF_RUN=experiments/qvf_repro/results/20261108_vanilla
$QVF_PY experiments/qvf_repro/verify_vanilla.py \
  --results "$QVF_RUN" --official "$QVF_OFFICIAL" \
  --output-stem "$QVF_RUN/my_verified_results"
$QVF_PY experiments/qvf_repro/vanilla_report.py \
  --data "$QVF_RUN/my_verified_results.json" \
  --output-stem "$QVF_RUN/my_report"
```

Each training run records exact image IDs, configuration, source hashes, source
snapshots, training history, raw and clipped errors, and adjacent-pixel gradient
error. With `--save-checkpoint`, it also saves final model weights, optimized
image codes, optimizer states, originals and raw reconstructions. The `_all.png`
montage includes all images, not selected examples. Heavy checkpoints stay local
and are ignored by git.

Existing training artifacts are protected against overwriting. The sweep skips
completed runs. A `.started.json` without the final `.json` identifies an
unfinished run; inspect its log and use a fresh output name rather than overwrite
it. Training additions capture provenance and checkpoints; the numerical model
and training objective remain unchanged.
