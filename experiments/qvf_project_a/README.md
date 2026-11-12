# Project A with a QVF VAE and latent diffusion

The user named the current pre-A decoder/representation work
[**Project Z**](../qvf_project_z/README.md) on 2026-11-10. New studies live there;
historical experiments below remain at their original paths.

Current focus, clarified by the user on 2026-11-10: a useful quantum INR with
explicit quantum execution costs, better spatial representation than vanilla
QVF, and evidence of scaling beyond small image collections/resolutions.
Classical controls support attribution; outperforming classical models is not
the success criterion. The target remains a continuous coordinate field with a
clearly specified quantum role and direct quantum RGB readout as a reference.
The next-study proposal is recorded in the
[post-V11 plan](CONVOLUTIONAL_QVF.md#post-v11-quantum-feasibility-spatial-representation-and-scale);
its initial feasibility and architecture comparisons, the Z4 spatial-encoding
and five/seven-qubit pilot, and the Z5 comparison on 4,744 images with 50,000
updates are now complete in Project Z. Z5 has 27 fresh runs and independent
checkpoint/quantum audits. Longer training improves fitting; spatial loading
narrows its completion gap but still trails attention. Five qubits remain the
main cost-conscious setting. See Z for the current results and their limits.
Encoder and diffusion integration follow decoder work.

## Interface architecture study (V11)

The user authorized directions 1 and 2 separately/in parallel, then their
combination, then direction 3. All 27 V11 runs completed on 2026-11-10 under the
[saved protocol](results/20261109_v11_interfaces/protocol.json): nine conditions,
three seeds, 1,024 images and 10,000 updates each (27 runs). Every arm uses the
same 336-value image codes, RGB targets, paired data/query streams and pixel +
0.5 edge objective. Quantum variants retain five qubits. Sources and parameters
of the original QVF and V9/V10 remain unchanged.

| Stage | Condition | Shared parameters |
| --- | --- | ---: |
| Separate | Attention/amplitude reference | 43,986 |
| Separate | Wider attention capacity control | 55,164 |
| Separate | 1: Local patch input, amplitude loading | 54,642 |
| Separate | 2: Attention input, angle loading | 43,133 |
| Combined | 1+2: Local patches and angle loading | 53,789 |
| Combined | Same input rotations, shared circuit processing removed | 53,639 |
| Combined | Classical local-patch field | 54,061 |
| Grid output | 3: Quantum feature grid + convolutional RGB | 90,944 |
| Grid output | Matched classical feature grid + same conv architecture | 91,464 |

[interface_qvf.py](interface_qvf.py) implements replicate-border 3x3 patch
unfolding at each scale, bilinearly queried before the field. This provides 108
local values plus 26 coordinate features without increasing fitted image storage.
Angle loading replaces the initial amplitude map with five repeated data-RY
uploads, each followed by one original processing repetition: 25 input rotations,
125 effective shared processing angles, plus explicit learned coordinate slopes.
Initial slopes span declared scales; frequency bands are not enforced. The
original grouped-Z RGB rule is retained for the coordinate-field comparisons.

Direction 3 queries an 8x8 feature grid for 15 Z/ZZ statistics, then uses a small
shared convolutional upsampler for native 32x32 RGB. Its classical control has
slightly more shared parameters and identical initial convolutional weights.
This is a native-raster decoder, not the same continuous quantum RGB field.
Grid models use 64 feature queries per image/update; coordinate models use 128
sampled endpoint queries. Both receive the same 128 RGB targets. Full rendering
uses 64 versus 1,024 quantum queries/image; these counts are not runtime claims.

Five targeted tests passed, including independent PennyLane outputs/gradients,
independent patch interpolation/gradients, chunking, paired controls, training
and reload, and hidden-target isolation for completion. All nine GPU preflights
passed exact training/completion reloads. The study runs on GPU 0 only.
The runner enforces stage ordering; completed runs are skipped, interrupted
optimizer states are preserved rather than silently resumed or overwritten.
Completion matches V10: freeze the decoder, optimize fresh codes from 256
observed pixels of 32 excluded images for 1,000 updates, then score withheld
pixels and pairs with both endpoints withheld. Final checkpoints are primary.

Entry points: [interface_study.py](interface_study.py),
[verify_interface_study.py](verify_interface_study.py),
[interface_report.py](interface_report.py). Preserve protocol/source snapshots.
Do not report unfinished variants as results. No encoder/diffusion or hardware
experiment is included.

All 27 runs passed
[independent verification](results/20261109_v11_interfaces/verified/verification.json).
Every final training/completion image reloads exactly. Source/checkpoint hashes,
paired codes/common initial modules, independent image/query streams, quantum
query counts and NumPy float64 metrics match. Step-2,000 metrics were recomputed,
and all three attention references reproduced V10's final weights and codes
bit-for-bit. Report means, sample SDs and paired differences were checked against
the verified runs. No training remains running.

| Model | Training PSNR ↑ | Training edge MSE ↓ | Completion PSNR ↑ | Completion edge MSE ↓ |
| --- | ---: | ---: | ---: | ---: |
| Attention reference | 24.08 | 0.003730 | 21.19 | 0.005500 |
| Wider attention control | 24.19 | 0.003692 | 21.38 | 0.005321 |
| 1: Local patches | 23.68 | 0.004037 | 21.04 | 0.005763 |
| 2: Angle encoding | 23.44 | 0.004103 | 20.90 | 0.005906 |
| 1+2: Patches + angles | 23.12 | 0.004371 | 20.84 | 0.006015 |
| 1+2: Shared circuit processing removed | 21.90 | 0.005679 | 20.17 | 0.007436 |
| Classical patch field | 23.71 | 0.004065 | 21.22 | 0.005570 |
| 3: Quantum grid + convolutions | 22.47 | 0.004543 | 21.18 | 0.005343 |
| 3: Classical grid + convolutions | 22.93 | 0.004336 | 20.61 | 0.005764 |

Means across three seeds; PSNR is mean per-image PSNR in dB. Completion means
fitting new codes to 256 observed pixels in each of 32 excluded images, then
scoring withheld pixels/edges. It is not one-pass encoder generalization.
Full [per-seed statistics and paired differences](results/20261109_v11_interfaces/verified/summary.json),
[plain-text results](results/20261109_v11_interfaces/verified/summary.txt),
[edge comparison](results/20261109_v11_interfaces/verified/comparison.png), and
[pixel reconstruction comparison](results/20261109_v11_interfaces/verified/pixel_reconstruction.png)
are saved.

The tested versions of directions 1 and 2 do not improve the attention reference
at this budget: mean training edge error rises 8.24% and 10.02%, respectively.
Combining them raises it 17.19%, with worse completion scores too. This particular
patch implementation also changes retrieval from adaptive attention to bilinear
patch sampling, so the result does not prove that preserving neighbours is
unhelpful in general. Nor does one angle-loader recipe establish its converged
capacity or optimal initialization. The wider attention control is the strongest
V11 model by all four mean quality metrics: training edge error falls 1.00%,
completion edge error 3.25%, relative to the reference. Its pixel-PSNR gains vary
by seed, while its completion-edge improvement occurs in all three seeds.

Learned circuit processing matters within the combined angle model: compared
with removing that processing and retraining, it lowers training edge error
23.04% and improves PSNR 1.23 dB; completion also improves. The matched classical
patch field nevertheless performs better than the combined quantum field on
both fitting and completion. This does not isolate entanglement or demonstrate
a quantum advantage.

Direction 3 offers a different tradeoff. The quantum grid fits training images
less accurately, but reaches approximately the original attention reference's
mean completion PSNR (21.18 versus 21.19 dB). Its mean completion edge error is
2.85% lower, with mixed per-seed differences; there is no formal equivalence or
robust overall accuracy-win claim. Wider attention still has better mean quality
metrics. Against the matched classical grid, the quantum grid improves completion
PSNR by 0.56 dB and lowers completion edge error by 7.30%, with both improvements
in all three seeds. It has lower mean training PSNR and higher training edge
error than that classical grid. This is a useful completion result within one
paired architecture/optimization study, not evidence of general quantum superiority.

The query reduction is concrete: 64 versus 1,024 circuit evaluations to render a
full image, at the cost of a classical convolutional decoder and more shared
parameters (90,944 versus 43,986 for the original attention reference). During
training the counts are 64 versus 128 queries per image/update, so the reduction
there is twofold, not sixteenfold. Preparation/gate counts also differ; finite-shot
accuracy and actual hardware runtime remain untested. The grid's small completion
edge gain does not mean the visible fine-detail problem is solved; examples
remain soft.

Fixed galleries: [separate training](results/20261109_v11_interfaces/verified/separate_training_examples.png),
[separate completion](results/20261109_v11_interfaces/verified/separate_completion_examples.png),
[combined training](results/20261109_v11_interfaces/verified/combined_training_examples.png),
[combined completion](results/20261109_v11_interfaces/verified/combined_completion_examples.png),
[grid training](results/20261109_v11_interfaces/verified/grid_training_examples.png),
[grid completion](results/20261109_v11_interfaces/verified/grid_completion_examples.png).
Every gallery starts with the original; completion inserts observed-only input.
Remaining columns are recorded in the verification JSON. Rows are the first 12
protocol IDs from seed 0. Learning curves are saved separately for each stage.

Keep the attention family as the quality reference. Retain the quantum feature
grid as a candidate for reduced quantum query counts and completion, while
treating its convolutional output as a different decoder design. No V11 variant
establishes a substantial overall quality improvement over the existing attention
approach under this fixed recipe. All models are bounded pilots: no convergence,
full-data encoder or quantum-advantage conclusion follows.

Reproduction commands (run each stage for seeds 0, 1 and 2 before the next stage):

```bash
CUDA_VISIBLE_DEVICES=0 /home/agroz/anaconda3/envs/quantai/bin/python -m experiments.qvf_project_a.interface_study run --root experiments/qvf_project_a/results/20261109_v11_interfaces --stage separate --seed 0
CUDA_VISIBLE_DEVICES=0 /home/agroz/anaconda3/envs/quantai/bin/python -m experiments.qvf_project_a.interface_study run --root experiments/qvf_project_a/results/20261109_v11_interfaces --stage combined --seed 0
CUDA_VISIBLE_DEVICES=0 /home/agroz/anaconda3/envs/quantai/bin/python -m experiments.qvf_project_a.interface_study run --root experiments/qvf_project_a/results/20261109_v11_interfaces --stage grid --seed 0
CUDA_VISIBLE_DEVICES=0 /home/agroz/anaconda3/envs/quantai/bin/python -m experiments.qvf_project_a.verify_interface_study --root experiments/qvf_project_a/results/20261109_v11_interfaces
/home/agroz/anaconda3/envs/quantai/bin/python -m experiments.qvf_project_a.interface_report --root experiments/qvf_project_a/results/20261109_v11_interfaces
```

## Architecture follow-up after V10

The user asks about substantial classical/quantum architecture changes, including
the quantum input and output. The
[design and proposed controls](CONVOLUTIONAL_QVF.md#post-v10-redesigning-local-conditioning-and-the-quantum-interface)
prioritize preserving local neighbourhoods before encoding and directly encoding
coordinates through repeated quantum rotations. A quantum feature grid with a
small convolutional RGB decoder is recorded as a more radical alternative with
different QINR semantics. The selected comparison is complete in V11 above;
other proposed ablations remain untested.

A bounded [frozen-readout diagnostic](readout_probe.py) is complete on the three
V10 attention/edge models. Learning an affine RGB map of the existing five Z
means improves training PSNR by 0.118 dB; adding ten ZZ correlations increases
that to 0.125 dB. Neither improves edge error. The original model and image codes
remain frozen; this fits only a shared 18- or 48-parameter readout to the same
training images. It is not a generalization or end-to-end architecture result.
The finding lowers the priority of changing only the readout.

Two independent measurement/objective tests pass, and original RGB recovery is
within 5.97e-8. [Results and fitted maps](results/20261109_readout_probe/results.json),
[verification](results/20261109_readout_probe/verification.json), and the
[static comparison](results/20261109_readout_probe/comparison.png) are saved.
The probe remains separate from the completed V11 study. Reproduce into a fresh output directory:

```bash
CUDA_VISIBLE_DEVICES=0 /home/agroz/anaconda3/envs/quantai/bin/python -m experiments.qvf_project_a.readout_probe --root experiments/qvf_project_a/results/20261109_v10_edges --output /path/to/new/readout_probe
```

## Edge-fidelity study (V10)

The user authorized training the post-V9 comparison. All 12 V10 runs are complete under the
[saved protocol](results/20261109_v10_edges/protocol.json): attention versus
revised staged processing, each with pixel-only versus pixel-plus-edge loss.
All four conditions use five qubits, 336 code values per image, the same 1,024
training IDs, 10,000 updates, and three paired seeds (12 runs). The edge weight
is fixed at 0.5 before training; this is one coefficient, not an optimization
over loss weights.

[edge_qvf.py](edge_qvf.py) retains all three feature scales in initial amplitude
preparation and optionally re-encodes intermediate/fine features later. Extra
rotation maps start at zero, preserving the attention-only function up to
floating-point roundoff. Common parameters and initial codes are paired by seed.
Attention has 43,986 shared parameters and the revised staged model 44,036; the
latter adds ten input-dependent RY gates while retaining the same 125 effective
trainable QVF circuit angles.
The original QVF and V9 implementation files remain unchanged.

[edge_study.py](edge_study.py) samples 32 horizontal and 32 vertical neighbour
pairs per image, evaluating all 128 endpoints in every condition. Inverse-degree
weights correct the RGB objective for boundary-pixel under-sampling, so its
expectation is uniform full-image MSE. The edge term is signed neighbour-change
MSE, averaged equally across orientations, without coordinate-spacing rescaling.
The sampler differs from V9; within-V10 step-2,000 checkpoints provide duration
controls rather than treating V9's scores as matched baselines.

For excluded-image completion, every arm uses the same pixel-only code fitting:
freeze the decoder, optimize new codes for 1,000 updates using 256 observed pixels
in each of 32 excluded images, and score 768 withheld pixels. Edge evaluation
includes only pairs with both endpoints withheld. No held-out model selection,
encoder, diffusion or official test-set use is involved.

Seven targeted tests passed, including exact-metric sampling, baseline function
and shared-gradient equivalence, checkpoint recovery and hidden-target isolation.
All four GPU training/reload preflights passed with zero reconstruction error;
initial GPU equivalence was within 4.1e-7 RGB. All 12 runs passed
[independent verification](results/20261109_v10_edges/verified/verification.json):
every final training/completion reconstruction reloads exactly; NumPy float64
metrics, source/checkpoint hashes, initialization and independently regenerated
query streams match. Step-2,000 metrics were also recomputed from checkpoints.
Verification and static reports use [verify_edge_study.py](verify_edge_study.py)
and [edge_report.py](edge_report.py). No training remains running.

| Model and training loss | Training edge MSE ↓ | Training PSNR ↑ | Completion edge MSE ↓ | Completion PSNR ↑ |
| --- | ---: | ---: | ---: | ---: |
| Attention, pixel only | 0.004176 | 23.64 | 0.005531 | 21.11 |
| Attention, pixel + edge | 0.003730 | 24.08 | 0.005500 | 21.19 |
| Revised stages, pixel only | 0.004124 | 23.93 | 0.005369 | 21.41 |
| Revised stages, pixel + edge | 0.003730 | 24.32 | 0.005402 | 21.44 |

Values are means across three seeds; PSNR is mean per-image PSNR in dB.
Completion scores only withheld pixels/edges after fitting fresh codes to the
observed pixels. Per-seed values, sample standard deviations and paired differences
are in the [summary](results/20261109_v10_edges/verified/summary.json), with
[plain-text results](results/20261109_v10_edges/verified/summary.txt), a
[comparison plot](results/20261109_v10_edges/verified/comparison.png) and
[learning curves](results/20261109_v10_edges/verified/learning_curves.png).

Adding edge loss reduces training edge error by **10.68% for attention** and
**9.55% for revised stages**, with improvements in every seed. Training PSNR also
increases by 0.44 and 0.39 dB, respectively. Thus this loss improves the measured
neighbour differences without sacrificing average pixel reconstruction under
this recipe. It is a supervised objective improvement, not evidence of a quantum
architectural advantage or a perceptual sharpness guarantee.

Against attention trained with the same edge loss, revised stages have essentially
the same mean training edge error (0.01% worse), with mixed per-seed differences.
They improve mean training PSNR by 0.24 dB and completion PSNR by 0.25 dB; the
training PSNR difference changes sign across seeds. Completion edge error is
1.78% lower on average but worsens in one seed. Adding edge loss to revised stages
actually worsens mean completion edge error by 0.61%. These results do not
establish an additional edge-fidelity benefit from the added circuit stages.

Training duration matters: attention with pixel loss falls from 0.006395 edge
MSE at 2,000 updates to 0.004176 at 10,000, a **34.70% reduction** within V10.
This is larger than the added edge-loss effect at 10,000 updates. It is not a
convergence claim; inspect the saved learning curves before extending the budget.

Fixed galleries: [training](results/20261109_v10_edges/verified/training_examples.png)
and [completion](results/20261109_v10_edges/verified/completion_examples.png).
Training columns are original, attention/pixel, attention/edge, stages/pixel,
stages/edge; completion inserts observed-only input after the original. Rows are
the first 12 protocol IDs from seed 0. Fine detail remains visibly soft.

The practical result is to retain edge loss as a useful training option and keep
attention with that loss as the comparison baseline. Extra stages are not yet
justified by an edge-error win. This is still a 1,024-image simulation study with
three seeds, one loss weight and 32 sparse-completion images. It does not test a
full-data encoder, diffusion, hardware/finite-shot execution, or quantum advantage.
V10 does not rerun V9's coarse-only initial staging at this budget, so it cannot
isolate that earlier input restriction as the cause of V9's edge tradeoff.

```bash
CUDA_VISIBLE_DEVICES=0 /home/agroz/anaconda3/envs/quantai/bin/python -m experiments.qvf_project_a.edge_study run --root experiments/qvf_project_a/results/20261109_v10_edges --seed 0
CUDA_VISIBLE_DEVICES=0 /home/agroz/anaconda3/envs/quantai/bin/python -m experiments.qvf_project_a.edge_study run --root experiments/qvf_project_a/results/20261109_v10_edges --seed 1
CUDA_VISIBLE_DEVICES=0 /home/agroz/anaconda3/envs/quantai/bin/python -m experiments.qvf_project_a.edge_study run --root experiments/qvf_project_a/results/20261109_v10_edges --seed 2
CUDA_VISIBLE_DEVICES=0 /home/agroz/anaconda3/envs/quantai/bin/python -m experiments.qvf_project_a.verify_edge_study --root experiments/qvf_project_a/results/20261109_v10_edges
/home/agroz/anaconda3/envs/quantai/bin/python -m experiments.qvf_project_a.edge_report --root experiments/qvf_project_a/results/20261109_v10_edges
```

Completed runs are skipped; interrupted optimizer states are preserved but are
not automatically resumed. Do not modify the saved protocol or training sources.

## Token-QVF pilot (V9)

The user authorized implementing token conditioning and testing a modest qubit
increase with controls. The V9 pilot is complete under its
[saved protocol](results/20261109_v9_tokens_pilot/protocol.json): 1,024 training
images, 2,000 updates, two paired seeds, and five model variants at both five
and seven qubits. The variants are original global-code QVF (64 values/image),
global-code QVF with 336 values, multiscale bilinear lookup, multiscale token
attention, and token attention with staged quantum encoding. All three spatial
variants have 336 fitted values per image: 2x2, 4x4 and 8x8 grids, four channels
per grid cell. No image encoder or diffusion is trained.

The implementation is in [token_qvf.py](token_qvf.py), with an experiment runner
in [token_study.py](token_study.py). The original QVF source is unchanged.
The staged circuit has the same five processing repetitions, split 2+2+1, with
input-dependent RY rotations between blocks. It retains the quantum state until
the final readout. At seven qubits the original readout function uses its
seven-wire grouping and normalization; amplitude output width and gate count
also grow. Comparisons within each qubit count isolate the architecture from
that qubit-scaling package.

The final checkpoint also undergoes a separate completion diagnostic: freeze
the decoder, fit fresh codes for 32 excluded images using 256 observed pixels,
and score the remaining 768 pixels. This tests optimization-based transfer,
not encoder inference. The official test set remains unused. Checkpoint choice
is fixed at the final update, with no held-out selection.

Nine targeted tests passed, including independent PennyLane outputs/gradients,
coordinate chunking, training/reload, and a check that changing unobserved RGB
targets does not change fitted completion codes. All ten model configurations
passed GPU reload preflights with zero RGB discrepancy. Two initial CUDA driver
initialization failures, before optimization, are preserved in
`results/20261109_v9_tokens_pilot/launch_failures/`; both restarted seed processes
completed. No training remains running.

All 20 runs passed [independent verification](results/20261109_v9_tokens_pilot/verified/verification.json):
every training/completion image was rerendered from its checkpoint with zero RGB
error in the corresponding trainable/frozen execution mode. NumPy float64 scores,
source/checkpoint hashes, initialization, image visits and pixel RNGs match.

| Model | Training PSNR, 5q | Training PSNR, 7q | Completion PSNR, 5q | Completion PSNR, 7q |
| --- | ---: | ---: | ---: | ---: |
| Vanilla QVF, 64 code entries | 17.28 | 16.96 | 17.92 | 17.56 |
| Global QVF, 336 code entries | 18.30 | 18.05 | 19.30 | 19.07 |
| Spatial grid, bilinear lookup | 20.04 | 19.95 | 19.46 | 19.24 |
| Token attention | 20.21 | 20.09 | 19.76 | 19.55 |
| Token attention + staged circuit | 20.76 | 20.85 | 19.96 | 20.36 |

Values are mean per-image PSNR in dB, then averaged across two seeds. The three
spatial models all use 336 code entries/image. Completion scores only unobserved
pixels in the excluded images. Full per-seed scores and sample standard deviations
are in the [summary](results/20261109_v9_tokens_pilot/verified/summary.json), with
[static comparison plots](results/20261109_v9_tokens_pilot/verified/comparison.png),
[learning curves](results/20261109_v9_tokens_pilot/verified/learning_curves.png), and
[plain-text results](results/20261109_v9_tokens_pilot/verified/summary.txt).

At five qubits, staged tokens improve fitting by 3.48 dB over vanilla64 and
2.46 dB over global336. Thus the fixed-budget gain survives controls for qubit
count and per-image code storage. Most of the fitting improvement appears when
introducing the local grid; attention adds 0.18 dB and staged encoding adds a
further 0.55 dB. Spatial models also use fewer shared weights than global336
(about 43k versus 84k at five qubits), though compute and gate budgets differ.

Adding qubits alone is not an automatic improvement under this recipe. The
staged model gains only 0.09 dB training PSNR at seven versus five qubits; its
completion mean gains 0.40 dB, with substantial seed variation (seven-qubit
completion 20.36 +/- 0.42 dB sample SD, versus 19.96 +/- 0.06 at five qubits).
Other seven-qubit variants have lower means than their five-qubit counterparts.
This is a short-budget result, not a conclusion about converged capacity.

Fixed galleries: [five-qubit training](results/20261109_v9_tokens_pilot/verified/q5_training_examples.png),
[seven-qubit training](results/20261109_v9_tokens_pilot/verified/q7_training_examples.png),
[five-qubit completion](results/20261109_v9_tokens_pilot/verified/q5_completion_examples.png),
[seven-qubit completion](results/20261109_v9_tokens_pilot/verified/q7_completion_examples.png).
Training columns: original, vanilla64, global336, bilinear, attention, staged.
Completion adds observed-only input after the original. Rows are the first 12
protocol IDs from seed 0. The images still look soft. Staging improves pixel
PSNR but does not improve the gradient-error diagnostic over attention alone;
do not claim fine-detail recovery from PSNR alone.

The user's next question concerns improving edge fidelity beyond attention.
The [follow-up proposal](CONVOLUTIONAL_QVF.md#post-v9-follow-up-improving-edge-fidelity-beyond-attention)
separates retaining all scale summaries in the staged model's initial amplitude
map from adding a supervised neighbour-difference loss. Compare both changes
against attention trained with the same loss and query budget. V10 implements
and tests this comparison. The [edge diagnostic](results/20261109_v9_tokens_pilot/verified/edge_diagnostic.png)
plots existing V9 history only.

This remains a 1,024-image, 2,000-update pilot with two seeds and 32 completion
images. It does not establish convergence, a quantum advantage, an isolated
benefit from entanglement/scale ordering, or large-data encoder generalization.
Longer full-data runs, repeated-encoding/depth controls and trained classical or
circuit-removal controls remain necessary for those claims. Seven qubits were
manageable in simulation (maximum allocated GPU memory about 166 MiB across
this pilot), but simulation timings are not hardware-execution estimates.

Commands for the saved pilot:

```bash
CUDA_VISIBLE_DEVICES=0 /home/agroz/anaconda3/envs/quantai/bin/python -m experiments.qvf_project_a.token_study run --root experiments/qvf_project_a/results/20261109_v9_tokens_pilot --seed 0
CUDA_VISIBLE_DEVICES=0 /home/agroz/anaconda3/envs/quantai/bin/python -m experiments.qvf_project_a.token_study run --root experiments/qvf_project_a/results/20261109_v9_tokens_pilot --seed 1
CUDA_VISIBLE_DEVICES=0 /home/agroz/anaconda3/envs/quantai/bin/python -m experiments.qvf_project_a.verify_token_study --root experiments/qvf_project_a/results/20261109_v9_tokens_pilot
/home/agroz/anaconda3/envs/quantai/bin/python -m experiments.qvf_project_a.token_report --root experiments/qvf_project_a/results/20261109_v9_tokens_pilot
```

Completed runs are skipped. Partial runs are preserved rather than silently
overwritten; the runner does not resume an interrupted optimizer. For another
budget, create a fresh protocol/output directory using `token_study protocol`.

## Current direction: improve QVF with local features at several scales

Following V8, the user prefers improving the QINR architecture before returning
to an encoder and latent diffusion. The proposed next study is documented in
[the spatial-QVF design](CONVOLUTIONAL_QVF.md#next-study-directly-learned-local-codes-at-several-scales):
fit compact spatial codes at several scales and condition each QVF query on
local features from all scales. Retain the original quantum circuit as a control,
and also investigate injecting coarse, intermediate and fine features into
successive blocks of the same five-qubit register. The proposed circuit extension
keeps the RGB readout fixed and measures only after all stages; see the
[circuit design and controls](CONVOLUTIONAL_QVF.md#circuit-extension-inject-different-scales-at-different-stages).
The user also proposes [token-based conditioning](CONVOLUTIONAL_QVF.md#token-extension-coordinate-queries-select-local-information):
treat spatial codes as positioned tokens and compare bilinear lookup with small
classical coordinate-to-token attention. Test retrieval separately from the
circuit change; simply flattening a grid into tokens changes no computation.
Control image-code storage, classical capacity and quantum execution cost.
V9 implements these selected variants as a pilot; other designs in the proposal
remain untested. V6's generated single-query-scale grid and V8's pretrained RAE
features remain separate completed experiments.

## Frozen RAE features with trained QVF and classical readouts (V8)

The user-authorized 50,000-update study is complete under the
[saved protocol](results/20261109_v8_readout50k/protocol.json). All six runs
finished: three matched seeds for each of two output architectures. DINO and the
published RAE decoder body stayed frozen; only the new spatial adapter and
readout were trained. No training remains running.

| Reconstruction model/checkpoint | Training PSNR | Validation PSNR |
| --- | ---: | ---: |
| Published frozen RAE reference | 25.56 dB | 25.81 dB |
| QVF head, final 50,000 updates | 28.12 ± 0.39 dB | 24.97 ± 0.13 dB |
| Classical head, final 50,000 updates | 28.03 ± 0.35 dB | 24.99 ± 0.18 dB |
| QVF head, selected by validation MSE | 25.90 ± 0.18 dB | 26.03 ± 0.11 dB |
| Classical head, selected by validation MSE | 25.91 ± 0.05 dB | 26.13 ± 0.03 dB |

Trained-model values are mean ± sample standard deviation over three seeds;
the published reference is one unchanged pretrained checkpoint. All values use
the same images and native 32×32, unrounded/unclipped RGB scoring. The QVF
selection steps are 11,000 / 9,000 / 3,000 for seeds 0 / 1 / 2; classical
selection steps are 7,000 / 7,000 / 3,000. Selection minimizes validation pixel
MSE, not PSNR. No test-set or per-image code fitting is involved.

The QVF and classical readout packages reach similar accuracy on these frozen
features. The final paired QVF-minus-classical validation gap is −0.027 dB
(sample SD 0.210); at selected checkpoints it is −0.103 dB (SD 0.096).
This is not a formal equivalence or quantum-advantage result. Both readouts
overfit under this recipe: training PSNR keeps improving while validation
quality declines. The selected QVF models preserve much of the RAE's visual
structure and achieve comparable native pixel fidelity. Their small pixel-score
gain over the published zero-shot reference reflects local adaptation with a
different objective, not an isolated quantum benefit or universal perceptual gain.

See the [static learning curves](results/20261109_v8_readout50k/verified/learning_curves.png),
[50k examples](results/20261109_v8_readout50k/verified/validation_final_seed0_examples.png),
and [earlier selected examples](results/20261109_v8_readout50k/verified/validation_selected_seed0_examples.png).
Examples use the first 16 validation IDs in fixed split order. Complete static
galleries for all 256 validation images, each seed and both checkpoint choices
are saved alongside them. [Plain-text summary](results/20261109_v8_readout50k/verified/summary.txt),
[full measurements](results/20261109_v8_readout50k/verified/results.json),
and [verification](results/20261109_v8_readout50k/verified/verification.json).

All 36 tests pass. Independent NumPy calculations verify all metrics for the
5,000 images at every final and selected checkpoint. Every saved reconstruction
was rerendered with zero RGB error; fresh image-to-DINO/body-to-readout execution
also matches exactly on 32 validation images per checkpoint. Initial common
weights, every sampled training batch, optimizer budgets, per-image visits,
updated parameter groups, and source/cache/reference hashes pass verification.

The common input is the RAE decoder's **post-normalization hidden features before
its RGB projection**, with 1,152 channels on a 16×16 patch grid. The cache was
checked against the published RAE for all 5,000 images, recovering its RGB
reconstructions exactly. It contains hidden features, not predicted RGB images.
All variants use the same 4,744 training and 256 validation IDs from V7.

The trainable adapter projects each patch to 128 channels using a 1×1 convolution,
then applies a residual block containing two 3×3 convolutions and SiLU activations.
Each native 32×32 pixel reads the adapted feature of its containing patch, its
within-patch pixel-centre offsets, and the original QVF global coordinate encoding.
These experiments evaluate the native 32×32 coordinate grid; arbitrary-resolution
rendering is not implemented or established by this study.

Both variants have the same three-hidden-layer, width-128 MLP with 32 outputs.
The QVF variant applies the original amplitude map, original five-qubit/depth-five
circuit and grouped RGB measurements. The classical variant uses unconstrained
MLP outputs followed by a learned affine 32-to-3 RGB projection. The latter
factorizes a conventional final linear layer and closely matches the circuit's
parameter budget. Total trainable parameters are 500,150 for QVF and 500,099 for
the classical control. Shared adapter and MLP initial weights match exactly per
seed. There is no learned post-quantum image network or RGB bypass.

Each model receives 50,000 Adam updates, with batches of 32 images and 256 sampled
pixels per image: 1.6 million image observations and 409.6 million pixel queries.
Paired variants share the exact image/pixel sampling streams. The loss is native
pixel MSE against the original images. Learning rate warms up linearly to 0.001
over 500 updates, then decays by cosine to 0.00001; there is no weight decay,
gradient clipping, VAE/KL loss, teacher-image loss, perceptual loss or GAN.

All validation images are evaluated at updates 0, 1 and every 1,000 updates.
The fixed 50,000-update result is primary; checkpoints selected by minimum
validation pixel MSE are also preserved. A fixed training subset diagnoses
overfitting during the run, and full training/validation metrics are saved for
both final and selected checkpoints. The official CIFAR test set remains unused.

This is a controlled comparison of **new readout packages on the same large
pretrained classical features**. It is not a circuit-only ablation, a matched
comparison against the old small-code QVF models, or evidence of quantum
advantage. No diffusion is trained. The published RAE remains an unchanged,
zero-shot reference, with its own original training objective and pretraining.
The feature path retains 86,582,016 frozen DINO parameters and 414,467,200 frozen
RAE decoder-body parameters. The published 885,504-parameter RGB projection is
excluded from the new prediction paths. Keeping five qubits does not remove
these classical costs or establish hardware scalability.

Implementation: [feature cache](rae_features.py), [models](rae_qvf.py),
[training and protocol](rae_readout_study.py),
[independent verification](verify_rae_readout.py),
[static figures](rae_readout_report.py). Use fresh cache/result paths:

```bash
QVF_PY=/home/agroz/anaconda3/envs/quantai/bin/python
RAE_PY=/home/agroz/.cache/quantai/rae/venv/bin/python
CUDA_VISIBLE_DEVICES=0 $RAE_PY -m experiments.qvf_project_a.rae_features \
  --output experiments/qvf_project_a/data/my_rae_features
$QVF_PY -m experiments.qvf_project_a.rae_readout_study init \
  --cache experiments/qvf_project_a/data/my_rae_features \
  --output experiments/qvf_project_a/results/my_readout50k --seeds 0 1 2
CUDA_VISIBLE_DEVICES=0 $QVF_PY -m experiments.qvf_project_a.rae_readout_study pair \
  --output experiments/qvf_project_a/results/my_readout50k --seed 0
CUDA_VISIBLE_DEVICES=0 $QVF_PY -m experiments.qvf_project_a.rae_readout_study pair \
  --output experiments/qvf_project_a/results/my_readout50k --seed 1
CUDA_VISIBLE_DEVICES=0 $QVF_PY -m experiments.qvf_project_a.rae_readout_study pair \
  --output experiments/qvf_project_a/results/my_readout50k --seed 2
CUDA_VISIBLE_DEVICES=0 $RAE_PY -m experiments.qvf_project_a.verify_rae_readout \
  --study experiments/qvf_project_a/results/my_readout50k
$QVF_PY -m experiments.qvf_project_a.rae_readout_report \
  --study experiments/qvf_project_a/results/my_readout50k
```

The paired commands can run serially. The completed study used three processes
on GPU 0, with each process sharing one feature cache across its QVF and classical
models. Do not use GPU 1. `pair --resume` restores the saved optimizer, update
count, complete sampling state and per-image visits, and skips completed variants.
An interrupted-versus-uninterrupted test verifies identical resumed parameters,
sampling histories and reconstructions. Logs are in
`results/logs/20261109_v8_readout50k_seed{0,1,2}.log`.

## Published DINO–RAE reconstruction reference (V7)

The first classical RAE baseline is complete. It uses the published **DINOv2-B
with registers encoder and ViT-XL RAE decoder**, both frozen, from the
[official RAE implementation](https://github.com/bytetriper/RAE) and
[released weights](https://huggingface.co/nyu-visionx/RAE-collections).
This is evaluation of an existing pretrained model on our dataset: **zero local
training updates**, no fine-tuning and no diffusion. It does not reproduce the
paper's ImageNet training or generation experiments.

All 4,744 training-split and 256 validation-split dog images are reconstructed,
using the exact previous seed-0 IDs. Input images remain the original CIFAR-10
32×32 images. The official encoder resizes them to 224×224, produces a
16×16×768 feature grid, and the official decoder reconstructs 256×256 RGB.
An explicitly recorded bicubic antialiased downsampling returns the output to
32×32 for scoring against the original pixels. Enlarging the input adds no
image information. Primary scores use unrounded, unclipped float outputs and
the mean of individual-image PSNR values, as in the prior experiments.

| Split (no fitting in this experiment) | Images | Native PSNR | Pixel MSE | Gradient MSE |
| --- | ---: | ---: | ---: | ---: |
| Training split | 4,744 | 25.5565 dB | 0.003463874 | 0.001258879 |
| Validation split | 256 | 25.8080 dB | 0.003081972 | 0.001233700 |

The corresponding earlier width-128 QVF-VAE selected checkpoint, seed 0, scores
20.9415 dB on these same validation images. The pretrained RAE has 65.25% lower
pixel MSE. Fixed, unselected examples visibly preserve more faces, outlines and
background detail, while some colour shifts and texture loss remain. See the
[validation examples](results/20261108_v7_rae_pretrained/validation_examples.png)
and [training examples](results/20261108_v7_rae_pretrained/train_examples.png).
Each strip shows original, pretrained RAE, then the earlier selected QVF-VAE.
The displayed originals and reconstructions are all native 32×32 pixels enlarged
with nearest-neighbour sampling; these are reconstructions, not generated images.

This is a **pretrained quality reference, not a matched architecture or
compression comparison**. The encoder has 86,582,016 parameters and the decoder
415,352,704; their total is 501,934,720. Its latent contains 196,608 entries,
versus our earlier global code's 64 (and even exceeds the original image's 3,072
channel values). It benefits from external pretraining and the authors' decoder
training recipe. Local validation exclusion does not certify that images were
absent from the external pretraining data. These results establish a working
reconstruction target; they do not isolate encoder, latent size, decoder,
pretraining or quantum effects, or prove that this is optimal for CIFAR.

The implementation uses the unedited official stage-1 model modules. Source and
weight revisions are pinned; downloaded weight hashes match upstream LFS hashes;
the entire decoder state loads strictly with no missing or unexpected keys.
Inference uses float32, deterministic algorithms and no TF32 on GPU 0 only.
The original quantai environment is unchanged: additional dependencies live in
a separate venv. This compatibility setup uses Transformers 4.51.3 and the
existing PyTorch 2.1.2, rather than the full upstream training environment.

All 29 project tests pass. Independent NumPy calculations verify all 5,000 raw
native-resolution metrics. A fresh model load reproduces every validation image
and 32 fixed training images with zero RGB error. The pinned assets, source
snapshots, split IDs and listed historical reference files also pass checks.
The official CIFAR test set remains unused. Artifacts:
[protocol](results/20261108_v7_rae_pretrained/protocol.json),
[full measurements](results/20261108_v7_rae_pretrained/results.json),
[verification](results/20261108_v7_rae_pretrained/verification.json).
Raw tensors for all reconstructions are saved beside them as `train.pt` and
`validation.pt`. No interactive report is produced, following the user's updated
preference for plain explanations and static examples.

Reproduce from the repository root, with a fresh output path:

```bash
QVF_PY=/home/agroz/anaconda3/envs/quantai/bin/python
$QVF_PY -m venv --system-site-packages /home/agroz/.cache/quantai/rae/venv
RAE_PY=/home/agroz/.cache/quantai/rae/venv/bin/python
$RAE_PY -m pip install -r experiments/qvf_project_a/requirements-rae.txt
$RAE_PY -m experiments.qvf_project_a.rae_assets
CUDA_VISIBLE_DEVICES=0 $RAE_PY -m experiments.qvf_project_a.rae_baseline evaluate \
  --output experiments/qvf_project_a/results/my_rae_reference
CUDA_VISIBLE_DEVICES=0 $RAE_PY -m experiments.qvf_project_a.rae_baseline verify \
  --output experiments/qvf_project_a/results/my_rae_reference
$QVF_PY -m unittest discover -s experiments/qvf_project_a/tests -v
```

Next research choice: measure the quality cost of a smaller spatial
representation/decoder, or use the same frozen DINO features for a controlled
classical-versus-QVF decoder comparison. Neither experiment has been run yet.
The working pretrained RAE should be retained as the quality reference.

## Existing QVF-VAE implementation

The default VAE now uses the exact decoder from the
[verified vanilla QVF baseline](../qvf_repro/README.md). `CoordinateDecoder`
wraps the original `QVFImage` class directly, with its original amplitude map,
quantum circuit and grouped RGB readout. The classical encoder and VAE objective
remain additions to QVF. The encoder predicts image codes, not a separate image
that QVF must reconstruct in latent space.

Earlier V1 and V2 experiments below used a modified RGB readout. They remain
unchanged and load through the explicit `legacy_affine` compatibility setting.
Saved configurations predating the `readout` field automatically retain that
setting; new models default to `vanilla`.

This experiment learns a shared image representation with a classical variational
encoder and a QVF coordinate decoder, then trains a classical diffusion model on
the frozen representation. Generation starts from noise and ends with QVF rendering
the generated code at image coordinates.

The first implementation uses 32 by 32 CIFAR-10 images and a global 64-number code.
The quantum circuit runs in the exact PyTorch simulator already checked against
the released QVF implementation. The experiment measures whether this hybrid
generative pipeline trains and produces useful representations and samples.

## Spatial features with the original quantum head

The completed V6 study tests a convolutional-feature decoder against an almost
parameter-matched pointwise QVF. Both use independent fitted image codes, without
an encoder or VAE loss. All six new runs completed 50,000 updates on the same
4,744 training images, over three seeds. The original width-128 V5 runs are a
preserved reference. See the [current interactive report](results/20261108_v6_verified/report.html),
[predeclared protocol](results/20261108_v6_spatial_protocol.json), and
[architecture explanation](CONVOLUTIONAL_QVF.md).

| Final training-image measurement, three seeds | Original QVF | Wider pointwise QVF | Spatial QVF |
| --- | ---: | ---: | ---: |
| Shared decoder parameters | 48,950 | 299,705 | 299,894 |
| PSNR, mean ± sample standard deviation | 22.08 ± 0.06 dB | 23.66 ± 0.04 dB | 24.27 ± 0.02 dB |
| Mean pixel MSE | 0.006805 | 0.004562 | 0.003977 |
| Adjacent-pixel gradient MSE | 0.004997 | 0.003955 | 0.003575 |

Spatial features add 0.612 dB over the matched pointwise control, with paired
gains of 0.583, 0.629 and 0.623 dB. Pixel MSE is 12.8 percent lower and gradient
MSE is 9.6 percent lower. Relative to the original small field, the spatial model
gains 2.196 dB and reduces pixel MSE by 41.6 percent. Widening the original field
also helps substantially: the evidence supports both extra capacity and the
spatial architecture package. It does not establish that missing convolutions
were the sole cause of blur. Reconstructions recover clearer structure but still
lose fine detail. Scores and visuals concern fitted training images, not unseen
images or generated samples.

The new decoder expands the same 64-number code into a 4×4×128 feature grid,
upsamples and convolves to 8×8×64 and then 16×16×32, and supplies interpolated local
features plus positional coordinates to an amplitude MLP. The original amplitude
map, five-qubit circuit, depth five and grouped RGB readout remain intact. Every
RGB output passes through that quantum head; there is no RGB bypass or
post-measurement image network. All feature grids come solely from the code.

The primary comparison fixes data IDs, initial codes and quantum angles,
coordinate encoding, image/pixel sampling streams, objective and optimizer.
Both models use the same direct-code training loop. The spatial variant changes
locality, activations and allocation of parameters together, so the comparison
does not isolate a single convolution operation. Matching parameters and updates
does not match compute. Observed training time was about 14.5–14.8 minutes per
run on shared GPU 0; it is not an isolated speed benchmark.

All six new runs and three reference runs passed
[independent verification](results/20261108_v6_verified/verification.json).
Every training image was rendered again with zero native GPU RGB error; raw
metrics, data IDs, optimizer/query budgets, RNG states, initial codes/angles,
source snapshots and unchanged baseline checkpoints were checked. All models
had their lowest recorded fitting MSE at the final update, so convergence is
not established. All 26 tests and the GPU preflights pass. Desktop/mobile UI
checks verify every gallery image crop and both measurement selections.

This supports testing the spatial decoder in the VAE next, using held-out images
to evaluate generalization. It does not yet establish practical large-scale
deployment, quantum benefit, or diffusion quality. There is still one quantum
query per coordinate; hardware state-preparation, measurement and noise costs
remain separate questions. No new VAE or diffusion model was trained in V6.

```bash
QVF_PY=/home/agroz/anaconda3/envs/quantai/bin/python
$QVF_PY -m experiments.qvf_project_a.spatial_study --tag my_spatial --seeds 0 1 2
CUDA_VISIBLE_DEVICES=0 $QVF_PY -m experiments.qvf_project_a.sweep spatial \
  --tag my_spatial --seeds 0 1 2
CUDA_VISIBLE_DEVICES=0 $QVF_PY -m experiments.qvf_project_a.verify_spatial \
  --protocol experiments/qvf_project_a/results/my_spatial_spatial_protocol.json \
  --output experiments/qvf_project_a/results/my_spatial_verified
$QVF_PY -m experiments.qvf_project_a.spatial_report \
  --data experiments/qvf_project_a/results/my_spatial_verified/results.json \
  --output-stem experiments/qvf_project_a/results/my_spatial_verified/report
```

Use fresh output paths. Seeds 0–2 reuse only the untrained quantum angles from
their V5 step-zero checkpoints; trained weights are never reused. The supplied
commands run serially, while the completed study used three processes, each
running its spatial model and then its pointwise control. Raw results are in
[results.json](results/20261108_v6_verified/results.json).

## Direct QVF fitting on the larger image set

The completed diagnostic fits the same 4,744 images with the original
QVF field and an independent learned 64-number code for each image. It removes
the image encoder, posterior sampling and VAE KL objective. The three
`20261108_v5` runs completed on GPU 0, with 50,000 updates each.
The [protocol](results/20261108_v5_direct_qvf_protocol.json) was saved before
launching them. The [direct-fitting report](results/20261108_v5_verified/report.html)
shows the matched training images and direct-fitting learning curves.

| Final training-image measurement, three seeds | QVF VAE | QVF alone, directly fitted codes |
| --- | ---: | ---: |
| PSNR, mean ± sample standard deviation | 21.52 ± 0.04 dB | 22.08 ± 0.06 dB |
| Mean pixel MSE | 0.007710 | 0.006805 |
| Adjacent-pixel gradient MSE | 0.005833 | 0.004997 |

Direct fitting improves PSNR by 0.562 dB on average, with gains of 0.654, 0.532
and 0.501 dB for seeds 0, 1 and 2. It reduces pixel MSE by about 11.7 percent
and gradient MSE by about 14.3 percent. The gallery shows more structure in many
direct fits, but substantial blur and missing fine detail remain. Removing the
VAE therefore helps without solving the larger-data reconstruction problem under
this recipe. The same quantum head can produce better fits than the current
VAE, but this result does not identify a fundamental circuit limit or isolate
the encoder from VAE regularization and optimization.

All three runs achieve their lowest recorded fitting MSE at the final 50,000
updates; the learning curves are still improving. The result is a fixed-budget
comparison, not a demonstration of convergence. The
[verification record](results/20261108_v5_verified/verification.json) confirms
exact native GPU reconstruction reloads for every training image from both model
families, raw metrics, data IDs, update/query budgets, per-image visit counts,
source snapshots and unchanged previous VAE checkpoints. Full measurements are
in [results.json](results/20261108_v5_verified/results.json).

The field retains width 128, three hidden layers, five qubits, depth five and
the original grouped RGB readout. Each run uses the same training-image IDs,
batches of 32 and 256 pixel queries per image. Its original direct-code objective
is pixel MSE plus `1e-4 * mean(code L2 norm)`. Separate Adam optimizers update
the field and code table at a constant learning rate of 0.001. Like the original
direct QVF trainer it uses no gradient clipping; like the recent VAE comparison
it keeps a constant learning rate rather than adding a plateau scheduler.

The primary comparison uses the final checkpoint and **training reconstructions
on both sides**. No held-out image supplies fitted codes or selects checkpoints.
This tests how well the shared field can fit the larger collection without an
encoder or VAE objective; it does not isolate encoder inference from VAE
regularization, nor test inference on unseen images. Each training image
contributes to its code's gradients about 337 times. A negative result at this
budget would not prove a fundamental quantum capacity limit.

All 24 numerical and integration tests pass, including the direct objective and
Adam update checked against explicit original-QVF calculations, nonsequential
image-code indexing, and a small training/checkpoint round trip. A GPU 0 preflight
also reloads exactly. No convolutions or diffusion are trained in this diagnostic.

```bash
QVF_PY=/home/agroz/anaconda3/envs/quantai/bin/python
$QVF_PY -m experiments.qvf_project_a.direct_qvf --tag my_direct --seeds 0 1 2
CUDA_VISIBLE_DEVICES=0 $QVF_PY -m experiments.qvf_project_a.sweep direct_qvf \
  --tag my_direct --select large50k --seeds 0 1 2
CUDA_VISIBLE_DEVICES=0 $QVF_PY -m experiments.qvf_project_a.verify_direct_qvf \
  --protocol experiments/qvf_project_a/results/my_direct_direct_qvf_protocol.json \
  --output experiments/qvf_project_a/results/my_direct_verified
$QVF_PY -m experiments.qvf_project_a.direct_qvf_report \
  --data experiments/qvf_project_a/results/my_direct_verified/results.json \
  --output-stem experiments/qvf_project_a/results/my_direct_verified/report
```

These commands compare against the preserved width-128 `20261108_v4` VAE runs.
Use fresh output paths. The visual gallery consists of 256 positions fixed
before training, spread across the training list; metrics and raw saved tensors
cover all 4,744 images.

## 50,000-update and fixed-qubit capacity study

The completed study extends the larger-data vanilla-QVF VAE from 7,680 to 50,000
optimizer updates, then widens the classical amplitude MLP from 128 to 512 at
the same 50,000-update budget. All six runs completed and passed verification.
The [duration/capacity report](results/20261108_v4_verified/report_ready.html) contains
the learning curves and image comparisons. The
[predeclared protocol](results/20261108_v4_capacity_protocol.json) records the
run order, configurations, data split and previous checkpoint hashes.

| Configuration | Held-out PSNR, selected checkpoint | Held-out PSNR, final checkpoint | Training PSNR, final checkpoint |
| --- | ---: | ---: | ---: |
| Width 128, 7,680 updates | 20.52 ± 0.06 | 20.51 ± 0.07 | 20.67 ± 0.06 |
| Width 128, 50,000 updates | 20.96 ± 0.02 | 20.94 ± 0.03 | 21.52 ± 0.04 |
| Width 512, 50,000 updates | 20.93 ± 0.01 | 20.22 ± 0.02 | 23.48 ± 0.10 |

Values are mean and sample standard deviation across three seeds, in dB, using
encoder mean codes. Longer training improves selected held-out PSNR by 0.439 dB.
Widening at the same budget changes that score by -0.028 dB. At the final
50,000-update checkpoint, widening improves training PSNR by 1.959 dB while
reducing held-out PSNR by 0.713 dB. This is evidence of overfitting under this
recipe: the larger model fits seen images better but reconstructs unseen images
worse as training continues. Its selected checkpoints occur at steps 6,400,
9,280 and 5,440; the smaller model selects steps 39,360, 40,960 and 40,640.

The selected wider model has slightly lower held-out edge error (gradient MSE
0.006259 versus 0.006364); at the final checkpoints it is worse (0.007284 versus
0.006371). PSNR alone is not a complete measure of visible detail. These results
show that training duration mattered and that additional pointwise width alone
does not improve held-out PSNR. They do not isolate a fundamental qubit, latent
dimension or encoder limit. Both runs retain the existing constant learning
rate and loss, so this is not a demonstration of optimal convergence.

Both variants keep five qubits, circuit depth five, the original grouped RGB
readout, the same 64-number code, encoder, KL schedule, learning rate and pixel
sampling. Each run makes 1,600,000 image observations and 409,600,000 coordinate
queries over the same 4,744 training images. The wider decoder has 588,470
parameters versus 48,950, with the same 427,872-parameter encoder. Equal update
counts do not mean equal computation: widening makes each update more expensive.
No diffusion is trained in this study.

The baseline reruns reproduce every recorded validation event through step
7,680 exactly, across all three seeds. This checks that the duration experiment
extends the same training trajectories. They start from the same seeds instead
of approximating a resume from the middle of a data-loader epoch.
The [verification record](results/20261108_v4_verified/verification.json) also
confirms exact native GPU checkpoint reloads, raw metrics and per-image scores,
training/validation IDs, source snapshots, budgets and unchanged prior checkpoints.
Full measurements are in [results.json](results/20261108_v4_verified/results.json).

At the end of V4, the separate [convolutional QVF design](CONVOLUTIONAL_QVF.md)
was an untrained numerical prototype. It generates a 16 by 16 feature grid from the
same global code, then supplies local features to the original five-qubit head.
It has 299,894 decoder parameters and passes interpolation, reconstruction
gradient, chunked-query and GPU checks. Its image quality was not measured in V4;
the completed V6 comparison is reported above. At the V4 milestone, all 21
numerical and integration tests passed, including three for this prototype.

Reproduce the duration and width comparison with a fresh tag. Finish the
baseline group before starting the wider group:

```bash
QVF_PY=/home/agroz/anaconda3/envs/quantai/bin/python
$QVF_PY -m experiments.qvf_project_a.capacity_study --tag my_capacity --seeds 0 1 2
CUDA_VISIBLE_DEVICES=0 $QVF_PY -m experiments.qvf_project_a.sweep capacity \
  --tag my_capacity --select baseline50k --seeds 0 1 2
CUDA_VISIBLE_DEVICES=0 $QVF_PY -m experiments.qvf_project_a.sweep capacity \
  --tag my_capacity --select wide51250k --seeds 0 1 2
CUDA_VISIBLE_DEVICES=0 $QVF_PY -m experiments.qvf_project_a.verify_capacity \
  --protocol experiments/qvf_project_a/results/my_capacity_capacity_protocol.json \
  --output experiments/qvf_project_a/results/my_capacity_verified
$QVF_PY -m experiments.qvf_project_a.capacity_report \
  --data experiments/qvf_project_a/results/my_capacity_verified/results.json \
  --output-stem experiments/qvf_project_a/results/my_capacity_verified/report
```

The protocol compares against the preserved `20261108_v3` runs. The report
shows all 256 held-out images and the first 64 training images in the fixed
split, for all seeds and both selected and final checkpoints. Training metrics
use all 4,744 images.

## VAE connected to vanilla QVF: completed reconstruction checks

The [vanilla-reconnection report](results/20261108_v3_verified_final/report_final.html)
shows all 50 fitting images and all 256 held-out images for each larger-data seed.
Four fresh VAE runs completed on GPU 0, with the existing encoder and VAE loss.
The encoder and field were trained from scratch; the verified vanilla checkpoint
was used to check weight compatibility, not as the initialization for these runs.

| Check | Measured PSNR |
| --- | ---: |
| Same 50 training images, encoder mean codes, final checkpoint | 31.19 dB |
| Same 50 training images, posterior samples, final checkpoint | 30.14 dB |
| Earlier direct-code vanilla fit on those 50 images, seed 0 | 29.93 dB |
| Larger-data VAE, vanilla decoder, held-out mean codes, three seeds | 20.52 ± 0.06 dB |
| Matched earlier VAE, modified decoder, held-out mean codes, three seeds | 20.53 ± 0.05 dB |

The small fitting check uses 25,000 updates, batches of 10, and all 1024 pixels,
as in the direct-code baseline. It retains our VAE's constant learning rate,
posterior sampling, gradient clipping and KL penalty. The direct-code baseline
uses a different code penalty and learning-rate scheduler, so the two fitting
scores do not establish that the VAE is a better method. This small-data VAE
overfits: its validation-selected checkpoint reaches 14.23 dB on 256 other
images, and its final checkpoint reaches 13.38 dB on them.

The larger runs match the earlier V2 experiment's 4744 training images, 256
validation images, encoder, latent size, optimizer, KL schedule, queried pixels,
and 7680 updates. The change restores the vanilla field implementation and RGB
readout. Paired changes for seeds 0, 1 and 2 are -0.098, +0.107 and -0.022 dB,
with a mean of -0.004 dB. Their training reconstructions also remain limited
(20.62 to 20.74 dB), so the limitation cannot be described solely as an error on
unseen images. Restoring the original decoder does not by itself solve the
larger-data reconstruction problem. The cause among representation capacity,
training budget, inference and VAE regularization is not isolated by this check.

All four checkpoints passed independent raw-metric, dataset-ID, budget,
checkpoint-selection and source-snapshot verification. Native GPU reloads
reproduce the checked reconstructions exactly. The existing VAE runtime permits
TF32 cuDNN convolutions in both the old and new runs. CPU versus TF32 GPU outputs
can therefore differ slightly (maximum observed 0.000378 RGB units); CPU versus
full-float32 GPU outputs agreed within 2.21e-6. The first overly strict CPU-only
verification attempt is preserved in `results/20261108_v3_verified/`; the resolved
checks are in the [final verification record](results/20261108_v3_verified_final/verification.json).
Reloading the earlier diffusion checkpoint also reproduced its original 64
generated images exactly, confirming that its RGB interpretation was preserved.

No diffusion model was trained in this update. These checkpoints provide the
verified vanilla-decoder VAE baseline for further reconstruction work. Raw
reconstructions and trained weights are saved locally under each run's `selected/`,
`final/` and `vae/checkpoints/` directories. The 50-image visual fitting comparison
uses `last.pt`; larger-data validation comparisons use `best.pt`.

The [protocol](results/20261108_v3_protocol.json) and
[aggregate results](results/20261108_v3_verified_final/results.json) record the
exact choices. Audit the completed `20261108_v3` study and render it again using
fresh output paths:

```bash
QVF_PY=/home/agroz/anaconda3/envs/quantai/bin/python
CUDA_VISIBLE_DEVICES=0 $QVF_PY -m experiments.qvf_project_a.verify_vanilla_vae \
  --tag 20261108_v3 --output experiments/qvf_project_a/results/my_vanilla_verified
$QVF_PY -m experiments.qvf_project_a.vanilla_vae_report \
  --data experiments/qvf_project_a/results/my_vanilla_verified/results.json \
  --output-stem experiments/qvf_project_a/results/my_vanilla_verified/report
```

The verifier defaults to the completed study's protocol and uses GPU 0 to check
the original convolution precision. A custom tag needs its corresponding
`<tag>_protocol.json` recording the run list and reference hashes.

## Earlier experiments with the modified decoder

The earlier [first pilot report](results/20261108_v1_report_final.html) contains the measured
results, training curves, reconstructions and generated images. The QVF VAE reached
19.35 dB validation PSNR using mean codes and 19.20 dB using sampled codes on 256
validation images after training on 1024 images. The selected diffusion checkpoint
was step 750; further training increased validation loss. Samples remain coarse
and blurry. The three initial controls reached 19.16 to 19.41 dB using mean codes,
so this single seed does not establish a meaningful ranking.

Eighteen numerical and integration tests pass, including exact five-qubit vanilla
output and gradient checks, VAE reconstruction gradients, checkpoint compatibility,
and a complete small CPU pipeline using the vanilla decoder. Reloading the earlier QVF VAE diffusion
checkpoint in a separate GPU 0 process reproduced the same 64 latent codes and
images exactly with the same seed and software environment.

The [reconstruction study](results/20261108_v2_study.html) compares data size,
training duration and classical model capacity across three seeds. Increasing
training from 1024 to 4744 images helps at the longer training budget: validation
PSNR rises from 19.71 to 20.53 dB with the same five-qubit circuit. More data at the
short budget makes little difference. The report includes image sliders and a
separate diagnostic that fits image codes while keeping QVF frozen.

## Run the experiment

Run commands from the repository root. Use the existing environment without
upgrading its packages. GPU 0 is the only permitted GPU; GPU 1 belongs to other work.
The data loader reuses the existing CIFAR-10 download in
`experiments/qvf_repro/data`. It does not download or modify the data.

```bash
QVF_PY=/home/agroz/anaconda3/envs/quantai/bin/python

# Numerical tests and a complete small CPU integration test; no download needed.
$QVF_PY -m unittest discover -s experiments/qvf_project_a/tests -v

# VAE with the vanilla decoder, reconstruction only. Use a fresh tag.
# Same 50 images as the vanilla fitting baseline, 25,000 full-image updates.
CUDA_VISIBLE_DEVICES=0 $QVF_PY -m experiments.qvf_project_a.sweep vanilla_vae \
  --tag my_vanilla --select fit50 --seeds 0

# 4,744 training / 256 validation images; same protocol as the earlier larger run.
CUDA_VISIBLE_DEVICES=0 $QVF_PY -m experiments.qvf_project_a.sweep vanilla_vae \
  --tag my_vanilla --select larger --seeds 0 1 2

# Small GPU integration run: 32 training and 16 validation images.
CUDA_VISIBLE_DEVICES=0 $QVF_PY -m experiments.qvf_project_a.sweep smoke --tag my_smoke

# Pilot: 1024 training and 256 validation images, 80 VAE epochs,
# then 4000 diffusion optimizer steps. Each run needs a fresh tag.
CUDA_VISIBLE_DEVICES=0 $QVF_PY -m experiments.qvf_project_a.sweep pilot --tag my_pilot

# Inspect the four control configurations without launching them.
$QVF_PY -m experiments.qvf_project_a.sweep controls --tag my_controls --dry-run

# Run the remaining controls if the QVF VAE pilot already exists.
CUDA_VISIBLE_DEVICES=0 $QVF_PY -m experiments.qvf_project_a.sweep controls \
  --tag my_controls --select no_circuit_vae mlp_vae qvf_ae

# Optional longer run, still recorded as a named experiment.
CUDA_VISIBLE_DEVICES=0 $QVF_PY -m experiments.qvf_project_a.sweep pilot \
  --tag longer_pilot --vae-epochs 160 --diffusion-steps 8000
```

Use `--device cpu` for CPU execution. `--seeds 0 1 2` runs separate seeds
sequentially. Existing output directories are rejected instead of overwritten.
`FAILED.json` identifies an interrupted or failed run; `COMPLETE.json` is written
only after the requested stages finish (both training stages and generation for
the full pipeline, representation training for the reconstruction group). Checkpoints contain
optimizer and random-generator states, but this version has no resume command.

Generate another sample batch from a completed run with:

```bash
CUDA_VISIBLE_DEVICES=0 $QVF_PY -m experiments.qvf_project_a.sample \
  --checkpoint experiments/qvf_project_a/results/my_pilot_pilot_qvf_vae_seed0/diffusion/checkpoints/best.pt \
  --output experiments/qvf_project_a/results/my_pilot_pilot_qvf_vae_seed0/another_sample_batch \
  --count 64 --seed 1234
```

The diffusion checkpoint includes the autoencoder weights and latent normalization
statistics, so generation does not require the original images or latent cache.
Only load trusted checkpoints: PyTorch checkpoint loading uses pickle.

## Representation training

The image encoder has three convolutional layers and two output heads. One predicts
64 means and the other 64 log-variances. The latter starts at -4 and is bounded to
[-12, 6] for numerical stability. Each image gets one sampled code per optimizer
step. That code is reused across all pixel queries for the image.

The QVF decoder concatenates the code with encoded coordinates and applies the
classical amplitude network, followed by the quantum circuit and RGB readout.
The default field has three hidden layers of width 128, six positional frequencies,
five qubits and five circuit layers with Gaussian initialization. With
`readout="vanilla"`, the decoder executes the unchanged `QVFImage` implementation:
its original Boltzmann normalization and square root produce the amplitudes, and
the released grouping produces RGB. At five qubits the groups are `Z0 + Z3`,
`Z1 + Z4`, and `Z2`. No extra shift, scaling, sigmoid or clipping is applied to the
training predictions. Coordinates follow the released row/height and column/width
convention, with no extra factor of pi. Metrics use raw predictions; image files
are clipped only for display.

The older `legacy_affine` option computes amplitudes using
`exp(0.5 * log_softmax(scores))` and maps the first three Z expectations from
[-1, 1] to [0, 1]. It exists to reproduce earlier results and load their checkpoints.
The two amplitude expressions describe the same mathematical map but use different
floating-point operations. The vanilla default calls the original code directly.

The reconstruction objective averages squared error over images, queried pixels
and RGB channels. KL divergence to a standard Gaussian is summed over the 64
latent entries and then averaged over images. The default total objective is
reconstruction MSE plus `1e-5 * KL`, with a ten-epoch linear KL warmup. This is a
weakly KL-regularized VAE, not the unweighted textbook ELBO under an implicit
unit observation variance. Changing these reductions changes the meaning of
the KL weight.

Larger-data training queries 256 uniformly sampled pixel coordinates per image per step.
The `vanilla_vae` group's `fit50` check uses all 1024 pixels.
Validation renders all 1024 pixels. The best checkpoint minimizes validation
sampled-code MSE plus the final KL weight times validation KL; the same validation
sampling seed is used each time. Both mean-code and sampled-code PSNR are reported.
PSNR is computed per image and then averaged, with image data range one.

CIFAR-10 dog images are split by a fixed random permutation into training and
validation images. Their original dataset indices are recorded in `split.json`.
The official test set is not read. Validation selects checkpoints, so its results
are development measurements rather than a final test-set assessment. The
training mean image and shuffled image codes provide basic reconstruction controls.

## Latent diffusion training

After representation training, the selected encoder and decoder are frozen. The
cache stores means and log-variances separately for training and validation images.
Diffusion draws fresh samples from those posteriors during training. In the
deterministic autoencoder control, it uses the means without sampling.

Normalization uses only the training posterior mixture. For each latent entry,
its variance includes both the variance of encoder means across images and the
average posterior sampling variance. Generation reverses this normalization
before passing a code to QVF.

The denoiser is a classical residual MLP with width 256, four residual blocks and
sinusoidal timestep features. The DDPM has a 100-step cosine noise schedule,
velocity prediction and fixed posterior sampling variance. Velocity is a standard
combination of clean code and noise used as the prediction target; it avoids
dividing by a nearly zero signal coefficient at high noise levels. Latent values
are not clipped to the RGB range.

An exponential moving average of the denoiser weights is used for validation and
generation. Its target decay is 0.995 with an initial warmup. Validation uses four
fixed posterior and noise draws per held-out image. The best checkpoint minimizes
that validation denoising loss. This loss does not measure generated image quality.

Diffusion training uses only the cached numerical codes. It executes no QVF
circuits and does not rerun the image encoder. During generation, the denoiser runs
for 100 steps and QVF then renders the resulting code. The two stages are trained
separately; there is no adversarial or perceptual loss in this first pilot.

## Outputs and controls

Every run records the configuration, environment, dataset indices, source hashes
and a source snapshot. It also writes training histories, selected and final
checkpoints, latent statistics and sample images. Heavy checkpoints, caches and
source snapshots are ignored by git; metrics and preview images remain reviewable.

- `vae/validation_reconstruction.png`: original validation images, reconstructions
  from mean codes, then reconstructions from sampled codes, in three rows.
- `generation/diffusion_samples.png`: samples generated by the learned diffusion model.
- `generation/fitted_gaussian_samples.png`: a simpler baseline using independent
  Gaussian entries with the training latent means and variances.
- `generation/standard_normal_samples.png`: samples from the VAE's standard Gaussian prior.
- `generation/nearest_training_images.png`: generated images in the first row and
  their nearest training images in pixel distance in the second row. This is a
  visual diagnostic, not a proof of novelty or absence of memorization.

The `controls` group includes QVF with a VAE, the same amplitude network with the
circuit removed, a sigmoid-output classical MLP decoder with a VAE, and QVF with
a deterministic encoder. They share data splits and training budgets. The plain
MLP is an architectural comparison, not an exactly parameter-matched control.

The tests check agreement with the existing QVF decoder under the declared RGB
mapping, reconstruction gradients through the stochastic encoder and circuit,
Gaussian KL, disjoint data splits, latent normalization, DDPM identities and
posterior variance, checkpoint reload, repeatable sampling and stage separation.

## Reconstruction study: more data, capacity and code fitting

The study completed 18 representation-training runs: six settings with seeds
0, 1 and 2. All use the same 256 validation IDs from the original pilot. The larger
training set contains all 4744 remaining CIFAR-10 dog images from the official
training set, including the original 1024 training IDs. Validation images never
enter a training batch, and the official test set remains unused.

Each optimizer step uses 32 images and 256 coordinates per image. The short budget
is 2560 steps (81,920 image observations and 20,971,520 coordinate queries); the
long budget is 7680 steps (245,760 observations and 62,914,560 queries). An
observation can be a repeat of a training image. Incomplete batches are dropped
to keep each step the same size; all 4744 images were nevertheless seen in every
larger-data run. The two budgets correspond to about 17.3 and 51.8 passes through
the larger dataset, versus 80 and 240 passes through the smaller dataset.

KL warmup lasts 320 optimizer steps, and validation occurs after step 1 and every
320 steps. The original pilot used epoch-based warmup, so a fresh baseline was
trained under this common step-based protocol. Each longer run exactly reproduces
its shorter counterpart's validation trajectory for the first 2560 steps.
Checkpoints use the same selection rule: minimum sampled-code validation MSE plus
the final KL weight times KL. The reported PSNR uses encoder mean codes.

| Setting | Training images | Optimizer steps | Validation PSNR, dB |
| --- | ---: | ---: | ---: |
| Baseline | 1024 | 2560 | 19.32 ± 0.04 |
| More images | 4744 | 2560 | 19.35 ± 0.11 |
| Longer training | 1024 | 7680 | 19.71 ± 0.04 |
| More images and longer training | 4744 | 7680 | 20.53 ± 0.05 |
| 128-number code instead of 64 | 1024 | 2560 | 19.45 ± 0.05 |
| Field width 256 instead of 128 | 1024 | 2560 | 19.60 ± 0.05 |

Values are means and sample standard deviations across three training seeds on
one fixed validation split, not confidence intervals. More data improves PSNR by
0.74–0.85 dB across the three seeds at the longer budget (mean gain 0.81 dB). At
the short budget, the gains range from −0.10 to +0.13 dB (mean +0.03 dB). Combining
more data with longer training improves the baseline by 1.20 dB on average.

All settings retain five qubits and circuit depth five. Increasing the code size
also increases the encoder output layers and field input layer; increasing field
width adds classical parameters. Those capacity comparisons match optimizer
steps and queried coordinates, not FLOPs. Concurrent GPU jobs also make elapsed
times unsuitable as clean speed benchmarks. The two data-size comparisons use
identical architectures and training budgets.

The frozen-decoder diagnostic starts with the encoder's mean code and optimizes
only that image's code for 400 Adam steps at learning rate 0.03. It measures full
32 by 32 image MSE against the target photograph. Both networks and the source
checkpoint are verified unchanged.

| Seed-0 source model / fitting objective | Encoder code, dB | Fitted code, dB | Gain, dB |
| --- | ---: | ---: | ---: |
| Original pilot / pixel MSE | 19.35 | 21.48 | 2.13 |
| Original pilot / pixel MSE and mean-code prior penalty | 19.35 | 21.40 | 2.06 |
| More data and longer training / pixel MSE | 20.58 | 21.54 | 0.96 |

These are averages over the same 256 validation images. The prior check adds
`0.5 * 1e-5 * sum(code**2)`, the mean-dependent part of the Gaussian KL. It does
not reproduce the full stochastic VAE objective. The image bootstrap intervals
in the raw diagnostic output describe uncertainty over images conditional on one
model, not variation across training seeds.

Code fitting shows that the frozen QVF can reconstruct these photographs better
than it does with the encoder's initial codes. The smaller fitting gap after
more-data training is consistent with better code inference. It does not isolate
encoder learning from changes in the jointly trained decoder, establish an optimal
code, or establish the decoder's maximum capacity. Fitted codes use the known
target photograph repeatedly; their PSNR is not a feed-forward inference or
unconditional generation result.

The strongest tested representation is the larger-data, longer-training setting.
Reconstructions remain blurry. Diffusion was **not retrained** in these 18 runs,
so generation quality for the improved representation remains unmeasured. The
next diffusion comparison should use a newly cached representation and a newly
trained denoiser, with matching training and validation splits.

Reproduce the study with fresh output names:

```bash
CUDA_VISIBLE_DEVICES=0 $QVF_PY -m experiments.qvf_project_a.sweep reconstruction \
  --tag my_reconstruction --seeds 0 1 2 --device cuda:0

# Select individual settings to run a smaller comparison.
CUDA_VISIBLE_DEVICES=0 $QVF_PY -m experiments.qvf_project_a.sweep reconstruction \
  --tag my_data_comparison --select baseline more_data longer more_data_longer \
  --seeds 0 1 2 --device cuda:0

# Default source is the original pilot; source weights remain frozen.
CUDA_VISIBLE_DEVICES=0 $QVF_PY -m experiments.qvf_project_a.refine \
  --output experiments/qvf_project_a/results/my_refinement \
  --steps 400 --lr 0.03 --prior-weight 0

# Build portable and local-assets HTML plus audited JSON from the completed study.
$QVF_PY -m experiments.qvf_project_a.study_report \
  --output-stem experiments/qvf_project_a/results/my_study_report
```

`--reference-run` on the reconstruction sweep selects the original configuration
and locked validation split; `--run` on the refinement command selects its frozen
checkpoint. The report command defaults to the completed `20261108_v2` runs and
`20261108_reconstruction_v2_refine` diagnostics; `--tag` and
`--refinement-prefix` select another complete study with the same protocol.

The [aggregate JSON](results/20261108_v2_study.json) preserves per-seed results,
paired comparisons and learning curves. The
[verification record](results/20261108_v2_study_verification.json) confirms exact
splits, observed training-image coverage, step/query budgets, checkpoint selection,
matching source-snapshot hashes and unchanged original quantum-simulator code.

## Research scope

This pilot tests a global latent code on small images. It does not implement a
spatial latent grid, text conditioning, Project B's quantum denoiser, finite-shot
measurements or quantum hardware execution. The initial generation runs use one
seed; the reconstruction study uses three. These experiments cannot establish
sample quality, scalability or quantum advantage. Reconstruction quality
limits the detail the decoder can produce regardless of diffusion quality.

Further work includes diffusion on the improved representation, matched decoder
controls at the larger data budget, KL-weight comparisons, and evaluation of
sample quality and diversity. A spatial latent grid would be a separate
architectural extension.

The starting points are [QVF](https://arxiv.org/abs/2508.10900),
[variational autoencoders](https://arxiv.org/abs/1312.6114),
[latent diffusion](https://arxiv.org/abs/2112.10752),
[DDPM](https://arxiv.org/abs/2006.11239), the cosine schedule from
[Improved DDPM](https://arxiv.org/abs/2102.09672), and the velocity parameterization
described in [Progressive Distillation](https://arxiv.org/abs/2202.00512).
