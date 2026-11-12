# Project Z: spatial quantum implicit neural representations

Named by the user on 2026-11-10. Z is the representation/decoder research before
Project A's encoder and diffusion. The objective is a useful quantum INR with
better spatial representation than vanilla QVF, explicit quantum execution costs,
and a credible path to larger datasets and genuine higher-resolution images.
Classical controls support attribution; beating classical image models is not a
prerequisite. Historical V9–V11 artifacts stay in `../qvf_project_a/` unchanged.

## Z8: all methods plus vanilla QVF at both resolutions

The user explicitly selected **50,000 updates at both32×32 and256×256,all13
Z6/Z7 variants plus vanilla QVF,and three seeds**, on2026-11-12. The separate Z8
queue was launched at05:50UTC,PID2202707,and is waiting for the complete Z7
pipeline to finish before taking two GPU0 worker slots. GPU1 remains reserved.
Read the [live queue](results/20261112_z8_complete_methods/queue_status.json)
before reporting progress. The current measurements are not complete.

There are84 method/dataset/seed entries:39 existing256px runs are reused after
their audits finish;45 fresh runs comprise42 full32px runs and three256px vanilla
QVF runs. Both plots always list all14 methods. Pending cells are not filled from
the older2,000-update pilot. See the [current report](results/20261112_z8_complete_methods/REPORT.md),
[coverage manifest](results/20261112_z8_complete_methods/manifest.json),and
[Git-friendly exported records](../../reports/qvf_benchmarks/README.md).

The32px panel uses native CIFAR-10 dogs: the established4,744 training and256
development IDs,plus64 predeclared official-test dog images. Exact selected test
duplicates against train/development are excluded; train/development membership
is preserved and near duplicates are not ruled out. The256px panel uses exactly
the Z6/Z7 Stanford Dogs data,masks and query streams. Within each dataset,all
methods share50k updates,seeds0/1/2,batch16,Adam0.001,RGB+0.5edge training,and
1,000-step frozen-decoder new-code fitting. Coordinate endpoints/image/update
are128 at32px and2,048 at256px. Completion sees25% of pixels and scores the rest.
Full-image new-code reconstruction and a256-image training probe are also saved.

`VanillaField` calls the unchanged audited`QVFImage` during training. It retains
five qubits,five original circuit repetitions,neural amplitude encoding,original
grouped RGB readout,and64 global code values per image. Positional frequencies
are six/nine at32/256px to match the other methods. The benchmark uses Z's common
optimization/loss recipe,not the paper's original training schedule. Spatial
methods have336/21,504 classical code values,so gains cannot be attributed solely
to spatial retrieval or called equal-storage gains. The two panels use different
datasets and do not isolate resolution scaling.

Three numeric tests passed,including original-model outputs/gradients,cached
evaluation,native PennyLane states/gradients,and independent primitive execution.
All45 discarded synthetic runs passed training,checkpoint reload,stream replay,
independent metrics,finite shots,gate-noise tests and reporting. A short GPU0
preflight checked the full256px vanilla workload and32px coherent workload.
These are correctness checks,not quality measurements. The112 Python source
hashes are frozen; Z4/Z5/Z6/Z7 sources remain unchanged.

The detached queue automatically runs`coverage_verify`,`coverage_audit` and
`coverage_report`; final completion requires all checks and all84 entries.
New modules are`coverage_models.py`,`coverage_data.py`,`coverage_study.py`,
`coverage_verify.py`,`coverage_audit.py`,`coverage_run.py`,`coverage_report.py`,
`coverage_preflight.py`,`coverage_export.py`,and`test_coverage.py`.
`coverage_export` copies compact records for Git without publishing datasets or
checkpoints. It does not commit or push by itself.

Vanilla QVF uses **red** (`#d62728`) labels, points and scores in both plots;
the other methods use teal. The presentation-only `coverage_plot_style` module
redraws saved `comparison.json` values without changing frozen experiment code.
A separate CPU companion runs it with `--watch` beside the Z8 queue and reapplies
the style after report updates; it exits when the queue finishes or stops.
After a manual `coverage_report` refresh, run
`python -m experiments.qvf_project_z.coverage_plot_style --root Z8_ROOT`
before `coverage_export` to keep the exported PNG/PDF files red as well.

## Z7: coherent scale-routing benchmark running

Status checked **2026-11-12 at 04:11 UTC**: six runs complete, two active near
50k updates, four pending, zero failures. Full verification/quantum audits and
extended development fits remain pending. On the same first two seeds,
completion PSNR is23.190dB for coherent routing,23.152 without phase gates,
22.994 for the old independent scale heads,and24.282 for attention5q.
Coherent routing improves the independent-scale reference by0.196dB on average,
with a positive gain in both seeds,but remains below the stronger Z6 references.
Phase effects are mixed.
These are provisional results,not the final three-seed comparison. Independently
recomputed saved completion metrics for six models × two seeds agree within
1.07e-14; all102 Z7 and88 Z6 source hashes remain unchanged. Timestamped evidence:
[status snapshot](results/20261111_z7_coherent/STATUS_20261112_0411UTC.json).

The user requested vanilla QVF in the comparison plot. The
[updated plot](results/20261111_z7_coherent/STATUS_20261112_0411UTC_with_QVF.png)
includes a separate historical panel: verified V9 vanilla64/global336/token
attention on CIFAR-10 dogs at 32×32, 1,024 training images, 2,000 training updates,
and **500 code-fitting updates**, as recorded in its frozen protocol. Vanilla
QVF has no matched 256×256 Stanford Dogs result; the current panel explicitly
marks that score unmeasured. Do not subtract scores across the panels or describe
Z6's spatial/attention references as vanilla QVF. No new vanilla training was
launched for this plot request. The plotting script and source hashes are saved
beside the figure, outside frozen training sources.

Launched **2026-11-11 at 09:47 UTC** on physical GPU0, with two concurrent workers.
Runner PID at launch:1982870. The user authorized training and benchmarking the
coherent prototype against the previous study. **The full Z7 comparison is not
complete.** Read the [live queue](results/20261111_z7_coherent/queue_status.json)
and [runner log](results/20261111_z7_coherent/runner.log) before reporting status.
The [frozen protocol](results/20261111_z7_coherent/protocol.json) matches Z6's
4,744 photographs at256×256,50,000 updates,seeds0/1/2,latent grids,coordinate
samples,loss and primary evaluation. GPU1 is reserved for the user's other work.

There are **12 fresh runs: four conditions × three seeds**. All27 completed Z6
runs are reused as verified references; they are not retrained or used as initial
weights. The main references include separate scale heads,the three all-scale
heads,the spatial reference,joint statistics,and the five-qubit attention model.

| New condition | What it tests |
| --- | --- |
| `coherent_routing` | Nine-qubit model with content-controlled mixing and relative phases between scale branches, before later quantum processing |
| `real_routing` | Same coherent routing, with the eight added phase angles fixed to zero; this is still a coherent quantum model |
| `local_capacity9` | Same nine-qubit register and16 additional trainable quantum parameters, applied to the payload without mixing scale branches |
| `classical_attention16` | Original three independent7q heads with16 additional classical parameters that weight their measured RGB outputs using content,position and scale |

The capacity control tests extra parameters without non-additive cross-scale
interaction; it does not match primitive gate count. Classical attention is
output fusion,not feedback or attention over grid cells. Parameter counts refer
to stored/trainable entries,not independent functional degrees of freedom.
Routing disabled already reproduces the original three-head mean numerically.
The completed Z6 mean-fusion model supplies that trained reference.

Primary evaluation retains Z6's1,000-step frozen-decoder new-code budget and
fixed25%-observed completion mask. An additional **development-only** experiment
compares coherent routing,the old separate-scale model,and attention5q after
1,000,5,000 and10,000 code updates, with the optimizer continued between points.
This tests whether slower code fitting helps explain a new-image deficit without
changing the primary test budget. Z6 test scores were inspected before Z7 design:
this is benchmark reuse,not a fresh confirmatory holdout.

After training,the queue automatically runs checkpoint/stream/metric verification,
native quantum state and gradient checks,finite-shot evaluation,both per-circuit
and equal-total-shot budgets,synthetic gate-noise diagnostics,the longer development
fits,and reports with static figures. Noise uses four test images ×32 predeclared
withheld coordinates at seed0; old separate-scale and attention references are
re-evaluated on that exact subset. The classical softmax readout is nonlinear,
so its finite-shot loss uses measurement replicates rather than the linear RGB
expectation formula. Tiny trained-checkpoint dephasing checks test coherence
sensitivity; they are not a separately trained dephased competitor.

Before launch,three new numeric tests and all12 discarded synthetic end-to-end
runs passed,including verification,quantum audits,nine extended development fits,
and report generation. See [tests](results/routing_tests_20261111.log),
[smoke status](results/20261111_z7_smoke/queue_status.json),
[full-table preflight](results/routing_preflight_20261111.json),and
[launch validation](results/20261111_z7_coherent/launch_validation.json).
The preflight's largest allocated GPU memory was7,724.3MiB/job; only two workers
run concurrently. These checks establish implementation correctness,not image
quality. All102 source hashes are frozen,including unchanged Z6/Z5/Z4 sources
and the original coherent prototype. Do not edit them while this study runs.

New implementation and harness: `routing_models.py`,`routing_gates.py`,
`routing_study.py`,`routing_run.py`,`routing_verify.py`,`routing_audit.py`,
`routing_inversion.py`,`routing_report.py`,`routing_preflight.py`,
`routing_smoke.py`,`test_routing.py`. Completion requires queue phase`complete`,
`verification.json`,`quantum_audit.json` and`extended_development.json` all marked
complete,and`REPORT.md`. Results will be in`results/20261111_z7_coherent/`.

## Coherent communication between scales: validated architecture

User direction, 2026-11-11: combine the three scale branches with attention-like
information exchange, preferably **inside the quantum computation**. A classical
attention mixer of measured features is now a Z7 control; a two-round
measurement-feedback model remains future work. The coherent model uses
controlled operations and phase interference, with no intermediate measurement.
It is attention-inspired content-dependent routing; it does not implement
Transformer query/key dot products and softmax, and no novelty claim is made.

Implementation: [quantum_scale_attention.py](quantum_scale_attention.py).
Seven tests pass in [test_quantum_scale_attention.py](test_quantum_scale_attention.py),
with [recorded output](results/coherent_scale_tests_20261111.log).
[Full-table GPU0 preflight and compiled resources](results/coherent_scale_preflight_20261111.json).
The prototype was validated before the Z7 quality study described above launched.

The nine-qubit register contains two scale-address qubits and a shared seven-qubit
payload. Address states00/01/10 select coarse/medium/fine processing; state11
remains unused. All three original seven-qubit branch parameter sets and their
scale-specific inputs are retained as controlled blocks acting on the shared
payload. This is a direct-sum/multiplexed architecture, not three independent
seven-qubit registers and not their full21-qubit tensor-product state. It cannot
be executed as three independent parallel jobs. A literal coherent interaction
among three full seven-qubit registers would need a larger connected quantum
system, or a suitable quantum communication implementation.

At each of five processing stages, the scale address controls the feature uploads
and learned branch gates. Between stages, relative phases and rotations of the
address are controlled by a centre-data qubit and a work qubit. Thus image-dependent
quantum content affects how scale branches mix before later processing. RGB is
measured only at the end using the original five payload observables. There is
no new classical attention network or classical RGB decoder. The within-scale
input retrieval is still the Z6 local interpolation; attention over grid cells
is a separate, unimplemented factor.

Interleaving matters: a unitary acting only on the scale address after all branch
processing cannot affect RGB when that address is traced out. The prototype mixes
before later conditional processing. Disabling routing reproduces the existing
three-head mean exactly, including with nonzero address phases. Tests check that
relative phases and removal of address coherence can change the measured RGB;
these are synthetic sensitivity checks, not quality or quantum-advantage evidence.
A separate [interaction diagnostic](results/coherent_scale_interaction_20261111.json)
checks whether the response to coarse features depends on medium features.
The mixed input derivative is nonzero with routing and exactly zero when routing
is disabled, demonstrating a non-additive interaction between scales. Applying
the actual content-controlled mixer only at the end leaves RGB unchanged to
1.95e-16; placing it between processing stages is necessary for this readout.

Validation covers native PennyLane states and all quantum/input gradients,
independent RY/RZ/X/CNOT compilation, norm preservation and the unused address,
the no-routing reduction, coherence removal with a density simulator, joint
finite-shot statistics, full-field backpropagation and checkpoint reload.
Existing environments were not upgraded. The compiled circuit uses merged
rotation multiplexers: **1,000 RY, eight RZ, two X and 994 CNOT gates per queried
coordinate**, all-to-all primitive depth1,680 including preparation. This is one
valid construction, not an optimized lower bound; device routing/noise is not
included. The three independent Z6 seven-qubit heads use360 CNOTs in total, so
the nine-qubit encoding reduces register width versus a full21-qubit design but
does not make controlled execution free. There are16 new quantum parameters and
no added classical parameters; total shared parameters40,694, of which181 are
stored quantum entries.

The discarded10-update synthetic GPU0 test used4,744 full-size code rows,
batch16 and2,048 query coordinates/image. After two warmup steps, independent
heads averaged0.1116s/update and3,683.7MiB peak allocated memory; the coherent
prototype averaged0.2033s and7,188.3MiB. Routing/phase gradients were finite and
nonzero. These measurements exclude the CUDA context, concurrent jobs and full
evaluation, and do not establish training quality or hardware feasibility.

Z7 compares the original three-head mean,classical attention fusion,and coherent
routing,with the controls and cost-aware checks described above. Measurement
feedback remains future work. Preserve Z6 as a frozen completed study.

## Z6: four quantum designs on genuine 256×256 images

**Completed 2026-11-11: all27 runs, checkpoint verification, quantum audits and
reports passed.** No Z6 job remains active. The study took about15.06hours for the
complete pipeline, using GPU0 only with up to five concurrent workers. The user
added their three-circuit design as a fourth condition and requested genuinely
higher-resolution photographs. Read the [full report](results/20261110_z6_hires/REPORT.md)
and [paired comparisons](results/20261110_z6_hires/comparison.json).
Current status: [queue_status.json](results/20261110_z6_hires/queue_status.json).
Frozen settings: [protocol.json](results/20261110_z6_hires/protocol.json).

Three-seed completion PSNR means: spatial reference23.896dB, joint statistics23.945,
successive scales23.488, complex phases23.860, separate scale heads23.257,
three all-scale heads23.967, attention5q24.328. Joint statistics gains0.049dB over
the spatial reference and0.078dB over marginal-product readout, both3/3 paired
seeds; mean edge error does not improve over the spatial reference. Separate
scale heads trail the reference by0.639dB and three all-scale heads by0.709dB,
both3/3 seeds, under the fixed1,000-step new-code budget. Phase changes do not
show a clear benefit. The five-qubit attention context baseline is strongest,
but also differs in retrieval, encoding and circuit design, so it is not an
isolated comparison of one versus three heads. The earlier two-seed
[status snapshot](results/20261110_z6_hires/STATUS_20261111_0400UTC.json) is historical.
Checkpoint replay max error1.02e-6; independent metric error1.43e-14; native quantum
state/gradient discrepancies2.11e-15/4.67e-15. All88 frozen source hashes remain
unchanged. Full diagnostics are in `verification.json` and `quantum_audit.json`.

| Primary condition | Quantum change | Attribution control |
| --- | --- | --- |
| Joint measurement statistics | Five individual Z means plus four centre–neighbour ZZ means from the same bitstrings; learned affine RGB map | Same RGB map using products of individual means |
| Successive scales, one circuit | Coarse/coarse/medium/medium/fine uploads into one seven-qubit state | Shared spatial head receiving all scales at every upload |
| Complex phase gates | Trainable RZ and pairwise IsingZZ gates, with subsequent real rotations that make phases observable | Same number of added stored parameters and CNOTs using real gates |
| Three circuits, one per scale | Three independent seven-qubit heads, with fixed-average RGB fusion | Three independent heads each receiving all scales |

All four are separate experiments. The common seven-qubit spatial reference and
the earlier five-qubit attention-chain family are also trained on the new data.
The latter is a context baseline with a different feature interface. The other
three controls above give nine conditions, each with seeds 0, 1 and 2. Three
heads can reuse seven physical qubits sequentially, or use 21 simultaneously;
there is no entanglement between their three registers.

**Data and optimization.** Stanford Dogs photographs, centre-cropped and
downsampled to 256×256 with no upsampling; both original dimensions must be at
least 256. Selected 4,744 training images and 256 development images from the
official-training split supplied by the pinned mirror; 64 images are reserved
from its official-test split. The original Stanford server was unavailable.
Source mirror `maurice-fp/stanford-dogs`, revision
`ee102b9a35f00829e65bdad7cda8adcfa185b400`; JPEG and array hashes, original sizes,
crop boxes and row IDs are in
[the data manifest](data/stanford_dogs_maurice/prepared_256/manifest.json).
Exact selected duplicates are excluded; near duplicates and original MATLAB
split membership have not been independently checked. The older failed mirror
extraction is retained separately for provenance; no package was installed or
upgraded to prepare the data.

Each run uses 50,000 updates, batch16, 2,048 coordinate endpoints per image/update,
the existing RGB-plus-edge objective, and Adam at 0.001. The coordinate budget
is 1,638,400,000 per run: approximately 5.27 presentations per training pixel on
average. Three-head models execute three times as many quantum queries. Latent
grids are 16×16, 32×32 and 64×64 with four channels each: **21,504 classical
code values/image**, shared across all conditions. Nine positional frequencies
and sparse four-corner interpolation support the higher resolution. This is a
substantial classical code budget, not a seven-qubit storage/compression claim.
The shared MLP remains width128 with three hidden layers; no DINO, encoder or
large convolutional decoder is used.

**Evaluation.** Training reconstruction is measured on a fixed 256-image probe.
New-image fitting uses 64 predeclared development images and all 64 reserved test
images, always freezing the decoder. New codes receive 1,000 updates. Completion
observes only a fixed 25% pixel mask and scores withheld pixels/edges; a separate
test fit observes all pixels. These are optimized codes, not encoder inference.
Full native-resolution reconstructions and fixed detail crops are saved.
Finite-shot tests use both equal shots per head and equal total shots, on 16 test
images × 256 predeclared withheld coordinates. Gate-noise simulation uses four
test images at seed0 only. No hardware feasibility or quantum advantage is claimed.

**Validation before launch.** Five numeric tests passed, including dense-versus-
sparse interpolation, native PennyLane states and gradients, phase visibility,
distinct joint information, and exact finite-shot MSE for the nonlinear marginal-
product control. All 27 discarded synthetic end-to-end runs passed reconstruction
reload, stream replay, independent metrics, quantum audits and report generation:
[smoke status](results/20261110_z6_smoke/queue_status.json).
Full-table GPU preflight at 2,048 queries used roughly 2.0–3.6 GiB allocated per job:
[measurements](results/hires_preflight_2048.json). These are correctness/timing
tests, not quality results. Z4's 61 and Z5's 77 frozen source hashes were checked
unchanged at launch.

New code is in `hires_models.py`, `hires_gates.py`, `hires_data.py`,
`hires_study.py`, `hires_run.py`, `hires_verify.py`, `hires_quantum_audit.py`,
`hires_report.py`, `hires_preflight.py`, `hires_smoke.py` and `test_hires.py`.
The completed study's source snapshot is frozen: do not edit those files.
Every run has `summary.json`; queue phase is `complete`, verification/audit flags
are complete, and `REPORT.md` exists.

The dataset, code budget, resolution and coordinate budget differ from Z5.
Compare models **within Z6**; differences from Z5 are not a controlled estimate
of circuit improvement or resolution scaling alone.

## Earlier study: Z5, 4,744 images and 50,000 updates at 32×32

The user explicitly selected the larger dataset and longer training immediately.
**All 27 fresh runs and their independent quantum audits are complete.** Nine
architectures/controls, three seeds each, physical GPU0 only, up to six concurrent
jobs. The runner exited successfully and no Z5 job remains active. The completed
Z4 pilot below is preserved as a separate smaller-data study.

The main result is **stronger fitting with longer training, a narrower spatial
completion gap, and little quality gain from adding two qubits**. Explicit
spatial loading remains a circuit-cost tradeoff, not an overall quality winner.
This is a larger-dataset result at 32x32; higher resolution remains untested.

| Architecture | Qubits | Train PSNR | Train edge MSE | Completion PSNR | Completion edge MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Attention + shared chain | 5 | 25.994 | 0.0026905 | 21.436 | 0.0054629 |
| Attention + shared star | 5 | 25.969 | 0.0027168 | 21.518 | 0.0053311 |
| Spatial slots + shared star | 5 | 24.437 | 0.0035354 | 21.011 | 0.0056706 |
| Centre-only slots + shared star | 5 | 24.466 | 0.0035729 | 20.960 | 0.0057461 |
| Attention + shared chain | 7 | 26.017 | 0.0026962 | 21.457 | 0.0053722 |
| Attention + shared star | 7 | 25.941 | 0.0027031 | 21.520 | 0.0052938 |
| Spatial slots + shared star | 7 | 24.471 | 0.0035027 | 21.119 | 0.0055714 |
| Centre-only slots + shared star | 7 | 24.635 | 0.0034808 | 21.039 | 0.0057409 |
| Wider classical attention + shared chain | 5 | 26.022 | 0.0026591 | 21.514 | 0.0053445 |

Values are three-seed means; PSNR is mean per-image dB, higher is better; edge
MSE is lower-is-better. Completion freezes the field and fits fresh codes for
1,000 steps to 256 observed pixels in each of 32 excluded development images.
Scores use the 768 withheld pixels and edges whose endpoints are both withheld.
This is code inference, not an encoder or automatic reconstruction of new images.
Seed variability matters: spatial 5q completion is 21.011 +/- 0.338 dB sample SD,
versus attention-chain 5q 21.436 +/- 0.059. These SDs are not confidence intervals.

### What changed with data and training

On the SAME 4,744 targets, step10k to step50k fitting PSNR improves from
21.558 to 25.994 for attention-chain 5q, 21.173 to 24.437 for spatial 5q,
21.595 to 26.017 for attention-chain 7q, and 21.262 to 24.471 for spatial 7q.
All four still gain about 0.47–0.54 dB between 40k and 50k. Longer training
helps substantially; these curves do not establish convergence.

Compared with the Z4 pilot on the same completion images, spatial 5q improves
0.380 dB and spatial 7q 0.554 dB, with positive PSNR differences in every seed.
Their mean completion edge errors fall 4.32% and 6.11%. Attention-chain gains
0.122/0.225 dB at 5q/7q with mixed seed signs; its 5q completion edge error is
actually 1.43% higher. Better training fit is not uniform improvement on new images.

The common 1,024 training IDs are scored separately: attention-chain 5q improves
from 24.413 to 25.997 dB, with 25.993 dB on the additional 3,720 images. Spatial
5q improves from 22.656 to 24.439 on the common IDs, with 24.436 on the additional
images. This rules out attributing the full-dataset fitting gain solely to a
different mixture of training examples. Data and total updates change together
between studies: average image presentations are 168.63 versus 156.25, not five
times more presentations per image. The within-Z5 duration comparison holds data fixed.

### Representation, circuit and qubit comparisons

Spatial slots still lose completion PSNR to attention-chain in every paired seed:
0.425 dB at 5q and 0.339 dB at 7q on average, versus pilot gaps of 0.683/0.668 dB.
Their mean completion edge errors are 3.80%/3.71% higher (5q signs mixed; 7q all
worse). Fitting losses remain about 1.55 dB and 30% higher edge error. The
matched-star comparison isolates encoding more closely: spatial versus attention
with the SAME star head loses 0.507/0.402 dB and has 6.37%/5.24% higher completion
edge error at 5q/7q, with all seeds worse on both metrics.

Access to neighbours versus centre-only slots gives small mean completion gains:
0.051/0.080 dB and 1.31%/2.95% lower edge error at 5q/7q. Both metrics have mixed
seed signs at both counts. Do not claim a reliable neighbour-access improvement.

Seven versus five qubits gives attention-chain only +0.022 dB mean completion
PSNR (mixed signs), though its completion edge error falls 1.66% in all seeds.
Spatial gets +0.107 dB and 1.75% lower edge error, both with mixed signs. The
width-148 five-qubit attention control slightly exceeds seven-qubit attention-chain
on mean completion PSNR (+0.056 dB) and edge error (0.52% lower), also with mixed
signs. It has 56,336 classical parameters versus 56,220 for width-128 attention 7q.
This is not evidence of a distinctive benefit from the larger quantum register.

Five-qubit attention-star improves mean completion PSNR by 0.083 dB (mixed signs)
and edge error by 2.41% (all seeds) over attention-chain. Mean fitting is slightly
worse, and logical depth rises from 146 to 223. **Keep attention + shared chain
at five qubits as the main cost-conscious reference; keep attention-star 5q as
a completion-edge candidate.** Spatial slots remain an explicit-loading prototype.
This experiment does not establish quantum advantage or compare directly with a
fresh vanilla-QVF run under this exact larger-data protocol.

### Quantum execution and verification

| Main architecture | Qubits | RY/query | CNOT/query | Logical depth | Unoptimized line CNOTs | Completion PSNR, 1,024 shots |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Attention + chain | 5 | 176 | 110 | 146 | 242 | 20.740 |
| Spatial slots + star | 5 | 175 | 80 | 172 | 320 | 20.165 |
| Attention + chain | 7 | 342 | 246 | 337 | 930 | 20.781 |
| Spatial slots + star | 7 | 245 | 120 | 252 | 960 | 20.235 |

Counts include input preparation. Spatial loading saves 27.27%/51.22% logical
CNOTs at 5q/7q versus attention, but star connectivity incurs depth/routing costs;
its MLP is evaluated five times per query rather than once. Neither classical
FLOPs nor quantum execution budgets are matched. The seven-qubit spatial model
still has five data slots plus two work wires, not seven spatial inputs. All
models retain direct, continuous coordinate-to-quantum-RGB output and 336 fitted
classical code values/image. No encoder, diffusion or classical RGB bypass.

All 27 checkpoints passed explicit RY/CNOT reconstruction, joint finite-shot
sampling at 256/1,024/4,096 shots with five measurement seeds, analytical shot-MSE
and fourth-moment checks. Maximum explicit probability discrepancy was 2.47e-7;
normalized probability-to-saved-RGB drift was 1.71e-6. Ordinary checkpoint replay
was bit-exact (zero RGB error). Primitive parameter-shift checks against PennyLane
autograd passed, maximum absolute error 5.78e-15. Stored quantum entries are
45/55 at 5q/7q, with 45/53 output-relevant entries; conservative gate counts still
include two redundant terminal work-wire rotations at 7q.

Synthetic gate/readout noise was also tested on the first four completion images,
seed0, including state preparation. Larger circuits were more sensitive in this
small diagnostic; it is not calibrated hardware or a broad robustness benchmark.
Training used ideal simulation; the noise study evaluates frozen-code inference.
A 32x32 render at 1,024 shots still requires 1,048,576 circuit executions. These
results demonstrate a specified, simulatable quantum INR, not practical hardware
training, a speedup or higher-resolution scalability.

The frozen split has all 4,744 CIFAR-10 dog training images outside 256 fixed
validation IDs; the old 1,024 training IDs are its exact prefix. The official
test set was unused. Completion images, mask, initial codes and inference budget
match Z4. Initial decoder tensors match the same pilot architecture/seed exactly.
All 77 source snapshots/hashes, checkpoints, training streams and image visits
passed verification. Two regression tests and three discarded full-size GPU
preflights passed before launch. Old Z1–Z4 sources and results are unchanged.

Artifacts: [protocol](results/20261110_z5_full/protocol.json),
[complete marker](results/20261110_z5_full/PIPELINE_COMPLETE.json),
[all metrics and paired differences](results/20261110_z5_full/verified/summary.txt),
[pilot comparison data](results/20261110_z5_full/verified/pilot_comparison.json),
[training/pilot plot](results/20261110_z5_full/verified/pilot_comparison.png),
[quality/cost plot](results/20261110_z5_full/verified/decision.png),
[larger-label presentation](results/20261110_z5_full/verified/comparison_readable.png),
[training gallery](results/20261110_z5_full/verified/main_train_gallery.png),
[completion gallery](results/20261110_z5_full/verified/main_completion_gallery.png),
[checkpoint audit](results/20261110_z5_full/verified/verification.json),
[quantum audit](results/20261110_z5_feasibility/independent_verification.json).
Sources: `scale_study.py`, `scale_runner.py`, `scale_verify.py`, `scale_report.py`,
`scale_feasibility.py`, `scale_audit_verify.py`, `scale_decision.py`,
`scale_comparison.py`, `scale_preflight.py`, `tests/test_scale.py`.
All evaluation remains development-only; three seeds do not establish significance.
The presentation-only figure has a separate source/input manifest under
`verified/presentation_source/`; it does not modify the 77 frozen training/audit sources.

## Previous study: Z4 spatial encoding and 5/7 qubits

**All 27 runs and independent quantum audits are complete.** Nine conditions,
three seeds, 1,024 CIFAR-10 dog images, 10,000 updates each, 336 fitted values/image.
Training ran in three concurrent seed processes on GPU0. No Z4 process remains
running. Training source snapshots and all original Z1–Z3/A artifacts are preserved.

The new spatial representation assigns left, above, centre, below and right
latent features to five specific data wires. Each slot is sampled at three grid
scales and independently passed through a shared angle MLP. Five data uploads
are interleaved with a shared star-shaped quantum filter. At seven qubits the
same five slots feed five data wires, with two additional work wires initially
in |0>. All versions use the same first-five-wire RGB rule. There is no encoder,
diffusion model or classical RGB bypass. See [the method](SPATIAL_FILTER.md).

The main result is a **representation-cost tradeoff, not a quality improvement**.
Means across three seeds (PSNR is mean per-image dB):

| Architecture | Qubits | Train PSNR | Train edge MSE | Completion PSNR | Completion edge MSE | CNOT/query |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Attention + shared chain | 5 | 24.413 | 0.003592 | 21.313 | 0.005386 | 110 |
| Attention + shared star | 5 | 24.170 | 0.003816 | 21.486 | 0.005197 | 110 |
| Spatial slots + shared star | 5 | 22.656 | 0.004730 | 20.631 | 0.005926 | 80 |
| Centre-only slots + shared star | 5 | 22.982 | 0.004584 | 20.563 | 0.006061 | 80 |
| Attention + shared chain | 7 | 23.901 | 0.003738 | 21.232 | 0.005425 | 246 |
| Attention + shared star | 7 | 24.160 | 0.003722 | 21.255 | 0.005591 | 246 |
| Spatial slots + shared star | 7 | 22.602 | 0.004623 | 20.565 | 0.005934 | 120 |
| Centre-only slots + shared star | 7 | 23.020 | 0.004523 | 20.602 | 0.006022 | 120 |
| Wider classical attention + shared chain | 5 | 24.355 | 0.003645 | 21.458 | 0.005318 | 110 |

Completion freezes the field and fits fresh codes to 256 observed pixels of each
of 32 excluded development images; the table scores the 768 withheld pixels and
edges with both endpoints withheld. It is not automatic encoder generalization,
and these are repeatedly used development images, not a fresh final test.

Compared with attention + chain at the SAME qubit count, spatial slots lose
0.683 dB at five qubits and 0.668 dB at seven on completion, with 10.03% and 9.38%
higher completion edge error. Every paired seed loses on both metrics at both
counts. Fitting also deteriorates. Do not explain this as evidence against all
spatial quantum encodings: this is one particular architecture and a fixed budget.
Its learning curves do not establish convergence.

Access to neighbours does have a small effect within the slot family: versus
centre-only slots, five-qubit completion edge error drops 2.22% in every seed;
completion PSNR gains 0.068 dB on average with mixed signs. At seven qubits the
edge reduction is 1.46% with mixed signs, while mean completion PSNR is 0.037 dB
lower. The centre-only comparison holds the MLP and circuit exactly fixed.

Seven qubits do not give an overall quality benefit here. Attention + chain
changes completion PSNR by -0.081 dB (mixed signs); spatial slots change it by
-0.066 dB (all three seeds lower). The width-148 five-qubit attention control has
56,336 classical parameters, slightly more than seven-qubit width-128 attention's
56,220. It exceeds seven-qubit attention + chain completion PSNR by 0.225 dB in
all three seeds. Parameter-count matching does not eliminate every architectural
or optimization confound; it does rule out calling the larger register a clear
win at this training budget.

The five-qubit attention + star control improves mean completion PSNR by 0.172 dB
and edge error by 3.51% versus attention + chain, but both differences have mixed
seed signs and fitting is worse. It also has greater depth (223 versus 146 logical
gate layers). Keep it as a secondary completion candidate. **Retain the five-qubit
attention + shared chain as the main model for the next larger-data test.** The
spatial-slot branch needs a better quality/cost tradeoff before replacing it.

### Explicit quantum costs and audit

Counts include input preparation; these are logical all-to-all circuits, not
native-device timings. Fewer CNOTs do not always mean less depth or routing.

| Representation | Qubits | RY/query | CNOT/query | Depth | Unoptimized line CNOTs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Attention + chain | 5 | 176 | 110 | 146 | 242 |
| Spatial slots + star | 5 | 175 | 80 | 172 | 320 |
| Attention + chain | 7 | 342 | 246 | 337 | 930 |
| Spatial slots + star | 7 | 245 | 120 | 252 | 960 |

Spatial angle loading avoids arbitrary amplitude preparation. CNOT reductions
versus attention + chain are 27.27% at five qubits and 51.22% at seven. But the
five-qubit spatial circuit is deeper, and this particular unoptimized line route
uses more CNOTs for the spatial architecture at both counts. Routing counts are
constructive examples, not optimized bounds. Spatial models also evaluate their
MLP five times/query rather than once; classical FLOPs are not matched.

Finite-shot completion PSNR at 1,024 shots/query, averaged across three training
and five measurement seeds: attention-chain 5q **20.540**, spatial 5q **19.790**,
attention-chain 7q **20.449**, spatial 7q **19.760**. All 27 checkpoints passed
explicit-gate reconstruction, joint-shot sampling and analytical shot-MSE checks,
including exact fourth-moment sampling-error calculations. Trained-circuit
primitive shifts agree with independent PennyLane gradients. Noise diagnostics
include preparation, but use only the first four development images/seed0. Both
families deteriorate substantially under stronger noise; this is not calibrated
hardware robustness or hardware training. One full 32x32 image at 1,024 shots per
coordinate still costs 1,048,576 circuit executions.

Ten CPU tests and nine GPU preflights passed. All final training/completion
renders replay bit-exactly, NumPy metrics and sampling streams agree, and all
three five-qubit attention-chain runs reproduce Z3 weights/codes bit-for-bit.
The separate normalized-probability audit differs from saved float32 RGB by at
most 2.007e-6; explicit-gate probabilities differ by at most 2.857e-7. These are
separate numerical checks, not a discrepancy in the bit-exact checkpoint replay.

Metadata erratum: frozen training records call the stored quantum parameter
count `effective_quantum_parameters` (45/55). At seven qubits, two terminal RY
entries act only on unmeasured work wires and have no RGB effect. The verified
summary corrects the count to **45/53 output-relevant entries**, reports the two
redundant entries, and leaves executed gate counts conservative. The frozen
training sources/results are not rewritten.

Artifacts:

- [Frozen protocol](results/20261110_z4_spatial/protocol.json), [verification](results/20261110_z4_spatial/verified/verification.json), [all results and paired differences](results/20261110_z4_spatial/verified/summary.json).
- [Compact comparison](results/20261110_z4_spatial/verified/decision.png), [PDF](results/20261110_z4_spatial/verified/decision.pdf), [training gallery](results/20261110_z4_spatial/verified/main_train_gallery.png), [completion gallery](results/20261110_z4_spatial/verified/main_completion_gallery.png).
- [Quantum audit](results/20261110_z4_feasibility/summary.json), [independent audit verification](results/20261110_z4_feasibility/independent_verification.json), [shot/noise plots](results/20261110_z4_feasibility/feasibility.png).
- [Detailed method and limitations](SPATIAL_FILTER.md).

This is still a 32x32 development pilot. It is not a demonstrated larger-data or
resolution scaling result, a convergence comparison, or evidence of quantum
advantage. The controlled seven-qubit spatial version adds work wires; seven
independent spatial input slots have not been tested here.

## Completed initial study (Z1–Z3)

The user authorized the feasibility audit, small convolutions on latent grids,
and a result-guided next circuit/scaling step. **All 21 training runs and their
independent audits are complete:** nine latent-grid runs and twelve local-head
runs. Every condition has three seeds, 1,024 images, 10,000 updates, five qubits
and 336 fitted values/image. Completed original-head controls are referenced by
protocol and result hashes instead of retrained again. Cross-study initial
codes, common parameters and sampled data/query streams also match. No Z1–Z3 study
process remains running.

The initial feasibility audit is in `results/20261110_z1_feasibility_r2/`.
An earlier attempt used a different inference batch size and failed a numerical
replay tolerance; its partial output/source and FAILED marker are preserved in
`results/20261110_z1_feasibility/`. Matching the original batch size resolves the
check. No model weights were changed. The successful audit includes six V11
checkpoints, finite shots, explicit preparation, density-matrix noise, analytical
expected shot error checks and primitive parameter shifts on trained circuits.

`results/20261110_z2_latent_conv/verified/` contains all nine independently
verified fits and completion diagnostics. Across three seeds, latent 3x3
convolution changes training PSNR from 24.080 to 23.371 dB and training edge MSE
from 0.003730 to 0.004220. Completion PSNR is 21.142 versus 21.186 dB; completion
edge error is 0.005265 versus 0.005500, a 4.27% reduction with improvement in all
three seeds. The slightly larger pointwise control reaches 20.996 dB and 0.005657
completion edge error. Keep spatial refinement as a completion candidate, not an
overall fitting breakthrough. These are repeatedly used development images.

`fold_check.py` confirms exact absorption of the same-width refinement into the
336-value image codes. For freely fitted codes, this changes optimization and
the code prior/regularization, not the unrestricted representable image family.
The folded code penalty and an eventual encoder distribution are different.
The separate finite-shot/noise audit of all nine Z2 checkpoints is in
`results/20261110_z2_feasibility/`.

Two circuit seed processes failed during CUDA initialization before any run
directory was created; their logs are preserved and both seeds restarted. One
verifier launch in each study had the same transient failure, then succeeded. GPU 0 only;
the user's GPU 1 processes were not used or interrupted.

## Completed results and decision

Means across three seeds; these are fixed-budget development comparisons, not
convergence results or fresh final-test estimates. PSNR is mean per-image dB.

| Architecture | Training PSNR | Training edge MSE | Completion PSNR | Completion edge MSE |
| --- | ---: | ---: | ---: | ---: |
| Attention, original QVF head | 24.080 | 0.003730 | 21.186 | 0.005500 |
| Latent 3x3 convolution, original head | 23.371 | 0.004220 | 21.142 | 0.005265 |
| Pointwise capacity control, original head | 23.781 | 0.003840 | 20.996 | 0.005657 |
| Attention, shared local quantum head | 24.413 | 0.003592 | 21.313 | 0.005386 |
| Attention, untied local quantum head | 24.162 | 0.003707 | 21.196 | 0.005458 |
| Latent convolution, shared local head | 23.411 | 0.004170 | 20.950 | 0.005378 |
| Latent convolution, untied local head | 23.474 | 0.004129 | 21.057 | 0.005389 |

The shared local head without the added latent convolution is the next main
candidate. Relative to the same attention representation with QVF's original
head, it has 3.68% lower training edge error (all three seeds), +0.332 dB mean
training PSNR, +0.128 dB mean completion PSNR and 2.07% lower mean completion
edge error. PSNR and completion-edge differences have mixed signs across seeds;
do not describe them as uniform improvements. Sharing versus untied filters also
has mixed seed differences. The combined latent-convolution/local-head variants
do not improve overall quality over the local head alone.

| Explicit circuit resource per coordinate | Original QVF head | Local head, shared or untied |
| --- | ---: | ---: |
| RY primitives including preparation | 256 | 176 |
| CNOT primitives including preparation | 230 | 110 |
| Gate depth with all-to-all connectivity | 322 | 146 |
| Effective learned processing angles | 125 | 45 shared / 105 untied |

The local head removes 52.17% of CNOTs at fixed qubit count and the same amplitude
preparation. It still requires one circuit query per coordinate, with repeated
shots; this is not a full-image runtime or asymptotic scaling result. The line
routing construction is explicitly unoptimized and is not used to claim a
hardware speedup. The full model has 43,881 shared parameters for the shared local
head versus 43,986 for the original head (the latter includes 25 unused circuit
entries). Latent convolution adds 3,516 classical parameters; its pointwise
control adds 3,684.

All Z2/Z3 checkpoints also passed the finite-shot/explicit-gate audit and the
separate analytical shot-error/primitive-gradient checks. In the bounded noise
diagnostic (first four excluded images, seed0, 1,024 shots), the original head
goes from 19.140 dB without gate noise to 17.894 dB with p1=0.0001, p2=0.001 and
readout flip probability 0.005. The shared local head goes from 18.833 to 18.184
dB under the same setting. This supports further noise testing; it is a small
synthetic-noise result, not proof of device-level robustness. Both deteriorate
substantially at the stronger noise setting. Ideal simulator training remains
distinct from hardware training; the conservative primitive-gate shift recipe
requires 512 shifted evaluations/query for the original and 352 for the local
head, before forward evaluations, shots and classical work.

The next scale comparison should retain the original-head attention reference
and the shared local head, with the latent convolution as a secondary completion
option. Increase image count using the existing fixed validation split and
adjust training duration to account for image visits/convergence. Test genuine
higher-resolution sources separately. Include vanilla/global-code storage
controls before claiming representation gains over the full original QVF model.
No larger-data or higher-resolution training was launched in this initial Z
study. Images still lose detail. A true spatial quantum-filter encoding is a
separate next architecture factor, described below; it must not be conflated
with the current local-wire circuit.

Artifacts:

- [Combined summary and paired differences](results/20261110_z_summary/summary.json),
  [readable summary](results/20261110_z_summary/summary.txt),
  [quality/resource figure](results/20261110_z_summary/decision.png),
  [finite-shot/noise figure](results/20261110_z_summary/feasibility_comparison.png).
- [Training gallery](results/20261110_z_summary/training.png) and
  [completion gallery](results/20261110_z_summary/completion.png): first 12 IDs,
  seed0, no quality selection. Columns follow the table order, after originals
  (and observed-only images for completion). Clipping is display-only.
- [Z2 verification](results/20261110_z2_latent_conv/verified/verification.json),
  [folding check](results/20261110_z2_latent_conv/verified/folding_check.json),
  [Z3 verification](results/20261110_z3_local_quantum/verified/verification.json).
- [Z2 feasibility](results/20261110_z2_feasibility/summary.json),
  [Z3 feasibility](results/20261110_z3_feasibility/summary.json), and their
  `independent_verification.json`, protocols and source snapshots.

Entrypoints are `study.py protocol/run`, `verify.py`, `report.py`,
`feasibility.py`, `audit_report.py`, `consolidate.py`, `fold_check.py` and
`decision_report.py`, run as `python -m experiments.qvf_project_z.MODULE` in the
pinned `quantai` environment with `CUDA_VISIBLE_DEVICES=0`. Study protocols
preserve exact CLI-equivalent configs, source hashes and referenced controls.
Completed training runs are skipped; partial runs are never silently overwritten.

Six CPU tests passed: initial-function/capacity controls, spatial versus pointwise
scope, independent PennyLane circuit outputs and gradients, explicit state
preparation, Kraus-reference noise, primitive parameter shift, joint shot moments,
checkpoint reloads and hidden-target isolation. GPU preflights use synthetic data
only and no learned preflight weights enter the study.

`models.py` adds a zero-initialized residual block to each of the 2x2/4x4/8x8
four-channel code grids, BEFORE the existing attention retrieval. Each block
uses 4->16->4 shared 3x3 convolutions, zero padding and SiLU. A 4->136->4 pointwise
control has slightly more shared parameters but cannot mix neighbouring cells.
All arms keep 336 fitted values/image, five qubits and original grouped-Z RGB.
The original amplitude head and all original source files remain unchanged.

`study.py` reuses the V10 data, neighbour sampling, pixel + 0.5 edge objective and
optimization recipe. Default comparison: 1,024 training images, 10,000 updates,
three paired seeds. Completion fits fresh codes for 1,000 updates using 256
observed pixels of 32 excluded images, scoring withheld pixels and edges. This
is optimization-based inference, not an encoder. `verify.py` independently
checks raw metrics, replay, initialization, streams and resource counts.

## What convolution inside the quantum head means

The retained attention/amplitude reference uses QVF's original five repetitions
of all-directed-pair controlled-RY gates and per-wire RY gates, followed by the
original RGB measurement rule. Some V10/V11 alternatives changed input encoding
and inserted additional rotations; those are not identical to vanilla QVF.

The first proposed circuit change repeats a trainable two-qubit filter over
neighbouring wires in an even/odd arrangement, then applies the original per-wire
rotations. Compare shared filters with untied filters on the SAME graph and gate
schedule. Both start with identical functions. There is no pooling, reset or
intermediate measurement. This borrows locality/parameter sharing from
[quantum convolutional networks](https://arxiv.org/abs/1810.03787), but is not a
full QCNN architecture or a spatial image convolution: current amplitudes do not
assign neighbouring image pixels to neighbouring wires.

A genuine spatial quantum filter would apply the same small circuit to explicit
local neighbourhoods of the latent grids and return measured features at their
locations. One concrete direct-head candidate assigns five qubits to the centre,
left, right, above and below samples of a latent grid, uses repeated slot-wise
angle encoding for the local features/scales, then applies shared filters before
RGB measurement. Bilinear sampling of each offset keeps the query continuous.
This changes the encoding as well as the circuit; compare shared versus untied
or dense gates with the SAME slot encoding. It can still use one circuit per
queried coordinate. It is proposed, not implemented in the current local-wire test.

Alternatively it could preprocess a grid of quantum features before a separate
field; that adds circuit queries unless those measurements replace other calls
or are reused. It has precedent in
[quanvolution](https://arxiv.org/abs/1904.04767); our trainable continuous-field
application would require its own controls and novelty assessment.

## Feasibility scope

`feasibility.py` decomposes positive amplitude preparation into RY/CNOT gates and
decomposes every controlled rotation. It reports all-to-all gate/depth counts and
an explicit, unoptimized line-routing construction. Joint measurement samples
preserve correlations between RGB channels. Shot budgets are 256/1,024/4,096,
with five measurement seeds per checkpoint on all 32 completion images.

Gate noise uses a density matrix, including the preparation gates. Synthetic
uniform nonidentity Pauli errors follow each primitive one/two-qubit gate, with
separate classical readout bit flips. The bounded noise diagnostic uses seed0,
the first four excluded images and all their coordinates. Codes are frozen after
noiseless partial-image fitting. It tests inference robustness, not noisy code
adaptation or hardware training. Noise connectivity is all-to-all; line routing
is counted separately and not silently treated as noise-free physical routing.

Future scaling must measure image count and native spatial resolution separately,
including fitted-code storage, retrieval complexity, preparation, depth and total
measurements. Local wires and few qubits alone do not prove image-scale efficiency.
The detailed research sequence remains in the
[post-V11 design](../qvf_project_a/CONVOLUTIONAL_QVF.md#post-v11-quantum-feasibility-spatial-representation-and-scale).
