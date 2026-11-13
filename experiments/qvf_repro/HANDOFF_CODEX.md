# Handoff: QVF reproduction, next attempts

Written 2026-11-07 for Codex, by Claude. Everything below is reproducible from the files in this folder. Read this file and `results/summary.md` before touching anything.

## Latest extension: Z9 full per-scale attention combination

2026-11-13: the user asked to add the full combination after we clarified that
the previous three-branch models lacked attention within each scale. Add three
arms to the existing two-resolution, 50,000-update, three-seed comparison:

- `scale_attention_mean`: per-scale attention, three independent7q spatial
  circuits, mean RGB fusion.
- `scale_attention_fusion`: identical attention and circuits, with the existing
  16-parameter classical output-attention gate.
- `scale_attention_coherent`: identical per-scale attention and branch gates,
  with the existing9q coherent scale-routing architecture.

Attention is **classical local query/key softmax**, independently learned for
each scale. Each of the existing five spatial input slots reads at most5x5
neighboring grid tokens; out-of-bounds tokens are masked. A distance prior is
retained. This preserves the quantum spatial encoding and keeps attention work
bounded per coordinate. It is not the older5q reference's dense attention and
amplitude encoding. The new arms retain the same original7q branch gates and
RGB readout as their respective Z6/Z7 parents. Coherent routing is one9q
multiplexed circuit, not three entangled7q registers.

Root: `../qvf_project_z/results/20261113_z9_scale_attention/`.
Runner launched2026-11-13 at02:32UTC,PID2395320,phase`waiting_for_z8`.
It waits for the **whole Z8 pipeline**, then runs at most two workers on
physical GPU0. GPU1 remains reserved. Check `queue_status.json` and `launch.json`
for live status/PID; do not infer progress from this static handoff.
The extension has18 fresh runs and84 reused entries, hence102 entries across
17 methods and both resolutions. Both plots include vanilla QVF in **red**.
No new reconstruction results exist at setup; new rows stay pending.

All datasets, splits, masks, seeds, optimizer settings, code budgets and query
streams match Z8 within each panel. Training and checkpoint-loader function
bodies are unchanged from Z8; the new module imports its new registry and kind.
The independent verifier also checks each panel's parent protocol hash and
training/evaluation settings. Quantum audits use the unchanged independent
native-gate, finite-shot and noise implementations with the correct circuit
alias. Synthetic smoke/preflight runs are discarded, not quality evidence.

New files use the `combination_` prefix, plus `test_combination.py`. Numerical
tests cover independently implemented attention/gradients, per-scale isolation,
identical front-end initialization, uniform-fusion/routing-off equivalence,
cached execution, independent readout and native quantum states/gradients.
The18-run smoke checks training through full audits and plots. Detailed evidence
is saved in `launch_validation.json` before launching the real queue.
All three numerical tests, all18 smoke runs and all six full-batch GPU preflights
passed. Both smoke panels have nine successful metric checks and nine successful
quantum audits. Smoke and full manifests contain the same123 Python source hashes.
Keep all previously frozen sources unchanged. New snapshots also freeze the
combination implementation; do not change it after the study is created.

Compare the new mean arm with `separate_scales` to assess learned local attention,
including its added classical parameters. Compare fusion strategies within the
new family. These do not match quantum gate cost or register size, and the old5q
attention reference differs in both retrieval and encoding. No hardware or
quantum-advantage claim; earlier benchmark results were already inspected.

Refresh compact Git records using `combination_export --root Z9_ROOT`. It also
exports the final Z7 report and refreshes Z8 records. Datasets, checkpoints and
large arrays stay local. Existing authorization to commit/push research updates
continues to apply. Z7 finished all12 runs and checks on2026-11-12 at14:54UTC;
Z8 is active as of this extension's setup.

## Latest authorization and queue: Z8 complete-method measurements

2026-11-12: user asked for measured QVF and all current methods in both plots,
then explicitly confirmed50,000 updates at both32px/256px,all13 Z6/Z7 variants
plus vanilla QVF,and three seeds. They also explicitly authorized adding,
committing and pushing all pending research updates,including literature and
reference edits. Keep datasets,checkpoints and large generated artifacts local.

Z8 root:`../qvf_project_z/results/20261112_z8_complete_methods/`.
Detached runner PID2202707,launched2026-11-12 at05:50UTC. Phase at launch is
`waiting_for_z7`: it waits for the **whole** Z7 pipeline,including audits and
extended development fits,before starting two GPU0 workers. Do not interrupt
Z7 or useGPU1. Existing pinned environment remains unchanged.

Coverage is14 methods×3seeds×2datasets=84 entries. Reuse27 Z6 +12 Z7 high-res
results after verification; train45 fresh runs:42 CIFAR32 +3 vanilla256. Vanilla
high-res jobs are prioritized when the new queue starts. Both figures list every
method and mark unmeasured rows pending; historical V9 scores are not substitutes.
The new32px panel replaces the old2,000-update panel with a full50k comparison.

Data: same established4,744 CIFAR training/256 development IDs,plus64 deterministic
official-test dogs,excluding exact selected test/train-development duplicates.
Native32px; grids2/4/8×4channels=336 values,PE6,128 coordinate endpoints/update.
The256px protocol exactly matches Z6/Z7:grids16/32/64×4=21,504 values,PE9,2,048
endpoints/update. Both usebatch16,RGB+0.5edge,Adam0.001,1,000-step new-code fits,
25% completion masks,64 held-out evaluations and256 training-probe images.

Vanilla baseline directly calls the audited`QVFImage`,with original5q circuit,
amplitude map,RGB readout and64-value global codes. Only benchmark coordinates/
training settings adapt to the common protocol. Do not call it the paper's
training schedule or an equal-code-budget control. Both datasets differ; compare
models within a panel,not resolution changes across panels.

Validation complete before launch:3 numeric tests;45 disposable end-to-end runs,
all checkpoint/stream/metric and native quantum/finite-shot/noise checks; final
smoke reports. CPU/GPU preflight outputs:
`coverage_tests_20261112.log`,`coverage_preflight_20261112.json`,and
`20261112_z8_smoke/`. New source manifest has112 frozen Python files; original
61/77/88/102 source hashes remain unchanged. Do not edit any frozen source.

The new`coverage_*.py` modules implement data/model registry,training,verification,
quantum audit,queue,plots and compact Git exports. The queue updates plots as
runs finish and marks the report complete only after all measurements/audits.
Check`queue_status.json`,`runner.log` and per-run logs before answering status.
User preference: vanilla QVF must be **red** in both resolution plots. Added
`coverage_plot_style.py` as a separate presentation-only renderer; a CPU
`--watch` companion reapplies it after the frozen reporter writes its plots.
It records `plot_style.json` and logs to `plot_style.log` in the Z8 root, then
exits when the queue ends. Frozen sources and all metrics remain unchanged.
Run `coverage_plot_style --root Z8_ROOT` after any manual report refresh and
before exporting compact figures. Use the same red baseline in chat visuals.
Compact Git reports live in`../../reports/qvf_benchmarks/`; they are dated
snapshots,not a live queue. Refresh with`coverage_export --root Z8_ROOT` after
completion before committing later results. The exporter does not push.

## Latest active study: Z7 coherent scale-routing benchmark

2026-11-12 plot request: user asked to include baseline QVF. There is **no matched
vanilla-QVF result on the current 256×256 Stanford Dogs benchmark**. The updated
`../qvf_project_z/results/20261111_z7_coherent/STATUS_20261112_0411UTC_with_QVF.png`
marks that missing row and adds a separate historical V9 panel with vanilla64,
global336 and token attention, all5q and seeds0/1. That older protocol uses32px
CIFAR dogs,1,024 training images,2,000 updates and **500** completion-code updates
(read the frozen protocol; old prose incorrectly says1,000). Cross-panel PSNR
differences are not improvement estimates. No new baseline training was launched.
The JSON records source hashes; `plot_qvf_context.py` lives inside results so no
frozen training source was changed. A matched vanilla baseline remains missing.

Checked2026-11-12 at04:11UTC: **six complete,two active,four pending,zero failed**.
Runner1982870 alive; active local_capacity9 seed1 at48,500 and classical_attention16
seed1 at46,500 of50,000 updates. Seeds0/1 complete for coherent and real routing;
the other two controls have completed seed0. All final audits remain pending.
Matched first-two-seed completion means:coherent23.190,real-routing23.152,
old separate-scales22.994,three-mixed23.974,spatial23.821,attention5q24.282dB.
Coherent gains0.196dB over separate scales with positive differences in both
seeds,but still trails the stronger references. Phase effects mixed. Do not
compare these two-seed means against three-seed means or call them final results.
Independent saved-array metrics for six models×two seeds agree within1.07e-14;
all102 Z7 and88 Z6 source hashes unchanged. Evidence:
`../qvf_project_z/results/20261111_z7_coherent/STATUS_20261112_0411UTC.json`.
At current rates,roughly10–12hours of training remain,plus final verification,
noise audits and extended development fits; this estimate is not a guarantee.

Launched2026-11-11 at09:47UTC. The user's latest instruction authorized training
the coherent architecture and benchmarking it against previous results.
**Full Z7 training is running; only the provisional results above are available.**
Root:`../qvf_project_z/results/20261111_z7_coherent/`.
Runner PID at launch1982870,two concurrent workers,physicalGPU0 only. GPU1 is
being used by the user and must not be touched. Continue using the existing
`/home/agroz/anaconda3/envs/quantai/bin/python`,with`CUDA_VISIBLE_DEVICES=0` and
`OPENBLAS_NUM_THREADS=1`; do not upgrade dependencies.

Check`queue_status.json`,`runner.log`,per-run`progress.json` and`run.log` before
reporting LIVE progress. The detached runner automatically trains12 fresh runs
and then executes`routing_verify`,`routing_audit`,`routing_inversion`,and
`routing_report`. Full completion requires queue phase`complete`,all three
verification/audit/extended-development completion flags,and`REPORT.md`.
Do not mistake a launch or synthetic smoke results for a completed benchmark.

Matched to completed Z6:4,744 genuine256px photographs,50,000 updates,seeds0/1/2,
same21,504-value codes,batch16,2,048 coordinate endpoints/image/update,RGB+edge
loss,Adam and streams,evaluation masks,and1,000-step new-code fitting. All27 Z6
results are reused as verified references; no old weights initialize new models.
Four new variants:

- `coherent_routing`:9q multiplexed architecture with content-controlled routing
  and phases between the scale branches. No intermediate measurement.
- `real_routing`:same coherent routing with eight phase parameters zero/frozen.
  It remains quantum/coherent,not a classical control.
- `local_capacity9`:same9q and16 added trainable quantum entries,payload rotations
  without scale mixing. No non-additive cross-scale input interaction; gate count
  differs from the coherent model.
- `classical_attention16`:old independent3×7q heads with16 classical fusion
  parameters,using measured statistics,coordinate and scale to softmax-weight
  quantum RGB. No direct classical RGB decoder or feedback.

The old separate-scale mean is the no-routing reference:synthetic tests show
exact reduction. Other primary comparisons include three-mixed,spatial,joint
statistics,and attention5q. Counts are stored/trainable entries,not independent
degrees of freedom. Test scores from Z6 informed Z7:reusing this benchmark does
not supply a new confirmatory holdout. No novelty or quantum-advantage claim.

Added development-only code-fitting curves at1k/5k/10k updates for new coherent
routing and old separate-scales/attention5q,all three seeds,with frozen decoders
and continued optimizer state. Primary test budget remains1k. Finite-shot tests
retain Z6 masks/images/coordinates and equal per-circuit/equal-total budgets;
nonlinear classical softmax uses measurement replicates. Synthetic gate noise
uses4images×32 evenly spaced predeclared hidden query IDs,seed0,with the old
separate-scales and attention references rechecked on exactly those points.
Coherence removal is a tiny trained-checkpoint diagnostic,not retraining.

Validation before launch:3 new numeric tests,12 synthetic training runs,all
verification/quantum checks,9 extended development fits,and final smoke report
passed. Full-table preflight used at most7,724.3MiB allocated/job; two workers
fitGPU0. Artifacts:`routing_tests_20261111.log`,`routing_preflight_20261111.json`,
`20261111_z7_smoke/`,and full-root`launch_validation.json`.
All102 source hashes are frozen; existing88 Z6,77 Z5,61 Z4 and3 prototype hashes
were checked unchanged. New code is in`routing_*.py` and`test_routing.py`.
Never edit frozen Python sources during this run. Preserve logs/checkpoints
if a failure occurs; the fresh-only launcher refuses to overwrite started runs.
See the [Z README](../qvf_project_z/README.md) for details and artifact links.

## Architecture background: coherent quantum communication among scales

2026-11-11: user wants the three scale branches to exchange information in an
attention-like way, and clarified that meaningful interaction should occur
**inside the quantum computation**. The coherent prototype was implemented and
validated first; Z7 above now trains it and a classical measured-feature attention
control. Two-round feedback remains unimplemented. Do not describe the coherent
model as Transformer QK-softmax attention,a quantum advantage,or proven novelty.

New files: `../qvf_project_z/quantum_scale_attention.py`,
`test_quantum_scale_attention.py`, `coherent_scale_preflight.py`.
Artifact: `../qvf_project_z/results/coherent_scale_preflight_20261111.json`;
seven passing tests: `../qvf_project_z/results/coherent_scale_tests_20261111.log`.
See the [Z README](../qvf_project_z/README.md) for detailed design and controls.

Architecture: nine qubits = two scale-address qubits + seven shared payload
qubits. Three conditional branches retain the Z6 separate-scales seven-qubit
processing parameters and scale-masked classical inputs. Address00/01/10 select
coarse/medium/fine;11 remains empty. This is a multiplexed/direct-sum state,
not three separate7q registers or their full21q tensor-product state. Circuit
branches cannot run as independent parallel jobs. Scale phases and rotations
controlled by encoded centre/work qubits are interleaved between processing
stages. No intermediate measurement, new classical attention, or new classical
RGB network. Retrieval within each grid remains local interpolation. DINO is
still a later encoder possibility for Project A, not part of this prototype.

Routing-disabled output exactly reduces to the previous three-head RGB mean.
Relative phases affect outputs with active routing. Dephasing the scale address
changes RGB on the synthetic diagnostic; this tests use of coherence, not
improved quality or quantum advantage. Independent native PennyLane state and
gradient tests, explicit primitive compilation, finite shots, normalization,
field backpropagation and reload all passed. A final address-only mixer would
be invisible to the address-traced RGB readout; interleaving avoids that error.
Additional synthetic derivative diagnostic:
`../qvf_project_z/results/coherent_scale_interaction_20261111.json`.
Coarse-to-medium mixed derivative is nonzero with routing and zero without it;
the actual controlled mixer at the end alone changes RGB by only1.95e-16.

Resource construction includes preparation:1,000RY +8RZ +2X +994CNOT,
all-to-all primitive depth1,680, no routing/noise included. This is a validated
Gray-code multiplexer construction, not a minimal compilation. Added quantum
parameters16; total shared40,694, quantum181, no added classical parameters.
GPU0 synthetic preflight:4,744 codes, batch16×2,048coords,10 discardedAdamsteps:
coherent0.2033s/step,7,188.3MiB peak allocated versus independent3heads0.1116s,
3,683.7MiB. Excludes CUDA context/evaluation; no training-quality conclusion.
No packages upgraded; all88 Z6,77 Z5 and61 Z4 frozen source hashes unchanged.

## Completed study: Z6, four quantum designs at native 256×256

Checked2026-11-11 ~08:20UTC: **all27 runs and final verification/quantum audits/
reports passed; no Z6 job active**. Queue phase `complete`; full pipeline took
54,214seconds (~15.06hours). `REPORT.md`, `comparison.json`, native galleries,
`verification.json` and `quantum_audit.json` are in the study root below.
Max checkpoint replay error1.02e-6; independent metric error1.43e-14;
native state/gradient errors2.11e-15/4.67e-15. All88 source hashes remain frozen.

Final3-seed completion PSNR: spatial23.896, joint23.945, sequential23.488,
phase23.860, separate-scales23.257, marginal-products23.867, extra-real23.825,
three-mixed23.967, attention5q24.328dB. Joint-reference+0.049dB and joint-marginal
+0.078dB both3/3 seeds, but joint mean edgeMSE does not beat spatial. Separate
scales-reference−0.639dB and separate-three-mixed−0.709dB both3/3. Phase effects
small/mixed. Attention5q remains strongest; this comparison also changes input
retrieval/encoding/circuit family. These results do not prove a fundamental
generalization ceiling; all new-code fits use the same1,000-update budget.

Historical status checked 2026-11-11 ~04:00 UTC: **19 complete, five active, three queued,
zero failed**; original runner1751351 alive. All four main variants have completed
seeds0/1. Final verification/quantum-audit stages are still pending. Source88hashes
remain unchanged. Independently recalculated saved test-completion metrics for
six models × seeds0/1 agree within7.11e-15. Two-seed means: spatial23.821dB,
joint23.890, sequential23.420, phase23.741, separate-scales22.994, attention5q24.282.
Three separate heads fit the training probe well but perform worse on unseen
codes under the same1,000-update code-fitting budget; do not infer a fundamental
generalization limit. Full three-seed results/controls/audits remain required.
Timestamped evidence: `../qvf_project_z/results/20261110_z6_hires/STATUS_20261111_0400UTC.json`.
Initial queue estimate ~5.3more hours for training/evaluation, excluding final
audits; this is an estimate from observed rates, not a completion promise.

Launched 2026-11-10, status at 16:25 UTC: five GPU0 workers are training; no full
Z6 quality result yet. The user requested a FOUR-way study, with their three
independent seven-qubit circuits (one per scale) as the fourth condition, and
normal/higher-resolution data. The previous 4,744-image / 50,000-update choice
persists. Do not ask for that budget again or confuse the one-circuit sequential-
scale condition with the three-independent-circuit condition.

Root: `../qvf_project_z/results/20261110_z6_hires/`. Runner PID at launch 1751351,
five workers. Check `queue_status.json`, `runner.log`, per-run `progress.json` and
`run.log` for LIVE state before reporting results. The runner continues through
`hires_verify`, `hires_quantum_audit` and `hires_report` automatically. Entire
pipeline completion requires queue phase `complete`, both audit completion flags,
and `REPORT.md`. A launch or a few checkpoints is not a completed comparison.

Primary conditions: `joint_statistics`, `sequential_scales`, `complex_phases`,
`separate_scales`. Additional conditions: `spatial_reference`, `marginal_products`,
`extra_real_gates`, `three_mixed`, `attention_chain5`. Nine × seeds0/1/2 = 27 runs.
Use the [Z README](../qvf_project_z/README.md) for architecture/control details.
All four modifications are tested separately. Three-head fusion is fixed RGB
averaging with a shared classical angle MLP, separate quantum parameters, and
no entanglement between heads. Phase condition adds trainable RZ/IsingZZ; control
matches extra parameter and CNOT counts, not every one-qubit gate count. Joint ZZ
statistics come from the same bitstrings as Z means; this is not proof of
entanglement or a quantum advantage.

Data: genuinely 256×256 Stanford Dogs, original images with BOTH dimensions ≥256,
centre crop + Lanczos downsample, never upsampling. Training 4,744 + development256
from the pinned mirror's original train split; reserved test64 from its test split.
Mirror `maurice-fp/stanford-dogs`, revision
`ee102b9a35f00829e65bdad7cda8adcfa185b400`; original Stanford server unreachable.
Prepared arrays, image IDs/sizes/crops/JPEG hashes and array hashes:
`../qvf_project_z/data/stanford_dogs_maurice/prepared_256/manifest.json`.
Original MATLAB split lists not independently checked; near duplicates not ruled
out. Exact selected duplicate images were excluded. Existing base Python/PyArrow19
was used ONLY to extract raw JPEG bytes; all training uses the pinned quantai
environment. A failed earlier RMASD mirror/Parquet attempt is retained separately.
Do not install or upgrade dependencies.

Protocol: batch16, 2,048 coordinate endpoints/image/update (512 pairs/orientation),
50k updates, Adam0.001, RGB + 0.5 edge loss. Code norm coefficient is
1e-4/sqrt(21504/336). Grids16/32/64 × four channels = 21,504 classical values/image;
PE frequencies9; shared MLP width128/three hidden layers. Sparse bilinear gathering
avoids scanning whole grids per local query. 1.6384 billion coordinate queries/run;
three heads require three quantum evaluations/query. Do not call this quantum
image storage or compare raw Z5/Z6 scores as a controlled resolution ablation.

Reconstruction metrics: fixed 256 training-image probe, not all4744. Frozen-decoder
new-code fitting: first64 development + reserved64 test, 1,000 updates,128 sampled
pixels/image/update. Completion observes25% and scores only withheld pixels and
edges with both endpoints withheld. Separate fully observed test inversion.
No encoder, VAE or diffusion training. Quantum audits: native PennyLane state and
gradient checks; joint-bitstring finite shots (including exact nonlinear-product
MSE expectation), equal total shot controls; synthetic density noise only4images,
seed0,256 sampled coordinates/image. Static native-resolution galleries/crops.

Validation passed BEFORE launch: five independent numeric tests, all27 disposable
synthetic pipeline runs including all audits/report generation, full-table GPU
preflight at2,048queries (2.0–3.6GiB allocated/job). These are not quality evidence.
Z4's61 and Z5's77 frozen source hashes were checked unchanged. All new Z6 sources
are now frozen in the full study snapshot. Do not edit them during training.
New run creates fresh checkpoints; the launcher refuses to overwrite started
runs. If anything fails, inspect logs and preserve provenance before recovery.

Historical feasibility question (answered before Z6 launch):

Feasibility check completed only (not a new quality study): three existing real
spatial heads, receiving separate 2x2/4x4/8x8 code slices by masking other slices,
were trained jointly on synthetic data with fixed-average RGB fusion. All heads
received finite, nonzero quantum gradients. Batch16 x 128 coordinates, 18 Adam
steps with the last12 timed: three-head median step 0.07766 seconds and peak
allocated GPU memory 216.57 MiB, excluding the CUDA context. The one-head check
was 0.02539 seconds / 117.47 MiB. These are short preflight measurements, not full
training timing estimates. Weights were discarded. Result:
`../qvf_project_z/results/scale_heads_feasibility_20261110T153429Z.json`.

Three independent heads can be simulated separately and executed sequentially
using the same seven physical qubits. Their measured outputs are combined
classically; there is no entanglement between scales. A coherent interaction
between scales is a different architecture. The proposed scale condition should
also have a three-head/all-scales control so extra capacity and circuit work are
not mistaken for a benefit from dedicating one head to each scale. The other two
changes should initially be separate conditions. This short check itself did not
implement or launch a quality study; the subsequent Z6 implementation and launch
are described above.

Latest completed study: **Z5 full-data/long-training comparison COMPLETE (2026-11-10)**.
The user explicitly selected 4,744 images and about 50k updates. All nine Z4
architectures/controls x three seeds were trained from scratch for 50,000 updates
on GPU0 only, with up to six concurrent jobs. All 27 runs and the entire audit/
report pipeline passed; no Z5 process remains running. Preserve the completed
Z4 pilot as a separate experiment, not as the result of this larger study.

Study: `../qvf_project_z/results/20261110_z5_full/`; audit:
`../qvf_project_z/results/20261110_z5_feasibility/`. `PIPELINE_COMPLETE.json`
records successful completion. Read the [Z README](../qvf_project_z/README.md)
for all nine models, paired contrasts, galleries, resource counts and limitations.

Three-seed means, attention-chain versus spatial-star:

- 5q: train PSNR 25.994 versus 24.437; completion PSNR 21.436 versus 21.011;
  completion edge MSE 0.0054629 versus 0.0056706.
- 7q: train PSNR 26.017 versus 24.471; completion PSNR 21.457 versus 21.119;
  completion edge MSE 0.0053722 versus 0.0055714.

Longer training matters: on the SAME 4,744 images, 10k to 50k raises attention-
chain fitting by 4.436/4.422 dB at 5q/7q and spatial fitting by 3.264/3.208 dB.
All four still gain roughly 0.5 dB from 40k to 50k; convergence is not established.
Spatial completion improves over the pilot by 0.380/0.554 dB in all paired seeds,
but remains 0.425/0.339 dB below attention-chain in all seeds. Mean completion edge
error is now 3.80%/3.71% worse, not the pilot's approximately 10%. Matched-star
controls also favour attention on both completion metrics in every seed.

Neighbour access versus centre-only slots gives only small mean completion gains,
with mixed signs across seeds at both qubit counts. Seven-qubit attention-chain
adds only 0.022 dB mean completion PSNR (mixed signs), while reducing completion
edge error 1.66% in all seeds. Spatial 7q adds 0.107 dB and reduces edge error
1.75%, both with mixed signs. The wider classical 5q control performs similarly
to 7q attention-chain. Do not claim extra qubits have no effect, or a clear gain.

Keep **attention + shared chain, five qubits** as the main cost-conscious reference.
Attention-star 5q is a secondary completion-edge candidate: +0.083 dB mean PSNR
(mixed signs), 2.41% lower edge error (all seeds), but logical depth 223 versus 146.
Spatial is an explicit-loading prototype, not a quality winner. Logical CNOT/query
counts including preparation remain 80/120 for spatial 5q/7q versus 110/246 for
attention. Spatial requires more routing in the declared unoptimized line mapping
and five classical MLP evaluations/query versus one. No hardware speedup is shown.

All 27 checkpoints replay bit-exactly. Source/checkpoint hashes, fresh decoder
initialization against the pilot, training streams/visits, NumPy metrics, and saved
2k/10k checkpoints passed. All 77 source snapshots are frozen. Two regression tests
and three discarded full-size GPU preflights passed; older sources are untouched.
Finite shots, explicit preparation, synthetic gate noise, analytical joint-shot
moments and primitive parameter shifts all passed independent checks. Maximum
normalized RGB drift was 1.71e-6; gate-probability error 2.47e-7; gradient error
5.78e-15. At 1,024 shots, completion PSNR is attention-chain 20.740/20.781 versus
spatial 20.165/20.235 at 5q/7q. Gate-noise diagnostics use only four images/seed0.

Same 256 validation IDs, 32 completion images/mask/inference budget and original
1,024-image training prefix as Z4. Same circuits, 336 classical code values/image,
optimizer/loss and queries per update. Completion fits fresh codes for 1,000 steps
from 256 observed pixels, then scores the 768 withheld pixels and withheld edges.
No encoder or diffusion is trained here. More data and total updates change together
between studies: 168.63 versus 156.25 average image presentations, not 5x per image.
Within-Z5 duration comparisons hold data fixed; the common 1,024 training images
are also scored separately. Official test set unused; all results remain at 32x32,
on reused development images, with ideal-simulation training. A full 1,024-shot
render needs 1,048,576 circuit executions. Higher-resolution and hardware training
remain untested; no quantum advantage or new matched vanilla-QVF comparison claimed.

Previous update (historical pilot): **Z4 COMPLETE (2026-11-10)**. The user authorized a comparison
of the shared local head with explicit neighbour-to-qubit spatial encoding,
at five AND seven qubits. All 27 runs (nine conditions x three seeds) finished
and passed independent source/checkpoint/stream, exact replay and NumPy metric
checks. Ten CPU tests and nine GPU preflights passed. All 27 checkpoints also
passed explicit preparation, finite-shot/noise and independent gradient/moment
audits. GPU0 only; no Z4 process remains running.

Results and frozen sources: `../qvf_project_z/results/20261110_z4_spatial/`;
quantum audit: `20261110_z4_feasibility/`. Read the Z README and
`SPATIAL_FILTER.md` for the method, controls, plots and precise limitations.
The study used 1,024 images/10k updates, 336 classical code values, and the SAME
five-wire RGB readout at both sizes. Spatial slots map left/above/centre/below/right
latent neighbours to five wires. Seven qubits add two work wires, not additional
spatial input slots. No encoder, diffusion or classical RGB bypass.

Main result: spatial slots work but do not improve reconstruction. At 5q,
attention-chain versus spatial-star completion is 21.313 versus 20.631 dB;
at 7q it is 21.232 versus 20.565. Spatial completion edge error is 10.03%/9.38%
higher, with losses in all paired seeds at both counts. Spatial input avoids
amplitude preparation: CNOTs/query 80/120 versus attention's 110/246. But the 5q
spatial circuit is deeper (172 vs146), and its star connectivity has routing
costs. Do not equate fewer logical CNOTs with a hardware speedup.

Neighbour access improves 5q slot completion edge error by 2.22% in all seeds
versus the matched centre-only control. That small gain does not close the gap
to attention. Extra qubits do not provide an overall quality gain; the wider
classical 5q attention control beats 7q attention-chain completion PSNR by
0.225 dB in every seed. Attention-star 5q is a secondary completion candidate
(+0.172 dB and 3.51% lower edge error on average, mixed signs, worse fit and
more depth). Keep **attention + shared chain, five qubits** as the main model
for the next larger-data test. No larger-data/high-resolution result is claimed.
All three new 5q attention-chain runs reproduce Z3 weights/codes bit-for-bit.

Finite-shot evaluation retains the main quality ranking. Gate-noise diagnostic
is only four reused development images/seed0; no hardware-training or broad
robustness claim. All current results are fixed-budget development comparisons,
not fresh final-test or convergence results. A 1,024-shot full 32x32 render still
requires 1,048,576 circuit executions.

Metadata label erratum: frozen Z4 training `effective_quantum_parameters`
counts stored entries (45/55). Two final work-wire RYs at 7q cannot affect the
five-wire RGB output, so the verified summary reports 45/53 output-relevant
entries and two redundant entries. Gate counts include their actual execution.
Sources/checkpoints are preserved; do not silently rewrite this historical field.

Project naming and authorization, 2026-11-10: the user names this pre-A quantum
INR work **Project Z** and authorizes the feasibility audit, small convolutions
on latent grids, and a result-guided circuit/scaling follow-up. They also ask
about convolution within the quantum head. New code lives in
[`../qvf_project_z/`](../qvf_project_z/README.md); historical A/V9–V11 artifacts
remain in place. The historical direct attention/amplitude reference retains QVF's
original processing/readout; shared local filters have now been tested.
Do not equate locality among amplitude-encoded qubits with locality among image
pixels. Execution status and results are in the Z README.

Initial Z study is now COMPLETE: 21 fresh 10k-update runs, three seeds per
condition, 1,024 images/five qubits/336 code values each. Nine latent-refinement
and twelve local-circuit runs, all independently verified. Explicit preparation,
finite-shot readout and synthetic density-matrix noise audits also completed.
Results: `../qvf_project_z/results/20261110_z_summary/`; protocols are in
`20261110_z2_latent_conv/` and `20261110_z3_local_quantum/`. No Z1–Z3 process remains
running. Original/A/V9–V11 training sources and checkpoints remain unchanged.

The Z1–Z3 main candidate was attention with a shared local quantum head: five sweeps
of shared two-qubit filters on neighbouring wires, no pooling/reset, same
amplitude input and grouped-Z RGB. Compared with the original head on the SAME
attention codes: 230 -> 110 CNOTs including preparation; train PSNR 24.080 ->
24.413 dB and edge error 0.003730 -> 0.003592 (3.68% lower, all three seeds).
Completion PSNR 21.186 -> 21.313 and edge 0.005500 -> 0.005386 (2.07% lower mean,
mixed seed differences). This is a circuit quality/resource result, not quantum
advantage or an image-space quantum convolution. Untied same-graph control and
combinations with latent convolution are also trained. No overall quality gain
from combining the two changes was established.

Latent 3x3 convolution alone improves completion edge error by 4.27% in all three
seeds but worsens training fits; keep it as a secondary completion candidate.
Exact folding checks show the same-width refiner can be absorbed into fitted
codes, so it changes optimization/regularization rather than unconstrained image
function capacity. Noise diagnostics include preparation but use only four
development images/seed0; finite-shot-only evaluation uses all32/three seeds.
No larger-data/resolution, encoder, diffusion or hardware-training result yet.
Z4 has now tested explicit spatial-slot encoding (see latest update above).
Next: scale the five-qubit shared local head with matched original-head and
storage controls. Preserve frozen study sources; extend in new files.

Current user research objective, 2026-11-10: build a useful QUANTUM INR, with
credible quantum feasibility (simulation is acceptable), a more suitable spatial
representation than vanilla QVF, and demonstrated or explicitly justified paths
to larger datasets and genuine higher-resolution images. Beating classical
models is not the project objective. Classical controls still identify what the
quantum and classical components contribute. Prioritize coordinate-to-RGB
prediction, explicit state-preparation/gate/measurement costs, finite-shot/noise
evaluation, and scaling at a fixed small quantum register. Small qubit count or
fast statevector training alone is not evidence of practical execution cost.

V11's fixed-raster quantum-grid/conv output remains a secondary branch; it does
not by itself meet the continuous-QINR objective. The proposed next sequence is
an execution-cost audit of existing fields, a controlled small convolution on
latent grids BEFORE attention and the quantum field, then circuit changes and
larger-scale evaluation. Preserve direct quantum RGB readout as a reference. If
per-coordinate shot cost dominates, a separately labelled continuous field with
cached quantum features/coefficients is a candidate for reducing measurement
work; it is unimplemented and requires accuracy/storage/cost evaluation.
An eventual bounded local lookup must replace all-token attention if grid sizes
grow; track fitted classical code storage explicitly. Project A's encoder comes
after validating this decoder; diffusion remains later. This proposal was
subsequently authorized and partially evaluated in the completed Z study above. See the
[post-V11 plan](../qvf_project_a/CONVOLUTIONAL_QVF.md#post-v11-quantum-feasibility-spatial-representation-and-scale).

Latest completed study, 2026-11-10 (started 2026-11-09): user authorized
direction 1 (local patches) and 2
(coordinate-aware angle loading) separately/in parallel, then together, then
direction 3 (quantum feature grid + small convolutional decoder). All 27 V11 runs
completed on GPU 0 in `../qvf_project_a/results/20261109_v11_interfaces/` under its
immutable protocol. Nine conditions x three seeds, 1,024 images and 10k updates
each. Baseline attention is retrained from scratch. Same 336 fitted values/image,
five qubits for quantum arms, V10 neighbour sampler and edge weight 0.5.

Stages: `separate` (patch_amp, attention_angle, attention_amp,
attention_amp_wide), then `combined` (patch_angle, patch_angle_identity,
patch_classical), then `grid` (quantum_grid_conv, classical_grid_conv).
`interface_study.py run --root ... --stage STAGE --seed SEED` enforces that all
earlier-stage runs finished. All 12 separate-stage and nine combined-stage runs
finished, followed by all six grid-stage runs. No training remains running.
Durable logs are `STAGE_seedN.log` in the root.
Completed runs are skipped; partial directories are preserved, not auto-resumed.

Implementation: `interface_qvf.py`, `interface_study.py`. Five tests passed;
independent PennyLane outputs and gradients match the new five-upload circuit.
Nine GPU preflights passed exact train/completion reloads. Preflight metadata is
`../qvf_project_a/results/20261109_v11_preflight.json`; no preflight learned weights
are reused. Audit/report entrypoints: `verify_interface_study.py` and
`interface_report.py`. Do not modify any snapshotted training source while runs
are active; preserve the completed source snapshots. Original QVF/V9/V10 sources
and checkpoints remain unchanged.

The patch input unfolds replicated-border 3x3 neighbourhoods before bilinear
sampling: 108 local features plus 26 coordinate features. Angle loading starts
from |00000>, adds 25 input RY rotations over five stages, and retains 125
effective trainable processing angles; input angles include explicit learned
coordinate slopes. No initial-function equivalence to amplitude loading is
claimed. Processing-identity control retains input rotations but removes all
shared processing; it does not isolate entanglement alone. Classical and wider
attention controls have slightly more parameters than their quantum/patch targets.

Grid output queries 64 feature states/image, takes 15 Z/ZZ statistics and upsamples
to 32x32 RGB with a small conv decoder. No feature/coordinate skip bypasses the
quantum/classical feature module. Its matched classical control uses the same
conv architecture/initialization. This changes to native raster output. RGB-target
budgets match but quantum-query/gate/FLOP costs differ; do not equate query counts
with hardware speedups. Completion uses the same V10 32 images/256 observed
pixels/1000 code steps and scores unobserved-unobserved edges. No encoder,
diffusion, official test-set use or hardware/shots experiment is included.
All 27 runs passed independent verification: exact final training/completion
reloads, NumPy float64 metrics, source/checkpoint hashes, paired initialization,
query streams/visits, quantum query counts and step-2,000 metrics. All three
attention references reproduced V10's final weights AND codes bit-for-bit.
Report means, sample SDs and paired differences were separately checked. Results,
summaries, static plots and fixed galleries are in `verified/`; see the
[V11 README section](../qvf_project_a/README.md#interface-architecture-study-v11).

Main V11 results, means across three seeds (training PSNR / edge MSE; completion
PSNR / withheld-edge MSE):

- Attention: 24.08 / 0.003730; 21.19 / 0.005500.
- Wider attention: 24.19 / 0.003692; 21.38 / 0.005321.
- 1, patch input: 23.68 / 0.004037; 21.04 / 0.005763.
- 2, angle loading: 23.44 / 0.004103; 20.90 / 0.005906.
- 1+2: 23.12 / 0.004371; 20.84 / 0.006015.
- 1+2 processing removed: 21.90 / 0.005679; 20.17 / 0.007436.
- Classical patch field: 23.71 / 0.004065; 21.22 / 0.005570.
- Quantum grid + conv: 22.47 / 0.004543; 21.18 / 0.005343.
- Classical grid + conv: 22.93 / 0.004336; 20.61 / 0.005764.

Directions 1/2/combined worsen mean training edge error by 8.24%/10.02%/17.19%
relative to attention; completion also worsens. These are results for the tested
recipe, not limits on all patch/angle architectures. Patch input replaces
adaptive attention with bilinear neighbourhood sampling as well as increasing
input width. Learned processing improves the combined model over its retrained
processing-removal control (23.04% lower training edge error, +1.23 dB), but the
matched classical patch field still beats it. Do not infer entanglement-specific
effects or a quantum advantage.

Direction 3 has weaker training fits but approximately the original attention
completion PSNR (21.18 versus 21.19 dB), with 2.85% lower mean completion edge
error and mixed per-seed edge differences. Wider attention has better mean quality
metrics overall. Quantum grid beats its matched classical grid on completion:
+0.56 dB and 7.30% lower edge error, both improvements in all three seeds; it has
worse mean training scores. Treat this as a small paired completion result,
not a generalization/quantum-superiority proof. Completion still optimizes new
codes from partial observations, and images remain soft.

The concrete grid benefit is query count: 64 versus 1,024 circuits per full image
(16-fold reduction); training uses 64 versus 128 per image/update (twofold).
It adds a classical conv head and raises shared parameters from 43,986 to 90,944.
No finite-shot or hardware speedup was measured. Keep attention as the quality
reference and the quantum feature grid as a candidate for lower quantum-query
cost and completion. No substantial overall quality breakthrough was established.

Latest architecture discussion, 2026-11-09: user asks for substantial improvements
to the classical/quantum model, including quantum inputs and outputs. The
[post-V10 design](../qvf_project_a/CONVOLUTIONAL_QVF.md#post-v10-redesigning-local-conditioning-and-the-quantum-interface)
recommends local patch conditioning and explicit coordinate-dependent repeated
angle encoding as separate factors. It also records small shared convolutions,
signed/complex encoding ablations, learned Z/ZZ readouts, and a more radical
quantum feature grid followed by a small convolutional decoder. The last changes
the QINR task if its output is fixed-resolution. These are proposals, not trained
results at that point; the user subsequently authorized V11 above.

One bounded diagnostic was completed in
`../qvf_project_a/results/20261109_readout_probe/` using `readout_probe.py`.
Freeze the three V10 attention/edge circuits AND fitted codes, then fit a shared
affine RGB map by exact full-training-set least squares with pixel + 0.5 edge
loss. Five-Z affine control has 18 parameters; five-Z + ten-ZZ has 48. Means:
original 24.0804 dB / 0.00372956 edge MSE; five-Z 24.1986 / 0.00373248;
Z+ZZ 24.2054 / 0.00373202. Readout-only improvement is small and does not improve
edges, so prioritize input/local structure over more measured features alone.
This does not rule out joint retraining or other readouts. There is NO held-out
or encoder generalization test in this probe. Two independent measurement/objective
tests pass; original rendering agrees within 5.97e-8 RGB and baseline metrics
match verified V10 scores. Saved checkpoints were unchanged. Protocol, source
snapshot, coefficients, per-seed metrics, verification and static plot are saved.

Previous completed study, 2026-11-09: user authorized training the edge-fidelity
follow-up. All 12 V10 runs completed on GPU 0 under
`../qvf_project_a/results/20261109_v10_edges/protocol.json`. Four conditions:
attention/staged_all, each with edge weight 0/0.5. Three seeds, 1,024 images,
10,000 updates, five qubits and 336 code entries/image. Revised stages retain
all scales in initial amplitudes, then add medium/fine RY uploads initialized
at zero. Attention and revised stages start with equivalent predictions and
identical common weights; staged adds 50 classical parameters and ten data gates.

Code is isolated in `edge_qvf.py` and `edge_study.py`; original QVF and V9 source
files are unchanged. All conditions query 32 horizontal and 32 vertical pairs
(128 endpoints/image). RGB inverse-degree weighting corrects endpoint bias.
Edge loss matches the signed finite-difference evaluation metric. Same paired
data streams, constant Adam lr 0.001 and final-checkpoint selection. Step 2,000 is
saved for within-protocol duration checks; V9 had a different sampler.

Completion uses the same pixel-only inference objective for every condition:
1,000 new-code updates on 256 observed pixels of 32 excluded images, then score
768 withheld pixels and edges with BOTH endpoints withheld. Seven targeted tests
and four GPU preflights passed. All 12 runs passed independent audits: exact final
training/completion reloads, NumPy float64 metrics, source/checkpoint hashes,
initialization, query streams and step-2,000 checkpoint metrics. No training
remains running. Verification/report entrypoints: `verify_edge_study.py`,
`edge_report.py`. Preserve protocol/source snapshots. Results, per-seed summaries,
static comparison/learning plots and fixed galleries are in `verified/` under
the V10 root. The [Project A README](../qvf_project_a/README.md#edge-fidelity-study-v10)
contains the full interpretation and commands.
No encoder, diffusion or finite-shot/hardware experiment was added.

Main V10 result: adding edge loss improves training edge error by 10.68% for
attention and 9.55% for revised stages, while also improving training PSNR. These
within-architecture improvements occur in all three seeds. Training edge MSE
means, in attention/pixel, attention/edge, stages/pixel, stages/edge order:
0.004176, 0.003730, 0.004124, 0.003730. Training PSNR: 23.64, 24.08, 23.93,
24.32 dB. Revised stages and attention with the same edge loss have essentially
identical mean edge error, with mixed per-seed differences; do not claim an extra
circuit edge-fidelity win. The revised model has 50 more shared parameters and
ten more data-dependent gates, with the same five qubits/125 effective trainable
processing angles and 336 code entries per image.

Completion edge MSE means in the same order: 0.005531, 0.005500, 0.005369,
0.005402; PSNR: 21.11, 21.19, 21.41, 21.44 dB. Revised stages with edge loss
have 1.78% lower mean completion edge error and 0.25 dB higher PSNR than attention
with edge loss, but the edge difference changes sign across seeds. Adding edge
loss to revised stages slightly worsens mean completion edge error (0.61%).
This is sparse completion with new-code optimization, not encoder generalization.
Images remain soft. Keep the edge objective as a useful fitting option and the
equally trained attention model as baseline; the extra circuit edge benefit is
unproven. Attention/pixel improves its edge error by 34.70% from 2k to 10k in
V10, so duration itself explains substantial progress. V9 has a different sampler
and completion budget, and its coarse-only staged variant was not rerun in V10;
do not infer a matched comparison or identify a cause of V9's tradeoff from it.
No full-data/50k run of these token architectures has been launched.

Previous completed study, 2026-11-09: the user authorized token conditioning and a
modest qubit increase, with same-qubit QVF controls. V9 is implemented and complete
on GPU 0 under `../qvf_project_a/results/20261109_v9_tokens_pilot/protocol.json`.
Twenty runs: two seeds, five/seven qubits, and vanilla64/global336/bilinear/
attention/attention_staged variants. Each fits 1,024 images for 2,000 updates.
Spatial models fit 336 values/image (2x2,4x4,8x8 grids, four channels); global336
controls the larger per-image storage. Each final decoder is frozen for fresh
code fitting on 32 excluded images, using 256 observed pixels and scoring the
remaining 768. This is a bounded pilot, not the previous 50k/full-data protocol.

Code: `token_qvf.py`, `token_study.py`, `verify_token_study.py` in Project A.
Nine targeted tests and all ten GPU model/reload preflights passed. Two CUDA
initialization failures occurred before any training; their logs are preserved
under the pilot's `launch_failures/`. Both restarted seeds completed; no training
remains running. All 20 models passed independent reload/metric/stream audits,
with zero RGB discrepancy when matching the trainable/frozen execution mode.
Results and static figures are in the pilot's `verified/` folder. Preserve the
immutable protocol/source snapshot and existing completed runs. Do not modify
training source during the study. Completion markers and histories are per run;
the verifier independently checks raw metrics, reloads, budgets and paired RNGs.
No encoder/diffusion or finite-shot/hardware experiment is included. Original
QVF source is unchanged. Further details and commands are in the Project A README.

Main V9 result: the architectural gain exists at fixed qubits and code storage.
At five qubits, training PSNR means are 17.28 vanilla64, 18.30 global336, 20.04
bilinear, 20.21 attention and 20.76 attention+stages. Corresponding completion
means are 17.92, 19.30, 19.46, 19.76 and 19.96 dB. Seven-qubit staged tokens reach
20.85 training and 20.36 completion, but the latter varies substantially across
two seeds (sample SD0.42). Other seven-qubit variants have lower means than their
five-qubit counterparts at this budget. Extra qubits are not an automatic win.
Most fitting improvement comes from local representation; attention and staging
give smaller additional gains. Images remain soft, and staged encoding does not
improve gradient error over attention alone. Do not overstate sharpness, large-data
generalization, convergence, quantum advantage, or scale-specific gate effects.
This is a pilot; longer full-data and additional circuit/classical controls are
not run yet. No longer full-data run has been launched for this architecture.

Post-V9 question: how to improve edge error beyond attention alone. The design
document now proposes (1) retaining all three scale summaries in initial amplitude
preparation while adding later scale-dependent rotations, and (2) a supervised
adjacent-pixel-difference loss. V9's staged model used only coarse initial features;
finer information entered exclusively through small angle maps. This is a candidate
bottleneck, not a diagnosed cause. V9 did not optimize edge loss directly. Compare
both models with/without the edge term at matching qubits, code storage, query and
training budgets; include longer training as a control. The user subsequently
authorized V10 above. The earlier additional edge plot uses V9 histories only.

Current user direction, after V8: improve QVF itself using local features at
several spatial scales before returning to Project A's encoder and diffusion.
The user clarified that the recent questions about INRs, VAEs, RAEs and quantum
latent models were philosophical; they still prefer the multiscale-local-QVF
direction. Do not interpret that discussion as a switch to Project B or another
large pretrained RAE pipeline.

The design is recorded in
`../qvf_project_a/CONVOLUTIONAL_QVF.md`, under "Next study: directly learned local
codes at several scales". It proposes fitting a compact pyramid of per-image
spatial codes, querying every scale at each coordinate, and feeding their local
features to a small amplitude MLP and the original five-qubit/depth-five QVF
circuit as a reference. The user additionally wants to investigate scale-specific
quantum components. The same design document proposes successive coarse,
intermediate and fine stages on one five-qubit register: coarse amplitude
preparation, then feature-dependent rotations between trainable circuit blocks,
with no intermediate reset or measurement and the original grouped RGB readout
at the end. The unchanged circuit is a control, not a requirement for all variants.
Separate measured circuits per scale are another possible control and do not
provide cross-branch entanglement. Direct code fitting isolates the decoder before
adding an image encoder or diffusion. Exact grid sizes, code budgets and training
settings were subsequently fixed for the V9 pilot described above; alternative
designs remain proposals. Do not describe an unfinished run as a measured result.

The user also wants to consider tokenization. The design document's "Token
extension: coordinate queries select local information" proposes continuous
positioned tokens at several scales, with small classical coordinate-to-token
attention replacing bilinear lookup. Grid cells already represent local vectors;
flattening them alone adds no expressive power. Initially fit tokens per image
to isolate the decoder; a small image-to-token encoder belongs to later Project A.
This does not require DINO, a discrete codebook or quantum attention. Keep token
retrieval and scale-conditioned circuit processing as separate experiment factors,
and count latent storage and attention cost. V9 now implements this selected
token design; the broader proposal remains a guide to subsequent controls.

Distinguish this from completed V6: V6 generated a pyramid from the same global
64-number code but queried only its final 16x16 feature grid. Directly learned
local codes and querying multiple scales are separate changes. Control total
per-image code storage, compare single-scale and multiscale conditioning, and
retrain circuit-removal/classical controls. For the circuit extension also control
depth, repeated encoding and total execution/shot budgets. Data-dependent gates
need a new simulation path; the original single shared matrix cannot represent
the entire query-dependent circuit. Preserve the original reproduction,
V6/V8 code, checkpoints and results. Continue with plain explanations and static
figures when useful; no sliders or interactive panels.

Previous completed experiment, 2026-11-09: run the proposed frozen-DINO/RAE-feature QVF
readout and classical control for 50,000 updates. V8 is complete:
`../qvf_project_a/results/20261109_v8_readout50k/protocol.json` declares three
paired seeds (0, 1, 2), each with QVF and classical heads trained for 50k updates.
Logs: `../qvf_project_a/results/logs/20261109_v8_readout50k_seed{0,1,2}.log`.
All six runs reached 50k. The entry point is `rae_readout_study.py pair`; it runs
QVF then classical for one seed while sharing the frozen feature cache on GPU 0.
No training remains running. Preserve completed output directories. Explicit
`--resume` uses the last saved checkpoint and skips completed variants.

Final native validation PSNR is 24.97 ± 0.13 dB for QVF and 24.99 ± 0.18 for
the classical control. Earlier checkpoints selected by validation MSE reach
26.03 ± 0.11 and 26.13 ± 0.03 dB, respectively; the unchanged published RAE
reference is 25.81 dB. QVF selection steps (seeds 0/1/2): 11k/9k/3k; classical:
7k/7k/3k. Final training PSNR is 28.12 ± 0.39 / 28.03 ± 0.35 dB. Both heads
overfit under the 50k recipe. The final paired validation gap is only −0.027 dB
for QVF relative to classical, but this is not a formal equivalence claim.

This establishes that the original QVF quantum readout can retain approximately
RAE-level reconstruction quality with strong pretrained spatial features and
appropriate checkpoint selection, in exact simulation. It does not show a
quantum advantage or validate a small/cheap end-to-end model: the frozen feature
path still contains about 501 million classical parameters. It does not isolate
the circuit alone, and no diffusion was trained. Keep the earlier selected
checkpoints as the useful reconstruction candidates; further training alone is
not supported as the next improvement by these learning curves.

The [README](../qvf_project_a/README.md) contains the completed comparison and
commands. [Verified results](../qvf_project_a/results/20261109_v8_readout50k/verified/results.json),
[verification](../qvf_project_a/results/20261109_v8_readout50k/verified/verification.json),
and static galleries/learning curves are in that `verified/` folder.

`rae_features.py` cached 1152-channel, 16×16 post-normalization decoder features
before the pretrained RGB projection, with exact original-RAE output recovery
for every image. Cache: `../qvf_project_a/data/20261109_v8_rae_features/`.
`rae_qvf.py` adds a learned 1×1 channel projection and spatial residual block,
then queries each pixel's containing patch with within-patch and global position.
The original amplitude map, five-qubit/depth-five circuit and RGB readout remain
unchanged. The classical control shares the adapter/MLP architecture and uses
a learned affine RGB readout. Trainable parameters: QVF 500,150, classical
500,099. Shared starting weights and image/pixel streams match per seed.
All 35 tests and both native GPU reload preflights passed before launch; the
final suite has 36 passing tests including exact interrupted/resumed training.
No trained preflight weights were reused. Independent checks recompute every
final/selected metric, rerender all 5,000 images from each checkpoint with zero
RGB error, verify matching initial weights and all sampling streams, and confirm
fresh image-to-frozen-features-to-readout execution with zero error on 32
validation images per checkpoint. Every trained parameter group changed; the
original QVF simulator, pinned environments and published RAE weights are intact.

Previous user request: try a classical DINO-based RAE reconstruction baseline.
V7 is complete: the **published frozen DINOv2-B-with-registers encoder and
ViT-XL RAE decoder** were evaluated without local training or fine-tuning on the
same 4,744/256 CIFAR dog split (seed-0 IDs). Source/weights are pinned in
`../qvf_project_a/rae_assets.py`; the evaluator is `rae_baseline.py`.
Run directory: `../qvf_project_a/results/20261108_v7_rae_pretrained/`.
The [Project A README](../qvf_project_a/README.md) has the setup, exact commands,
protocol, results, static examples and limitations. No training remains running.

Native 32×32 reconstruction PSNR is 25.5565 dB on the training split and 25.8080
dB on validation, with no local optimizer updates or checkpoint selection. The
earlier selected width-128 QVF-VAE seed 0 scores 20.9415 dB on the same validation
images; RAE pixel MSE is 65.25% lower. Visual inspection shows substantially more
preserved structure, with some colour and fine-detail errors. All 5,000 raw
metrics were independently recomputed; reloading the published weights reproduces
all 256 validation and 32 fixed training outputs with zero RGB error. All 29
project tests pass. The official CIFAR test set remains unused.

Interpret this as a pretrained quality reference, not a matched compression or
quantum/classical control: the RAE has 86,582,016 encoder and 415,352,704 decoder
parameters, and a 16×16×768 latent (196,608 entries, versus our previous 64).
Encoder input is resized to 224 and decoder output is 256; outputs are downsampled
with antialiased bicubic interpolation to score the original 32×32 pixels.
External pretraining and its possible data overlap are not controlled. No RAE
training, new QVF, VAE or diffusion training was performed. Testing smaller
spatial features/decoders or the same DINO features with QVF is a next proposal,
not a completed experiment or an established quantum advantage.

RAE assets and a separate dependency venv are in `/home/agroz/.cache/quantai/rae/`.
The pinned quantai environment and original QVF simulator remain unchanged.
Source checkout: RAE `a4d18c4db766419cbe7cb8c02cd9f7ceb0ec9041`.
Original model modules run with torch 2.1.2 and Transformers 4.51.3; this is an
inference compatibility setup, not the authors' full training environment.
GPU 0 only. Latest communication preference: **plain explanations, no sliders or
interactive panels**; use links to static examples when helpful.

Previous design review after V6: think through a stronger classical reconstruction
architecture and good generalization before further QVF/VAE training. The design
review recommends a residual convolutional autoencoder with an encoder-produced
spatial latent and decoder refinement through the full 32×32 resolution. Compare
latent budgets such as 64, 256 and 512 entries using held-out images before
choosing the representation; these are proposed tests, not measured results.
Then compare conventional RGB and QVF readout paths with the same latent and
feature architecture. First establish pixel fidelity; evaluate perceptual losses
and mild VAE regularization separately. No new model was implemented or trained
during this design review. The earlier recommendation to immediately reconnect
the VAE is deferred pending this architecture discussion.

Current user priority, 2026-11-08: improve the QVF architecture's spatial
organization while retaining the quantum component, toward a useful larger-data
representation. The `20261108_v6` spatial study completed on GPU 0: three
convolutional-feature decoders and three parameter-matched pointwise controls,
each using directly fitted codes on 4,744 images for 50,000 updates. Protocol:
`../qvf_project_a/results/20261108_v6_spatial_protocol.json`; logs:
`../qvf_project_a/results/logs/20261108_v6_spatial_seed{0,1,2}.log`.
The controls have 299,705 parameters versus 299,894 for the spatial decoder.
The five-qubit/depth-five original QVF circuit and readout are unchanged.
Fresh codes, initial quantum angles, data IDs, RNG sampling, objective and
optimizer settings are matched. No encoder, diffusion, extra qubits or RGB
post-processing are added. All 26 tests, exact original-trainer regression and
native GPU checkpoint preflights passed before launch. All six runs and
`verify_spatial.py` have completed successfully. No training remains running.

Final training PSNR is 22.08 ± 0.06 dB for original width 128, 23.66 ± 0.04 for
the matched width-357 pointwise field, and 24.27 ± 0.02 for spatial QVF. The
spatial gain over matched capacity is 0.612 dB, with 12.8 percent lower pixel
MSE and 9.6 percent lower gradient MSE. Both width and spatial organization help;
do not attribute the entire gain over the original small model to convolutions.
Spatial images recover clearer structure but still miss fine texture. This is
a fitting result, not unseen-image reconstruction or generation. Every model's
lowest recorded fitting MSE occurs at step 50,000; no convergence claim.

The [current report](../qvf_project_a/results/20261108_v6_verified/report.html)
contains the architecture, learning curves and a fixed 256-image comparison.
All 4,744 training images were rerendered from each of nine final checkpoints
(six new, three reference), with zero native GPU RGB error. Verification also
checks raw metrics, data IDs, budgets, identical initial codes/angles and RNG
streams, source snapshots, and preserved V5 checkpoints. Desktop/mobile UI
checks pass all 768 gallery states and 3,072 exact pixel crops per theme.
The next proposed experiment is reconnecting the VAE and evaluating held-out
images; no new VAE or diffusion model was trained in V6. The quantum head is
preserved, but neither quantum benefit nor hardware scalability is established.

The original paper's shared image experiment uses 50 images; its three-shape
collection has many coordinate samples per shape. Original arXiv v1 also
describes a four-image higher-resolution test. Thus the limitation is small
collections, not exclusively low image resolution. Absence of convolutions
does not imply inability to learn structure, and this experiment cannot
establish that it was the sole cause of blur. See the updated
[spatial design](../qvf_project_a/CONVOLUTIONAL_QVF.md).

Previous user priority: before training a convolutional extension,
test QVF alone on the same larger dataset to see whether it also produces blurry
reconstructions. The three `20261108_v5` direct-code runs are complete:
4,744 training images, original width-128 field, five qubits, depth five,
independent 64-number image codes and 50,000 optimizer updates. Compare against
the VAE's training reconstructions, not its held-out scores. See the
[Project A README](../qvf_project_a/README.md) for the saved protocol and commands.
The preceding VAE duration/capacity study is complete; at the end of that study
the convolutional extension was an untrained numerical prototype.

Direct QVF reaches 22.08 ± 0.06 dB on those fitted training images, versus
21.52 ± 0.04 dB for the VAE at the same 50,000-update budget. Mean paired gain
is 0.562 dB; pixel MSE falls about 11.7 percent and gradient MSE about 14.3
percent. Direct fits recover more structure in many images but remain visibly
soft and lose fine detail. Removing the VAE helps but does not solve the blur
under this recipe. It removes encoder inference and VAE regularization together,
so it cannot isolate their contributions. The direct trainer uses the original
code-norm penalty and no gradient clipping, with constant Adam learning rates
to match the recent training budget. The decoder itself is unchanged.

All three runs finish at their lowest recorded fitting MSE, so do not claim
convergence or a fundamental five-qubit limit. All 24 tests pass. Independent
verification reloads every training image from both model families with zero
RGB error and checks full raw metrics, data IDs, source snapshots, budgets,
per-image visits and unchanged prior checkpoints. See the
[direct-fitting report](../qvf_project_a/results/20261108_v5_verified/report.html).
The original fitting baseline is complete. See [README.md](README.md) and the fresh
`results/20261108_vanilla/` audit and fitting runs. The earlier Project A RGB
rescaling was an additional design choice, not part of this vanilla baseline.
The original reproduction model in `qvf_fast.py` remains unchanged. The historical
classical-comparison agenda below is background, not authorization to expand the
current task into another advantage study.

Completed in this update: the fresh source audit passed; all nine direct-fitting
runs and final checkpoint checks passed. Individual fits reached 45.89 to 52.16 dB.
The shared 50-image model reached 29.76 ± 0.16 dB using release conventions and
29.48 ± 0.42 dB using paper-text conventions, over three seeds each. The report
and exact reproduction commands are linked from README.md. No further Project A
training was run as part of this baseline task.

Subsequent user-authorized VAE reconnection is also complete. See
[Project A README](../qvf_project_a/README.md) and its
`results/20261108_v3_verified_final/report_final.html`. New VAE models wrap the
unchanged `QVFImage` class directly and use the released grouped RGB readout.
The encoder and VAE loss are unchanged; old checkpoints explicitly retain their
earlier affine readout. Eighteen numerical/integration tests pass, and loading
the historical diffusion checkpoint reproduces its original 64 samples exactly.

Four fresh reconstruction-only runs completed. On the same 50 training images,
the VAE reaches 31.19 dB with mean codes and 30.14 dB with posterior samples at
the final 25,000-update checkpoint. This is a fitting capacity check with VAE
training conventions, not a controlled superiority comparison against the
direct-code fit. It overfits that small set. With 4744 training and 256 held-out
images, three seeds reach 20.52 ± 0.06 dB, versus 20.53 ± 0.05 dB with the old
readout under the same training protocol. Their training images also remain
limited to about 20.6 to 20.7 dB. The original RGB mapping alone does not resolve
the large-data blur; capacity, training and VAE regularization remain unresolved.

All new raw metrics, splits, source snapshots and checkpoints passed verification.
Native GPU reloads match exactly; CPU differences were isolated to the existing
TF32 convolution setting. No diffusion model was retrained. Do not report these
reconstruction results as generated-image quality or quantum advantage.

The subsequent duration/capacity study is also complete. The report is
[Project A's current report](../qvf_project_a/results/20261108_v4_verified/report_ready.html).
Three unchanged width-128 VAE runs reached 50,000 updates before three width-512
runs began. All use the same 4744/256 split, 64-number code, encoder, VAE loss,
learning rate, five qubits, depth five and original RGB readout. Width increases
decoder parameters from 48,950 to 588,470. The simulator is unchanged.

Selected held-out mean-code PSNR changes from 20.52 ± 0.06 dB at 7680 updates to
20.96 ± 0.02 dB with longer training, then 20.93 ± 0.01 dB with the wider field.
Final 50,000-update training/held-out PSNR is 21.52/20.94 dB for width 128 and
23.48/20.22 dB for width 512. The wider model overfits under this recipe; its
selected checkpoints occur between 5440 and 9280 steps. Do not describe this
as proof of optimal convergence or a fundamental qubit/code bottleneck.
All six runs pass raw-score, data-ID, source, budget and checkpoint verification;
native GPU reload error is zero. The longer baselines exactly reproduce every
earlier validation event through step 7680. Prior checkpoints are unchanged.

At the end of V4, [the convolutional design](../qvf_project_a/CONVOLUTIONAL_QVF.md)
was an executable, untrained proposal: global code to generated spatial features, then
local coordinate queries through the original five-qubit QVF head. It has
299,894 parameters and passed numerical/GPU checks. At that milestone the suite
passed 21 tests. The planned parameter-matched comparison (width 357, 299,705
parameters, same code and objective) has now completed in V6 as described above.
No convolutional model or new diffusion model was trained in V4 itself.

## 1. The task in one paragraph

We reproduced "Quantum Visual Fields with Neural Amplitude Encoding" (Wang, Theobalt, Golyanik, NeurIPS 2025, arXiv 2508.10900, PDF at `../../reference/R372_2508.10900_quantum_visual_fields_neural_amplitude.pdf`) with an exact, fast simulator. Their model's own numbers reproduce in one setting. Their classical baseline number does not reproduce in that setting, so the paper's 3 dB gap over an MLP did not survive, and removing the quantum circuit entirely changes nothing measurable. Your job is to try to break these conclusions: audit the implementation, search harder for a setting that gives both of the paper's numbers at once, and extend the controls. Do not start from the assumption that we are right.

## 2. What the model is (so you can audit it without the paper)

Per pixel, the input is `x = [z, gamma(c)]`: a 64-dim learned code `z` for the image (one code per image, initialised N(0, 0.01), trained jointly) and the pixel coordinate `c = (i, j) / 32` with a positional encoding `gamma(c) = [c, sin(2^k c), cos(2^k c)]` for k = 0..5 (26 values; the released code has no factor of pi, the paper's Eq. 1 has one).

- QVF: MLP with three hidden layers of width 128 and ReLU, output 32 "energies" E. Amplitudes `a = sqrt(softmax(E))` (a real 5-qubit state). A real orthogonal circuit `M` (RY and controlled-RY gates, depth J = 5, all-to-all CRY per layer, angles do not depend on the input) acts: `psi = a @ M`. Probabilities `p = psi^2`. Pauli-Z expectations `Z_q = sum_k p_k s_kq` with `s_kq = +1/-1` for bit q of basis state k. Colour read-out, two readings: released code ("parity") gives `[Z0 + Z3, Z1 + Z4, Z2]`; paper text gives `[Z0, Z1, Z2]`.
- QVF without circuit: identical, with `M = I` (depth 0). This is purely classical: a fixed signed sum of softmax probabilities. Parameters 48,800.
- Plain MLP: the same trunk, last layer 128 to 3, linear output. Parameters 45,059.
- Full QVF parameters: 48,950 (Gaussian init: angles N(0, 0.1^2)) or 49,100 (identity init: each layer is a block followed by its inverse with tied random angles in [0, 2 pi), so the circuit starts as the identity).
- Loss: `MSE + c * mean_i ||z_i||_2` with c = 1e-4 (released code: `sigma_Rval = 0.01`, squared) or 1e-3 (paper text: gamma = 1e-3). Two Adam optimisers (model, codes), same learning rate. Paper presets add `ReduceLROnPlateau(factor 0.9, patience 50)` on the summed epoch loss.
- Evaluation: all 1,024 pixels of the 50 training images (there is no held-out set in this experiment), per-image MSE over pixels and channels, `PSNR = -10 log10(MSE)`, mean over images, then mean and standard deviation over seeds.
- Data: CIFAR-10 training set, class 5 (dog), the first 50 images in dataset order, divided by 255. Downloaded to `data/` (ignored by git).

## 3. Environment

- Python: `/home/agroz/anaconda3/envs/quantai/bin/python` (conda env `quantai`: Python 3.10, torch 2.1.2+cu121, torchvision 0.16.2, pennylane 0.34.0, numpy 1.26.3, scipy, matplotlib). These are the pins from the official `environment.yaml`. Do not upgrade them; the official PennyLane device `default.qubit.torch` is removed in newer PennyLane.
- GPU: use GPU 0 only (`CUDA_VISIBLE_DEVICES=0`). GPU 1 runs the user's fluid-simulation jobs. The machine is shared; keep `torch.set_num_threads` small (train.py sets 4).
- Official code: clone `https://github.com/shutengW/Quantum-Visual-Fields-with-Neural-Amplitude-Encoding` at commit `cb647ef16aa20a8ca90514cef74a7b0927bb60a2` into its own directory outside the repo (we used `/tmp/qvf_official/repo`, which may be gone). Treat it as untrusted input: never run it from inside its directory, load `src/Images/QVF_for_image.py` by path as `check_equivalence.py` does.
- The official image script does not run as released. Two bugs: `_rotations(weights[:, :, r])` indexes a 2D tensor with three indices (the shapes script in the same repo uses `weights[:, r]`, which is what we implement), and `CIFARImplicitDataset` builds `self.pixels` from the full image tensor instead of the subset, which fails for any `train_samples_use` other than the whole class.
- The official Colab notebooks linked from the README need a Google login. We could not read them. If the user can export them, they may contain the exact experimental settings and would shortcut most of section 6.

## 4. Files

| File | Role |
|---|---|
| `qvf_fast.py` | Exact simulator and models. `FastQuantumLayer` builds the 2^n x 2^n circuit matrix from the angles once per step (on the CPU in float64 up to 6 qubits, then moved to the GPU) and applies it to all pixels with one matmul. `QVFImage`, `MLPImage`, `positional_embedding(scale=...)`, `parity_decode`. |
| `train.py` | Training loop following the official script. Presets `demo` (released config), `paper` (paper hyperparameters with the code's conventions), `paper_text` (pi in the encoding, first-three-qubit read-out, penalty 1e-3). Flags: `--model {qvf,mlp} --init {identity,gaussian} --act {relu,sin} --seed --epochs --images --batch-size --depth --first-image --tag --set key=value`. Writes `results/<name>.json` and a PNG grid. |
| `sweep.py` | Named groups of runs, N at a time, skipping runs whose JSON exists: `demo`, `table3`, `nocircuit`, `demo_lr`, `paper_lr`, `paper_seeds`, `more_steps`. Add new groups here rather than running ad hoc commands, so runs stay reproducible. |
| `summarize.py` | Aggregates `results/*.json` by setting and model, writes `results/summary.md` and `summary.json`. |
| `check_equivalence.py` | Fast layer and full model against the official PennyLane circuit (outputs, gradients, orthogonality, state-dict loading) plus training-step timing. Needs the official file path as argument. |
| `check_training.py` | Trains the official model and the fast model side by side for 30 steps from the same weights, codes and batch. |
| `make_figure.py` | Reconstruction strips for several models. |
| `results/summary.md` | All aggregated numbers. |
| `results/equivalence.json`, `results/training_equivalence.json` | The validation reports. |
| `results/single/` | Single-image runs (section 5, point 7). |
| `results/logs/` | One log per run. |

Run everything from this folder:

```
PY=/home/agroz/anaconda3/envs/quantai/bin/python
CUDA_VISIBLE_DEVICES=0 $PY train.py --preset paper --model qvf --init gaussian --seed 0
CUDA_VISIBLE_DEVICES=0 $PY train.py --preset paper --model mlp --seed 0 --set lr_model=0.003 lr_latent=0.003 --tag _lr0.003
CUDA_VISIBLE_DEVICES=0 $PY train.py --preset paper --model qvf --init gaussian --depth 0 --tag _depth0
CUDA_VISIBLE_DEVICES=0 $PY sweep.py paper_lr --jobs 12
$PY summarize.py
CUDA_VISIBLE_DEVICES=0 $PY check_equivalence.py /path/to/official/src/Images/QVF_for_image.py
```

Run times with 9 to 15 runs in parallel on one RTX 4090: plain MLP about 1 min, QVF Gaussian 2 to 3 min, QVF identity 3 to 5 min for 5,000 epochs at one step per epoch. Five steps per epoch takes about five times longer. Keep `--jobs` at 15 or below (about 0.8 GB of GPU memory per run).

Result names are built by `run_name()` in train.py: `<preset>_<model>[_<init>]_<act>_n<images>_seed<seed><tag>`. The sweep skips a run if its JSON exists, so never delete or overwrite result files; use a new tag.

## 5. What we established (numbers are PSNR in dB, mean and std over seeds)

1. Simulator exactness. Over 20 circuit configurations (2 to 6 qubits, depth 1 to 5, both initialisations, initial and generic angles) the fast layer matches PennyLane to 1.3e-15 in outputs and 6e-14 in gradients (float64). A full model with the official state dict loaded matches to 1.3e-15. Trained side by side for 30 Adam steps on 20 images, the loss curves agree to a relative 4.8e-6 (identity) and 3.5e-7 (Gaussian). See `results/equivalence.json` and `results/training_equivalence.json`.
2. Speed. One training step on the 50-image batch (51,200 pixel rows), paper config: fast 19 ms; official code 52.8 s on the GPU and 1.98 s on the CPU. The official GPU slowness is in the backward pass (batched complex128 `bmm` calls). Demo config: 6.4 ms against 7.06 s (GPU) and 0.34 s (CPU).
3. Paper Table 3 (50-image collection, image columns, MSE x1e-3 and PSNR): Ours(Gaussian)+ReLU 1.02 +- 0.11 / 29.8 +- 1.1; Ours(Identity)+ReLU 1.03 +- 0.09 / 29.5 +- 1.2; MLP+ReLU 2.17 +- 0.13 / 26.57 +- 0.51; Ours(Gaussian)+Sin 0.62 / 32.2; Ours(Identity)+Sin 0.72 / 31.4; MLP+Sin 1.19 / 29.2. Paper Table 2 (50 single images, n = 5): Gaussian+ReLU 0.98 / 30.06; Identity+ReLU 0.99 / 30.02; Gaussian+Sin 0.55 / 32.59; Identity+Sin 0.54 / 32.67; QIREN 0.78 / 31.03. Paper settings stated: Adam, initial learning rate 1e-3, plateau scheduler window 50 scaling by 0.9, 5,000 epochs, gamma = 1e-3, n = 5, J = 5, p = 128, "0.52e5 parameters", default.qubit.torch on an A100, three repetitions. Not stated: batch size, code length, positional-encoding depth, image subset, the baseline's exact architecture, what "+Sin" is.
4. Our ReLU results on the 50-image collection, 5,000 epochs (8 seeds per cell, 3 in the last row):

| Setting | QVF Gaussian | QVF identity | QVF no circuit | Plain MLP |
|---|---|---|---|---|
| lr 0.001, 1 step per epoch (paper text) | 26.85 +- 0.33 | 27.34 +- 0.32 | 26.74 +- 0.50 | 27.35 +- 0.30 |
| lr 0.003, 1 step per epoch | 28.90 +- 0.97 | 29.08 +- 0.58 | 29.10 +- 0.45 | 28.42 +- 0.48 |
| lr 0.01, 1 step per epoch (released config) | 27.99 +- 0.73 | 26.83 +- 0.84 | 28.60 +- 0.46 | 26.54 +- 0.29 |
| lr 0.001, 5 steps per epoch (batch 10 images) | 29.76 +- 0.16 | 30.23 +- 0.14 | 29.64 +- 0.39 | 30.36 +- 0.50 |

   The `paper_text` reading (pi, first three qubits, penalty 1e-3) at lr 0.001 gives the same picture within noise (3 seeds): 26.87 / 27.31 / 27.29 / 27.67.
5. Reading of the table. The paper's QVF numbers are matched by the last row (and roughly by lr 0.003). The paper's MLP number is matched only by lr 0.01 at one step per epoch, where the MLP degrades and the QVF variants are 1.5 to 2.7 dB below the paper. No setting gives both. With the learning rate chosen per model out of the three, the plain MLP is 0.5 to 0.7 dB behind QVF (Welch p 0.03 to 0.2, 8 seeds); with five steps per epoch it is ahead. The no-circuit model is within noise of the Gaussian-init QVF everywhere, 0.6 dB behind the identity-init QVF at lr 0.001 (p 0.015, uncorrected), and ahead of both at lr 0.01.
6. Released demo config (100 images, 400 epochs, lr 0.01, 4 qubits, depth 2, width 256; 3 seeds): QVF Gaussian 24.54, QVF identity 24.05, no circuit 23.19 +- 1.23, plain MLP 21.97. A learning-rate sweep at the demo settings (3e-4 to 3e-2) shows the same pattern: QVF variants benefit from lr 3e-3 to 1e-2, the plain MLP peaks at 22.4 and degrades above 3e-3; both QVF variants diverge at 3e-2 in 2 of 3 seeds.
7. Single images (paper Table 2 says about 30 dB). With the released positional encoding our single-image fits reach 54 to 58 dB (QVF) and 62 dB (MLP) in 5,000 steps. Without positional encoding (`--set L_embed=0`) they reach 24 to 27 dB for both models, with no difference between QVF and MLP. So the paper's single-image setting is neither of these. See `results/single/`.
8. Sin variants. Our reading is standard SIREN (w0 = 30, SIREN initialisation, same inputs). It reaches 36.6 to 36.7 dB (QVF) and 35.7 dB (MLP), far above the paper's 31.4 to 32.2 and 29.2. The paper does not define its Sin variant and the released code has none.
9. Parameter count. Ours 0.49e5 against the paper's 0.52e5 at p = 128. Candidates that give 0.52e5: a deeper positional encoding (L = 12 with the 64-dim code) or a longer code (about 88) with L = 6. Not tested.

## 6. What to try, in order

Report each item as done, with numbers, or as skipped with the reason. Keep 8 seeds for anything you want to compare, 3 for scouting.

1. Audit before running anything. Check `qvf_fast.py` against section 2 and against the official script. Points to verify: bit order and sign convention in `signs` (wire 0 is the most significant bit, as in PennyLane); the CRY control and target axes; the adjoint ordering in the identity block; that `parity_decode` matches the official `parity`; that `MLPImage` is the fairest reading of "MLP+ReLU"; that the per-image PSNR and the clamped variant in the JSON agree with how the paper reports (the paper's MSE and PSNR pairs are consistent with PSNR of the collection MSE within 0.4 dB). Rerun `check_equivalence.py` and `check_training.py` yourself; do not trust our JSON.
2. Baseline fairness. Give the plain MLP the same parameter count (last layer to 32 then a trainable linear 32 to 3, or width 131), and try a bounded output (sigmoid or tanh) at lr 0.01. If a bounded-output MLP recovers the QVF numbers at lr 0.01, the whole "advantage" is an output-range effect. Also run the MLP with the Boltzmann head removed only partially: softmax head with a trainable 32 to 3 linear read-out instead of the fixed signs. This is the closest classical twin and a direct test of what the orthogonal circuit adds.
3. Search for one setting that reproduces both paper numbers (QVF 29.8 and MLP 26.57 from the same runs). Unknowns to vary, one at a time first, then the plausible combinations: batch size (50, 25, 10, 5 images), `L_embed` (0, 6, 10, 12), `pe_scale` (1 or pi), `hid_dim` (128 or 256, the released config uses 256), `latent_size` (64 or larger), scheduler patience (50 per the paper, 15 per the released config), separate learning rate for the circuit angles (`lr_model_quantum` exists in the released config), `gaussian_std`, number of hidden layers (3 or 4), `use_parity`, penalty weight. Budget: about 150 runs of 2 to 3 minutes, 12 in parallel, so a few hours. Stop as soon as one setting gives both numbers within the paper's error bars, then run 8 seeds of all four models there. If no setting does, say so and show the table of what came closest.
4. Table 2. Find an input encoding and budget under which a single CIFAR image fits to about 30 dB in 5,000 steps with 0.5e5 parameters. Candidates: `L_embed` between 1 and 4 without pi, the code input removed, fewer steps per epoch, or evaluation on pixels not trained on (check whether the paper's protocol could be interpolation). This matters because Table 2 is the QIREN comparison the paper leans on.
5. More seeds for the identity-init versus no-circuit difference at lr 0.001 (currently 0.6 dB, p 0.015 with 8 seeds and many comparisons). 16 seeds each, plus the plain MLP, decides whether the circuit has any measurable effect in this benchmark.
6. Only if items 1 to 5 leave the picture unchanged: the scaling test. Same four models on 500, 5,000 and 50,000 images of CIFAR-10 (all classes), codes learned by optimisation, learning rate tuned per model, 3 seeds. The question is whether the circuit ever separates from the no-circuit head when the code becomes the bottleneck. Add a `scaling` group to sweep.py; expect 10 to 20 ms per 10-image step, so 50,000 images at 5,000 epochs is 70 to 140 hours per run and needs a smaller epoch count or a pixel subsample per step; decide and state it.

## 7. Rules

- Do not commit, push, or change anything under `reference/`, `reports/` or `research_notes/`. The PDFs stay ignored by git.
- Do not delete or rewrite existing `results/*.json`; new experiments get new tags or new presets.
- Keep the official clone outside the repo, load it by path, and never execute it from its own directory.
- Keep the reporting style: plain prose, no em or en dashes, lead with the answer, state assumptions as assumptions. Separate "our measurement under our reading" from "what the authors ran", which we do not know.
- Record every new run through `sweep.py` groups so the next person can rerun it with one command.

## 8. Open questions to answer explicitly in your report

1. Did the audit find any error in the simulator, the baseline, the loss, the scheduler or the evaluation? If yes, which numbers in section 5 change?
2. Is there one setting that gives QVF about 29.8 dB and the plain MLP about 26.6 dB in the same runs?
3. Does any classical twin (bounded-output MLP, softmax head with trainable linear read-out, no-circuit head) fall measurably behind the full QVF anywhere, with 8 or more seeds?
4. What setting reproduces Table 2's 30 dB on single images?
5. One sentence on whether the published 3 dB gap should be cited as evidence that the circuit helps.
