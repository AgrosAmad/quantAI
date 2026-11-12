# Convolutional features for QVF

Current direction after V8: the user prefers improving QVF with local features
at several scales before returning to Project A's encoder and diffusion. The
[next-study proposal](#next-study-directly-learned-local-codes-at-several-scales)
below describes directly learned spatial codes and queries across scales.
It also includes the user's proposed extension to scale-conditioned quantum
blocks and a token-based variant with coordinate-to-token attention. The unchanged
QVF circuit and bilinear feature lookup remain controls. Selected variants are
now implemented in the [completed V9 pilot](README.md#token-qvf-pilot-v9);
other designs below remain proposals. The existing implementation and results in
the following sections describe the completed V6 experiment.

The extension generates a spatial feature grid from an existing
64-number global code, then uses those local features in the QVF query for each
pixel. The five-qubit circuit and original grouped RGB readout remain intact.
This introduces spatial structure into the classical amplitude generator.

The decoder is in [conv_qvf.py](conv_qvf.py). It passes interpolation,
gradient, chunked-query and GPU execution checks. The V6 direct-fitting study
completed under its [saved protocol](results/20261108_v6_spatial_protocol.json):
three spatial models and three parameter-matched pointwise controls, each with
50,000 updates. All runs passed independent verification. The preceding V4 duration and width
experiments used the original pointwise QVF architecture.

This first architecture comparison obtains codes by per-image optimization,
without an encoder or VAE loss. It follows the V5 diagnostic, where original
QVF alone still produced soft reconstructions on 4,744 training images. A later
VAE experiment can supply the same 64-number code through the encoder.

The spatial model reaches 24.27 ± 0.02 dB on the fitted training images, versus
23.66 ± 0.04 for the parameter-matched pointwise model and 22.08 ± 0.06 for the
original small field. Relative to the matched control, spatial features reduce
pixel MSE by 12.8 percent and adjacent-pixel gradient MSE by 9.6 percent. This
supports the architectural direction at this budget, while fine detail remains
imperfect. It does not yet test a VAE or held-out inference. See the
[verified interactive results](results/20261108_v6_verified/report.html).

## What the original paper established about scale

The paper's shared image-collection experiment fits 50 images; its 3D collection
contains three shapes with 100,000 spatial samples per shape. It also demonstrates
higher-resolution image representations, including a 400 by 350 image in v2.
The original v1 additionally describes a four-image higher-resolution experiment.
Many queried coordinates of one image are not the same as many distinct images
sharing a learned representation. The authors acknowledge experimental scale
limitations from simulation. Our 4,744-image, single-class, 32 by 32 experiment
extends the image-collection test, but remains small compared with large image
models. [QVF v2 sections 5–6](https://arxiv.org/html/2508.10900v2#S5),
[original v1 image experiments](https://arxiv.org/html/2508.10900v1#S5.SS2).

## How a conventional decoder handles an image

A convolutional decoder often expands the latent code into a small grid of
features, then repeatedly enlarges that grid and applies learned filters.
Each filter combines information from a local neighbourhood and is reused at
different positions. Intermediate grids contain many feature channels, not RGB
photographs. The final layer turns the features into three colour channels.

The released latent-diffusion decoder uses convolutional residual blocks and
upsampling followed by convolution. Its autoencoder training also uses
perceptual and adversarial losses, so its visual results cannot be attributed
to convolutions alone.
[Decoder implementation](https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/diffusionmodules/model.py),
[perceptual compression objective](https://arxiv.org/html/2112.10752v2#S3.SS1).

Our current QVF instead concatenates the same global image code with each
pixel's encoded coordinate. A shared MLP produces 32 energies, which become
five-qubit amplitudes. The circuit acts on that coordinate's state, and its
measurements produce RGB. Different pixels are coupled through the shared code
and network weights, but there is no explicit convolution between neighbouring
pixel features. The circuit mixes amplitudes within one query; its wires are
not neighbouring image pixels.

Increasing the amplitude MLP's width adds capacity while preserving this
pointwise organization. Convolutional features would change the organization
as well as the capacity.

The five qubits do not encode the whole photograph in a single state. Each
coordinate query constructs its own 32-amplitude state. The shared code and
network describe the image, and repeated queries render its pixels. Keeping
five qubits therefore does not restrict the output to 32 pixels, but rendering
more pixels still requires more queries. Adding circuit layers alone would
make the transformation within each query richer; it would not introduce
convolutions between neighbouring image locations.

## Proposed decoder with local features

The prototype uses this sequence:

1. Start with a 64-number global code, fitted directly for this decoder study
   or supplied by a VAE encoder in a subsequent experiment.
2. A linear layer expands it into a 4 by 4 grid with 128 channels.
3. Upsample and apply two convolutions to obtain an 8 by 8 grid with 64 channels.
4. Upsample and apply two more convolutions to obtain a 16 by 16 grid with
   32 channels.
5. At a requested pixel coordinate, interpolate a 32-number local feature vector
   from that grid. Concatenate it with the original 26 coordinate features.
6. A two-hidden-layer amplitude MLP, width 128, produces 32 energies. The original
   QVF amplitude normalization, five-qubit circuit and RGB readout finish the query.

This retains a coordinate-query interface instead of requiring a direct
32 by 32 RGB output tensor. The feature grid can be computed once per image and
reused across query chunks. Querying extra coordinates does not, by itself,
establish that the model recovers additional real image detail.

Local features plus a coordinate-conditioned image function have a classical
precedent in LIIF. LIIF learns its features from an observed image for
super-resolution; this proposal would generate features from a VAE code and
retain a QVF measurement head. It is not an implementation of LIIF, and does not
include LIIF's local ensemble, feature unfolding or cell-size conditioning.
[LIIF paper](https://arxiv.org/abs/2012.09161),
[released query implementation](https://github.com/yinboc/liif/blob/main/models/liif.py).

## What changes and what stays fixed

| Component | Current vanilla-decoder VAE | Local-feature prototype |
| --- | --- | --- |
| Image code | 64 global numbers | Same 64 global numbers |
| Encoder | Existing convolutional VAE encoder | Intended to remain unchanged |
| Features supplied per coordinate | Global code and coordinate encoding | Interpolated local features and coordinate encoding |
| Classical decoder | Three-hidden-layer pointwise MLP | Convolutional grid generator and two-hidden-layer query MLP |
| Quantum state | 32 amplitudes, five qubits | Same |
| Circuit | Depth five, Gaussian initialization | Same architecture and initialization convention |
| RGB readout | Released grouped Z expectations | Same, with no added affine map |
| Decoder parameters | 48,950 | 299,894 |
| Training evidence | Measured in the completed studies | V6 direct-fitting comparison completed |

The generated feature grid does not add image information beyond the code. It
is computed from the code, without an encoder skip connection or access to the
original image. This preserves the intended generative path: a future diffusion
model could produce a 64-number code and the decoder could render it.

A spatial VAE latent would be a different change: the encoder would output a
grid of posterior means and variances, often with many more latent entries.
That would change the information bottleneck and the latent diffusion model.
It should be tested separately from generating spatial features from the
existing global code.

At the quantum boundary the classical feature dimension does not determine the
number of qubits. The amplitude network still outputs 32 amplitudes. Larger
classical feature maps therefore do not require adding qubits to this head.
There is still one quantum query per queried coordinate. This construction
does not establish a reduction in state-preparation, measurement or hardware
cost, nor a quantum advantage.

## Numerical contract and implementation details

The prototype has 271,552 feature-generator parameters, 28,192 amplitude-network
parameters and 150 stored circuit-angle entries, for 299,894 in total. As in the
original simulator, 25 diagonal controlled-rotation entries are unused.

Coordinates remain ordered as row then column. The local sampler maps coordinate
0 and 1 to the first and last feature centres, with border handling outside that
range. This matches bilinear `grid_sample` with `align_corners=True` after swapping
to its column-row order. The sampling convention is explicit; changing it would
change the model.

The reference implementation builds bilinear weights explicitly. Tests compare
its outputs and feature gradients against PyTorch's sampler, including boundary
queries. This dense reference is easy to audit but may do unnecessary work;
an optimized four-neighbour sampler can be evaluated separately. GPU forward and
backward also pass under the project's existing runtime.

RGB reconstruction gradients reach the global code, convolutional generator,
amplitude network and circuit. Rendering coordinate chunks from a reused feature
grid reproduces the unchunked output. These checks establish that the proposed
computation is connected correctly, not that it learns a useful representation.
See [the prototype verification record](results/20261108_v4_convolutional_prototype_final.json).

## Controlled decoder comparison, V6

The primary comparison uses directly fitted codes for both architectures.
Width 357 gives the original three-hidden-layer amplitude MLP 299,705 decoder
parameters, within 0.1 percent of the spatial decoder's 299,894. The original
width-128 V5 runs are preserved as a secondary reference.

The two new arms use the same direct-code training loop, 4,744 image IDs,
64-number code, initial code values and initial circuit angles. They also share
the original amplitude map and grouped RGB readout, five qubits, depth five,
image order, pixel RNG, Adam settings, code-norm penalty, and 50,000 updates.
No trained weights are reused. The 256 validation images do not supply fitted
codes or select checkpoints. The objective remains pixel MSE plus 0.0001 times
the mean code norm; there is no encoder, KL, perceptual, adversarial or edge loss.

Full-training-set PSNR, pixel MSE and adjacent-pixel gradient MSE are measured
at the final checkpoint. A predetermined 256-image gallery supports visual
inspection, while raw tensors and metrics cover every training image.
All 26 tests and GPU preflights passed before launch. A numerical regression
against the archived V5 trainer gives identical original-model weights, codes
and metrics after the shared trainer gained a decoder-factory argument.

Matched parameter counts and coordinate-query budgets do not imply equal
compute: the spatial generator builds a reusable feature grid. Wall time and
peak memory are recorded, but GPU 0 is shared with unrelated work, so this is
not an isolated speed benchmark. The architecture comparison changes locality,
activation functions and parameter allocation together; it does not isolate a
single convolution operation.

## Gates toward a usable generative representation

1. Completed: the spatial decoder improves fitting at comparable capacity under
   the V6 protocol, consistently across three seeds.
2. Next: reconnect the unchanged encoder and VAE objective, then measure reconstruction
   of held-out images. Better fitting alone cannot establish generalization.
3. Broaden the image classes and resolution. If 64 global numbers constrain
   detail, separately test a spatial latent; that changes the encoder's
   information bottleneck and the later diffusion model.
4. Measure cached feature-grid costs, state preparation, quantum query count,
   shot precision and noise. A small qubit count is not a runtime guarantee.
5. Use separately trained circuit-removal/replacement controls to determine
   what quantum processing contributes. Trainable circuit angles or a mandatory
   quantum output path do not by themselves establish quantum benefit.

The improved fits support the spatial decoder as a candidate for the next VAE
test. The global code, positional encoding, activation functions, objective and
optimization remain candidates for further improvements. The result is specific
to this recipe and budget, not a proof of a fundamental quantum capacity limit.

## Next study: directly learned local codes at several scales

Status: direction recorded after the user's post-V8 architecture discussion.
Selected variants are now implemented and tested in the V9 pilot; consult its saved
protocol for actual grid sizes, qubit counts and budgets. The immediate research
question is whether local conditioning at several spatial scales improves a
shared QVF's reconstruction and transfer under explicit representation and
compute budgets. Classical coordinate-based INRs can represent images without
convolutions; the hypothesis concerns conditioning and resource allocation, not
an inherent inability of coordinate queries to describe images.

### Reference candidate: multiple scales, original circuit

1. Give each training image a compact collection of learned feature grids at
   coarse, intermediate and fine spatial scales. These are classical latent
   parameters fitted from RGB supervision, not image pixels supplied at inference.
2. For a requested coordinate, bilinearly sample local features from every grid.
   Coarse grids are intended to support broad context and fine grids local detail;
   their exact semantic roles are learned, not guaranteed by the grid sizes.
3. Concatenate those sampled features and the original coordinate encoding.
   Use a small shared MLP to produce 32 amplitude scores, followed by the original
   normalization and square root.
4. Apply the unchanged five-qubit/depth-five QVF circuit and released grouped
   Z-expectation RGB readout. Every colour channel passes through this path.
5. Jointly optimize the per-image grids, shared MLP and circuit angles. First
   study direct fitting; an encoder that predicts the grids and a diffusion
   model that generates them belong to later Project A work.

This candidate does not use the pretrained RAE decoder body, an RGB bypass or
a learned network after quantum measurement. It remains a hybrid model because
its codes and amplitude network are classical. Every queried coordinate still
requires its own quantum input state. Several feature scales do not, by
themselves, reduce quantum queries or measurement cost.

Exact grid sizes and channel counts are draft choices. Report each image's total
code entries as the sum of height times width times channels over all grids,
including any global code if one is later added. Keep code initialization and
regularization comparable; a norm penalty should apply consistently to the
complete code rather than acquire a different weight merely from splitting it
into more grids. No optimal code size or scale allocation is established.

### What this adds beyond V6

V6 creates intermediate 4x4, 8x8 and 16x16 feature maps from a global 64-number
code, but only the final grid feeds coordinate queries. It is a generated
feature hierarchy with a single query scale. The candidate instead fits local
codes directly and samples multiple scales for each query. Those are two
distinct changes and need separate controls. A growing grid must not silently
turn into storing the target image with little work left for the shared field.

### Comparisons and evidence needed

- Compare directly fitted single-scale and multiscale grids at the same total
  per-image code budget. Include a global-code QVF at that budget and record or
  closely match shared decoder parameters. Match quantum configuration, data IDs,
  query streams and optimizer budgets. Equal parameters do not imply equal compute.
- If expanding the code budget, treat it as a separate experiment factor. The
  previous 64-code runs are historical references, not matched controls for a
  representation with more independently fitted values.
- Retrain a circuit-free control that retains the amplitude map and measurement
  rule but uses identity evolution. Also compare a classical readout on the same
  feature organization. Report the latter as a readout-package comparison; turning
  off a trained circuit without retraining is not an adequate contribution test.
- Separate training-image fitting from transfer to images excluded from shared
  decoder training. For a completion test, freeze the decoder, fit new codes on
  a declared subset of observed pixels, and score the unobserved pixels. Declare
  observation masks and code-fitting budgets. This is optimization-based transfer,
  not one-pass encoder generalization. Preserve the official test set for a
  later locked evaluation.
- Use native raw RGB reconstruction and edge/detail diagnostics plus fixed image
  examples. Record shared weights, per-image code storage, classical computation,
  state-preparation assumptions, circuit calls and shot budgets. Exact simulation
  and finite-shot measurements are distinct evaluation conditions.

Start with numerical checks and a small fitting pilot before selecting a full
study protocol. Preserve earlier architectures and outputs under their own names.
Improved fitting, generalization, practical quantum execution and quantum benefit
are different claims; each requires its own evidence. Local or multiscale
conditioning already has classical precedents, including
[LIIF](https://arxiv.org/abs/2012.09161) and
[Locality-Aware Generalizable INR](https://arxiv.org/abs/2310.05624).
Using these ideas in a QVF is a research hypothesis, not an established novelty
or advantage claim.

### Circuit extension: inject different scales at different stages

The user also wants to explore the quantum circuit itself, with different
components handling different spatial scales. Keeping the original circuit is
therefore a reference condition, not a restriction on every proposed model.
Status: the staged single-register extension is implemented in V9 at five and
seven qubits. Independent branches and separate qubit groups remain proposals.

In the original implementation, input dependence enters through the prepared
amplitudes. The learned circuit angles are shared constants across images and
coordinates. Splitting that same input-independent circuit into named "coarse"
and "fine" blocks would not by itself route different features to those blocks.

The first proposed extension uses the same five qubits throughout one query:

1. Sample coarse, intermediate and fine local features at the requested
   coordinate, using the same spatial-code representation as the reference.
2. Prepare amplitudes from coarse features and the original coordinate encoding
   with a small classical MLP and the original square-root-softmax map. Apply a
   first block of shared trainable QVF gates.
3. Project intermediate features to five input-dependent RY rotation angles.
   Apply these rotations to the existing state, followed by a second block of
   shared trainable gates.
4. Project fine features to five input-dependent RY angles, apply them to that
   state, and apply a final block of shared trainable gates.
5. Measure only at the end, retaining the released grouped-Z RGB readout.
   Train all code grids, classical maps and circuit parameters jointly.

Here "shared" means reused across images and coordinates; the three processing
blocks can have different trained parameters. The scale-specific feature maps
produce different rotation angles for different queries. The state is neither
measured nor reset between scales. Replacing it with a newly prepared amplitude
state at each stage would discard earlier processing instead of implementing
this design. Real RY/CRY gates suffice for a first variant and preserve real-valued
state simulation. Exact angle scaling and initialization remain pilot choices.

An initial capacity control can distribute the original five QVF repetitions
across the three blocks, for example 2 + 2 + 1, rather than giving each scale
five repetitions. The feature-dependent rotations and their classical projection
weights still add resources; count them, and report compiled gate counts when
estimating hardware cost. Repeated encodings do not increase the register size,
but neither state preparation nor per-pixel shots become free.

The intended benefit is that the quantum evolution can respond to new local
information at successive scales, with the final prediction depending on their
composition. This is a hypothesis about conditioning and trainability, not a
claim that a five-qubit state stores all features losslessly or that coarse-to-fine
ordering guarantees better detail. Grid resolution does not enforce a frequency
band: a "fine" block may also change large-scale colour or shape.

Alternative designs to distinguish from this proposal:

- Independent circuit per scale, followed by classical combination of measured
  outputs. The same physical qubits can be reused sequentially, but each branch
  requires a new preparation and its own measurements. This provides no quantum
  entanglement between branches. Compare at a declared total execution/shot
  budget, not equal shots per branch while ignoring the extra branches.
- Separate qubit groups within one joint register, with feature-specific
  encodings followed by cross-group gates. This permits entanglement between
  groups. Partitioning the current five qubits leaves smaller groups; allocating
  five to every scale increases total qubits. Defer this design until the
  fixed-five-qubit staged variant has evidence behind it.

### Controls for the circuit extension

Keep spatial representation choices and circuit choices as separate experiment
factors. Establish the original-circuit reference on the same feature pyramid
before attributing a change to quantum scale processing. In addition to the
earlier reconstruction and transfer checks:

- Compare single-upload and staged encodings at controlled classical capacity,
  trained gate counts and per-query cost. Use a depth control where appropriate;
  equal trainable parameter counts alone are not an equal-compute comparison.
- Compare scale-specific stages with a repeated-encoding control that mixes
  scales at each stage. Match feature-map/projection and gate budgets where
  possible, and disclose remaining differences. A fixed invertible mixing of
  equal-width scale vectors is one possible control with no added learned
  parameters; it is not the same as concatenating every feature at every stage.
  Also compare scale order when input dimensions permit the same modules to be
  reused. These controls separate the value of scale routing from repeated
  input encoding alone.
- Retrain a variant without controlled entangling gates while retaining
  data-dependent single-qubit rotations. This tests those gates' contribution;
  amplitude preparation itself may already produce an entangled state, so it
  is not a fully unentangled model. Separately retrain a fully circuit-free
  classical readout on the same features. Removing trained constant gates but
  retaining feature-dependent rotations is not a circuit-free ablation.
- Keep the output rule fixed initially. Extra observables, learned RGB mixing,
  larger classical networks and different state-preparation maps are additional
  factors, not silent parts of the scale-routing comparison.

The original simulator builds one shared 32-by-32 circuit matrix per forward
pass. Input-dependent rotations prevent that shortcut from representing the
entire new circuit for all queries with one shared matrix. Fixed processing
blocks can still be precomputed separately, with batched state-vector operations
for the input-dependent gates. Implement any extension separately from
`../qvf_repro/qvf_fast.py`; check outputs and gradients against an independent
gate-by-gate implementation before a training pilot, and measure the actual
simulator cost rather than assuming the previous runtime.

Repeated classical encodings interleaved with quantum processing have precedent
in [data re-uploading](https://arxiv.org/abs/1907.02085). Feature-specific quantum
branches also appear in [3D-QISR](https://arxiv.org/abs/2601.05250), where the split
is between position and view-dependent information, not the proposed image-scale
pyramid. These connections inform the design; neither establishes novelty or
performance for this proposal. A fuller related-work check is needed before
claiming a new architecture. The immediate research question is whether routing
local scales into quantum evolution improves fitting and held-out transfer at
explicit representation and execution budgets.

### Token extension: coordinate queries select local information

The user also proposes tokenization. This is compatible with the multiscale
representation and staged quantum circuit, but should be tested as a separate
architecture factor. Status: continuous positioned tokens with classical
cross-attention are implemented and tested in the V9 pilot. Image encoders and
token-to-token self-attention remain later work.

Here a token is a continuous feature vector with an associated spatial position
and scale. It is not a discrete vocabulary index or a VQ-VAE codebook entry.
Each cell in a feature grid can already be viewed as such a token. Flattening
the grid to a token list, while retaining the same bilinear lookup, is a storage
change and should not change predictions or improve expressivity. The substantive
candidate change is how coordinates retrieve information from those vectors.

Two sources of tokens must remain distinct:

- For the initial decoder study, fit spatially anchored token vectors directly
  for each image, just as proposed for the grid codes. They are latent parameters,
  not observed RGB patches supplied to the decoder at inference. This remains
  an auto-decoder with optimization-based inference for new images.
- In later Project A work, train a small image encoder to predict the tokens,
  for example from image patches or convolutional feature maps at several scales.
  That supplies one-pass image-to-token inference. DINO and a large pretrained
  transformer are not requirements of tokenization. A VAE distribution or a
  discrete codebook would be further choices, not consequences of using tokens.

Candidate coordinate-to-token attention for one RGB query:

1. Embed the requested coordinate with a small shared classical map to form a
   query vector.
2. Map each token to a key and value. Keys describe what information is available;
   values carry the information to combine. Include explicit token position and
   scale, and a positional preference for nearby tokens. A content vector alone
   does not tell the model where a patch belongs.
3. Compare the query with candidate keys and normalize the resulting scores into
   weights. Form a weighted sum of the values at each scale. These are learned,
   query-dependent weights in place of fixed bilinear interpolation weights.
4. Produce one fixed-width feature summary per scale. First concatenate them
   into the reference QVF amplitude map; then test the same summaries with the
   proposed coarse, intermediate and fine quantum stages.
5. Use the same final RGB readout and reconstruction supervision. Attention,
   projections, token codes and circuit parameters are trained jointly.

Attention is classical in this first design. The quantum circuit processes the
retrieved summaries through amplitude preparation and/or feature-dependent gates;
it does not hold all tokens simultaneously or assign one qubit per token. This
still predicts RGB from a coordinate and an image representation, so it remains
a conditional INR. Token-to-token self-attention is a possible later addition
that lets tokens exchange information before querying, but is not required for
this coordinate-to-token cross-attention design.

The hypothesis is that learned retrieval can combine relevant context and local
detail more effectively than fixed spatial interpolation. Tokenization alone
does not guarantee semantic features, compression, sharper images, generalization
or reduced quantum cost. A coarse/fine token hierarchy also does not enforce
frequency-separated information without further constraints.

### Controls and cost for token retrieval

Use the same token positions, scale allocation, total per-image code entries and
summary dimensions when comparing bilinear and attention retrieval. Keep the
value projections comparable, count the extra query/key parameters, and include
a classical-capacity control if attention adds substantial shared capacity.
Factor the comparison into bilinear versus learned retrieval and original versus
staged quantum processing, so a gain from attention is not attributed to the
circuit. A pilot can change one factor at a time before a full comparison.

Dense attention over every token for every pixel scales with the product of
queried pixels and token count, as well as feature width. A positional bias alone
does not reduce that computation. Local candidate sets plus a small number of
coarse-context tokens are an efficiency option; overlapping or smooth selection
needs checking to avoid visible discontinuities at hard window boundaries.
Neither reshaping a grid into tokens nor using attention reduces the QVF's
per-coordinate circuit evaluations automatically. Measure retrieval, state
preparation, circuit execution and shots as separate costs.

Per-image storage is the sum of token count times token width over scales, plus
any other fitted per-image parameters. More tokens must not silently increase
the latent budget relative to the global-code or grid controls. Continue the
earlier fitting/held-out-completion protocol. During completion, tokens or an
encoder must only use the declared observed pixels, never the RGB targets being
scored; a full-image encoder reconstruction is a different evaluation condition.

Patch-based image tokens are established in
[ViT](https://arxiv.org/abs/2010.11929). More directly relevant,
[Locality-Aware Generalizable INR](https://arxiv.org/abs/2310.05624) uses an image
encoder to produce latent tokens and coordinate-conditioned cross-attention to
aggregate them for an INR, with coarse-to-fine modulation. Our initial directly
fitted tokens differ from that encoder setup, and replacing the classical field
with QVF would be a hypothesis to evaluate. Token-based INR conditioning alone
is not a new contribution. The proposed study asks whether compact token
retrieval helps a QVF, and whether scale-conditioned quantum processing adds
anything beyond that classical retrieval.

### Post-V9 follow-up: improving edge fidelity beyond attention

The user asks how to improve the staged model's edge error relative to attention
alone. The selected all-scale-input/edge-loss comparison is complete in
[V10](README.md#edge-fidelity-study-v10): four conditions, three seeds, five
qubits, 1,024 images and 10,000 updates. Its saved protocol fixes the loss weight,
sampling, duration and inference conditions. Adding edge loss reduced training
edge error by 10.68% for attention and 9.55% for revised stages, while improving
pixel PSNR. However, revised stages and attention with the same edge loss have
essentially identical mean training edge error. Sparse-completion edge benefits
are small and inconsistent; the extra stages do not yet establish an edge win.
All 12 runs passed independent verification. Longer training itself mattered:
the attention/pixel control reduced edge error by 34.70% from 2k to 10k updates.
The numbered design below records the rationale for V10; other suggested
variants remain untested. The
[edge curves](results/20261109_v9_tokens_pilot/verified/edge_diagnostic.png) are
derived from the completed V9 histories. The current edge metric is training-image
error in signed differences between horizontal/vertical neighbours, averaged
across orientations. It is not a perceptual sharpness score or a parameter-gradient
norm. V9 optimized pixel MSE plus code regularization; this edge metric was only
monitored. Edge error was still decreasing at the final pilot checkpoint.

Two targeted changes are worth separating:

1. Preserve all-scale amplitude conditioning when adding stages. Attention-only
   sends all three scale summaries into the amplitude MLP. V9 staged encoding
   instead sends only the coarse summary there, while intermediate/fine summaries
   enter through small linear-to-RY maps. This restricts their entry path; it is
   a possible explanation for the edge tradeoff, not an established cause. A new
   variant can retain the full attention input at state preparation and re-encode
   intermediate/fine information between blocks. At zero added rotation angles,
   it mathematically recovers the attention-only circuit, with the original five
   trainable repetitions split across blocks. All RGB predictions still pass
   through the quantum readout. Check this equivalence numerically, and initialize
   the added rotations at identity or near identity. Do not modify saved V9 code.
2. Add a supervised neighbour-difference loss alongside pixel MSE. Penalize the
   difference between predicted and target colour changes across adjacent pixels,
   using the same signed finite-difference convention as the metric. This is not
   a penalty on the predicted gradients alone, which would encourage smoothing.
   Retain pixel fidelity and code regularization; test modest loss weights rather
   than assuming one coefficient is optimal. Gradient-difference objectives have
   precedent in [image prediction](https://arxiv.org/abs/1511.05440).

Use a small factorial comparison at fixed qubits (five is a reasonable first
choice), fixed 336-value codes, identical sampling streams and declared total
quantum-query budgets: attention versus all-scale-initialized staged encoding,
each with pixel-only versus pixel-plus-edge reconstruction loss. A baseline
trained with the same edge objective is essential: improving a metric by adding
it to the loss does not isolate an architectural or quantum benefit. The old
coarse-initialized staged model remains a reference. Match training duration
across variants, and include a longer-duration control before inferring a limit
from the 2,000-step pilot.

The existing random pixel sampler is not sufficient for a direct neighbour loss:
sample actual adjacent pairs or small patches, never adjacent entries of a
shuffled pixel list. Count extra pixel queries and apply the same sampling/query
budget to all comparison arms. Declare how the sampled pixel and edge objectives
are averaged, including boundary handling and the absence of pixel-spacing
rescaling in the current metric. For sparse completion, only observed-observed
pairs may supply an edge training target; withheld pixel colours must not enter
code fitting. Score held-out edges with a separately declared mask, preferably
pairs whose two endpoints were both unobserved.

Keep native raw RGB MSE/PSNR, edge error and fixed galleries together. Evaluate
whether better edges trade away colour accuracy or introduce ringing. If the
two targeted changes are insufficient, test positional features in later upload
maps or a finer token grid as separate factors. A finer grid adds fitted image
information unless token storage is explicitly reallocated/matched. Frequency
losses or periodic amplitude networks are further options, not part of the first
comparison. Plain unweighted Fourier-domain squared error would duplicate pixel
MSE up to normalization; a genuinely frequency-selective objective requires
different weighting (for example [Focal Frequency Loss](https://arxiv.org/abs/2012.12821)).

### Post-V10: redesigning local conditioning and the quantum interface

The user asks for substantial architectural improvements, including how data
enters and leaves the circuit. The proposals below are research hypotheses. The
user subsequently authorized directions 1/2 separately, together, then direction
3; the selected comparison completed on 2026-11-10 in
[V11](README.md#interface-architecture-study-v11). All 27 runs passed independent
verification. Patch input, angle loading and their combination did not beat
attention at the fixed budget. The quantum feature-grid decoder had worse
training fits but approximately the original attention model's mean completion
PSNR and slightly lower mean completion edge error, with mixed seed differences.
It beat its matched classical feature-grid control on completion in all three
seeds, while wider attention remained better on mean quality metrics. Its main
concrete resource change is 64 instead of 1,024 circuit queries for a full image;
no hardware speedup or substantial overall quality gain was established.
The rationale and untested alternatives below remain a record of hypotheses,
not positive claims for designs that V11 did not support. A small frozen-readout diagnostic
was run to prioritize them; the original QVF, V9 and V10 implementations and
checkpoints remain unchanged. Other listed ablations remain untested.

The current attention model uses fitted 2x2, 4x4 and 8x8 grids, with four values
per location (336 values/image). Attention computes one four-value weighted
summary per scale, then concatenates those 12 values with 26 coordinate features.
A classical MLP produces 32 positive real amplitudes using square-root softmax.
Five-qubit RY/controlled-RY processing precedes five Z expectations. The fixed
five-qubit RGB rule sums wires 0+3, 1+4, and 2 for red, green and blue. V10's
extra uploads use medium/fine summaries through linear-to-RY maps; explicit
coordinates are absent from those maps, although the summaries themselves depend
on the query coordinate. These are architectural restrictions, not proofs of the
cause of blur. Three outputs suffice for RGB, and an expressive circuit can map
useful information into single-qubit averages.

**Completed diagnostic: a learned readout alone is not a strong edge fix.**
[readout_probe.py](readout_probe.py) freezes the three attention/edge V10 models
and their fitted image codes. It fits one global affine RGB map per model by
full-image least squares on the same pixel-plus-0.5-edge objective. Compare a
learned map of the existing five Z means (18 parameters) with a map of those
means plus all ten ZZ pair correlations (48 parameters). No coordinate or token
skip enters the RGB map. The original readout is contained in both affine
function classes. Code regularization is constant because codes are frozen.

| Frozen-state readout | Training PSNR, dB | Training edge MSE |
| --- | ---: | ---: |
| Original fixed RGB rule | 24.0804 | 0.00372956 |
| Learned affine map, five Z means | 24.1986 | 0.00373248 |
| Learned affine map, five Z plus ten ZZ | 24.2054 | 0.00373202 |

Means across seeds 0/1/2 on all 1,024 original training images. The larger map
adds only 0.0067 dB beyond learning the five-mean map. Edge error does not improve;
both affine alternatives worsen it slightly while lowering their combined
pixel/edge objective. This suggests little immediately recoverable detail in
these particular frozen states under an affine Z/ZZ readout. It does not rule out
benefits from joint retraining, nonlinear readouts, other measurement bases, or
better state encodings. There is no generalization or completion evaluation here.

[Protocol](results/20261109_readout_probe/protocol.json),
[per-seed results and coefficients](results/20261109_readout_probe/results.json),
[verification](results/20261109_readout_probe/verification.json), and
[static plot](results/20261109_readout_probe/comparison.png) are saved separately.
Two tests pass: independent PennyLane Z/ZZ expectations and autograd stationarity
of the explicit pixel/edge objective at the least-squares solution. Baseline
rendering agrees with V10 within 5.97e-8 RGB; baseline metrics match the independently
verified V10 metrics. The saved circuit/code checkpoints were not modified.

**Priority 1: preserve neighbourhood structure before quantum encoding.**
The current weighted summary requires the retrieval stage to compress local
structure immediately into four values. A more substantial alternative passes
an ordered patch of features to the shared decoder. One concrete implementation
unfolds a 3x3 neighbourhood into channels at each scale, then bilinearly samples
that expanded feature grid at the query. This gives 36 values per scale, 108
across scales, plus the existing coordinate features. Border handling must be
declared. This preserves separate neighbour slots while remaining continuous
under bilinear sampling and requiring one circuit evaluation per pixel query.
The stored code remains 336 values; the shared amplitude/angle network grows.
Match that extra shared capacity with a wider-attention control.

A separate classical factor is a small shared residual convolution before local
retrieval. Its filters can learn reusable neighbourhood patterns across images.
Do not add unfolding, convolutions, finer grids and wider codes all at once and
attribute a gain to one factor. A finer grid is a storage-budget experiment
unless values are reallocated from other scales. Local feature conditioning and
relative-coordinate decoding have precedent in
[LIIF](https://arxiv.org/abs/2012.09161) and its
[implementation](https://github.com/yinboc/liif/blob/main/models/liif.py).
Our first probe would fit local codes directly, not use LIIF's image encoder.
LIIF-style averaging of separately decoded neighbours can multiply circuit
queries; the proposed pre-circuit patch gather does not do that.

**Priority 2: make coordinates enter the quantum computation directly.**
The conservative test keeps initial amplitudes and adds coordinate information
to the later feature-controlled rotations. The more substantial alternative
replaces arbitrary amplitude preparation with a short sequence of learned angle
encodings: initialize a simple state, rotate using local features and position,
mix qubits, then repeat. Local and global coordinate features should have
declared spatial-frequency scales. This makes spatial variation directly affect
the gates and allows feature/position interaction repeatedly inside the circuit.

Repeated data encoding has established expressivity results in
[data re-uploading](https://arxiv.org/abs/1907.02085) and
[Fourier analysis of quantum models](https://arxiv.org/abs/2008.08605).
These motivate the experiment; they do not prove that the present nonlinear
amplitude MLP cannot represent high frequencies. It already receives sinusoidal
coordinate features. V10 did not test a full coordinate-aware angle-loading
architecture: it added two feature-only rotation layers to amplitude encoding.

Start with a declared real RY encoding and keep complex RY/RZ processing as a
separate ablation. RZ introduces relative phases, but phases must encounter later
mixing gates to affect a Z readout; adding final diagonal gates alone would have
no effect. A complex implementation must use absolute-squared amplitudes, not
the current real-state square, and independently verify outputs/gradients.
Allowing signed normalized input amplitudes is another distinct small ablation;
neither signs nor complex phases automatically improve trainability or accuracy.

Angle loading also gives an explicitly countable preparation circuit instead
of treating arbitrary state initialization as a free primitive. This is a
resource-accounting motivation, not a promised speedup: record actual preparation,
two-qubit gate, repeated-upload and measurement costs. Changing the loader also
changes the classical network, so retain classical capacity controls and report
both components. Five qubits are sufficient for the initial comparison.

**Output choices: modest readout extension versus a different decoder design.**
For a QINR, the modest extension is a learned 15-to-3 affine readout of Z/ZZ
statistics, initialized to reproduce the old RGB rule. All 15 statistics can be
estimated from the same computational-basis shot strings. They are not 15
independent noiseless measurements: report finite-shot variance of the final
weighted RGB estimate, and distinguish joint correlations from products of
marginal means. Correlations alone are not evidence of entanglement. The probe
above lowers the priority of a readout-only experiment, but the 18-parameter
five-Z control remains essential if testing it jointly with a new circuit.

A more radical output design is for the quantum module to generate a coarse
spatial feature map, then use a small classical convolutional decoder to produce
RGB. For example, query an 8x8 quantum feature grid and upsample to 32x32 with
shared convolutions. The grid count is 64 quantum queries rather than 1,024 per
image, before accounting for preparation/shot costs; this is an architectural
count, not a measured runtime improvement. Neighbouring outputs can then be
processed together, which the current per-pixel quantum readout does not do.
This does not require DINO or a large pretrained decoder. With a fixed-resolution
convolutional RGB output it is a hybrid image decoder rather than the same
continuous QINR task; retaining a continuous local field after the feature grid
is a separate variant. A retrained classical feature module with the identical
convolutional decoder is mandatory to identify the quantum module's contribution.

**Sequence subsequently approved and completed as V11.**
V11 compared attention versus patch conditioning and amplitude versus
coordinate-aware angle encoding as separate factors and in combination. It kept
the original RGB rule for that input/circuit comparison, with the established
edge objective, paired seeds/data and fixed code storage/qubits. Shared weights,
gates and coordinate queries were recorded separately. It then tested the quantum
feature grid and convolutional RGB output. Retrained processing-removal and
classical controls, training reconstruction and excluded-image completion were
included. Completion remains optimization-based until Project A's encoder is
added. Both pixel/edge metrics and fixed galleries were examined. Other readout
or encoding ablations described above remain proposals.

The prospective research claim is a more useful spatially conditioned QINR at
declared resources, with evidence identifying which quantum/classical components
help. Local conditioning, re-uploading and learned observables already exist;
combining their names is not itself a novelty or quantum-advantage claim.

### Post-V11: quantum feasibility, spatial representation and scale

User clarification, 2026-11-10: the objective is a useful quantum INR, with a
more suitable representation than vanilla QVF and a credible path beyond toy
datasets and resolutions. Quantum simulation is acceptable. Outperforming a
classical image model is not a prerequisite for this objective. The role of
classical controls is to explain where gains come from and what the circuit
contributes. The user subsequently named this work [Project Z](../qvf_project_z/README.md)
and authorized its initial steps. Z has completed its initial 21 architecture
runs, 27 Z4 spatial-encoding/five-versus-seven-qubit pilot runs, and 27 Z5 runs
on 4,744 images with 50,000 updates, with explicit preparation/finite-shot/noise
audits. The five-qubit attention/shared-chain head remains the main candidate.
Z5 shows much stronger fitting with longer training and a smaller spatial
completion gap. Explicit spatial-slot loading still loses quality, while using
fewer logical CNOTs; depth, routing and classical compute also matter. Two extra
work qubits give small mean quality changes, without a clear benefit over the
wider classical five-qubit control. Latent convolution previously helped completion
edges but not overall fitting. The sequence below remains the historical design
rationale; higher-resolution studies remain future work. All current Z runs use
32x32 images and ideal simulation for training, with development-only evaluation.
See Z for completed protocols, results and limitations.

The main architecture should continue to map an image representation and an
arbitrary coordinate to RGB through quantum measurements. V11's fixed 32x32
convolutional output is a secondary engineering branch. It would need a declared
continuous query mechanism to become an INR; interpolating its final raster
would not establish learned higher-resolution detail.

The candidate representation is a pyramid of small classical feature grids,
with shared local processing and a shared small quantum field. A coordinate
retrieves nearby information at several scales; a small classical map prepares
the circuit input; the circuit produces the queried RGB through measurements.
Convolutions act on latent features before the quantum field. No classical RGB
skip bypasses it. The stored grids are classical memory, not an image stored
coherently in a few qubits. Reusing a small register across queries decouples its
width from image resolution; vanilla QVF already reuses a circuit this way, so
this property alone is not a new contribution.

**First: measure quantum execution requirements on existing checkpoints.**

Use the attention/amplitude reference and the existing angle-input field to
establish a resource and quality baseline before choosing more circuit variants.
Decompose amplitude preparation and processing into explicit one- and two-qubit
gates on a declared connectivity. Include routing, input-dependent preparation,
depth and measurement costs. A direct simulator state assignment or a cached
unitary matrix is not a physical preparation/execution schedule. General state
preparation has substantial gate cost; an explicit construction is given by
[Mottonen et al.](https://arxiv.org/abs/quant-ph/0407010).

Evaluate saved fields under finite shots (candidate budgets 256, 1,024 and 4,096
per coordinate, with repeated measurement seeds) and a declared gate/readout
noise model. The grouped Z measurements can share computational-basis samples;
derive RGB variance from those joint samples rather than assuming independent
wire estimates. Label synthetic noise as such; a hardware-calibrated model
requires its calibration source/date. Report the cost of a complete image, not
only one coordinate. Account for noise in state preparation as well as processing.

Large-scale simulator backpropagation and hardware-compatible training are
separate claims. Check parameter-shift gradients on a bounded subset and account
for shifts through both trainable processing and data-dependent input gates;
code/MLP training also needs those derivatives. Shared parameters and controlled
rotations need their applicable gradient rules, not an assumed universal two
evaluations per stored parameter. See
[Wierichs et al.](https://arxiv.org/abs/2107.12390). A finite-shot inference result
does not establish feasible full hardware training.

**Second: improve local representation while preserving the working field.**

The first controlled change is a small shared residual 3x3 convolutional block
on each latent grid before the existing attention retrieval. Keep amplitude
loading, circuit, RGB readout, five qubits, 336 fitted values/image, sampler and
objective fixed. Initialize the residual to zero to preserve the reference
function at the start; report all added shared classical parameters. Include a
capacity control using pointwise processing or wider attention, and paired seeds.
This differs from V11's patch concatenation, which also replaced attention with
bilinear retrieval. The hypothesis is that learned neighbour interaction helps;
it is not an established explanation of the remaining blur.

For larger feature grids, replace all-token attention with a bounded local
lookup tested as its own factor. Use continuous interpolation or smoothly
overlapping neighbourhoods and verify boundaries; hard window changes can
introduce seams. A fixed-size local input keeps per-query classical and quantum
work bounded at fixed scales/channels/depth. Precomputing feature grids still
costs work and memory proportional to their size. Local continuous image
functions have precedent in [LIIF](https://arxiv.org/abs/2012.09161); using local
features by itself is not a novelty claim.

**Third: choose circuit changes using the measured resource budget.**

Retain the amplitude field as a quality reference and test a compact, explicit
rotation-based preparation with shallow local entangling blocks and repeated
feature/coordinate inputs. Hold local representation fixed to isolate this
factor. Compare quality against actual preparation, two-qubit gates, shots and
noise, not qubits alone. A nearest-neighbour layout is a candidate, not an assumed
quality improvement. V11's angle loading lost quality under its recipe; an
angle-only design must earn its place through a useful quality/cost tradeoff.
Keeping amplitudes and adding later coordinate-conditioned rotations is another
candidate if the preparation audit supports it. No circuit choice is locked yet.

If per-coordinate measurement cost dominates, evaluate a separate continuous
field built from cached quantum outputs. One concrete candidate measures local
coefficients at grid cells and combines them with fixed, smoothly overlapping
spatial basis functions at any requested coordinate. A small learned continuous
query head is another variant, with its classical capacity reported separately.
All image-dependent features must pass through the quantum module in either
case. This changes the quantum role from predicting every queried RGB directly
to producing the features or coefficients of a continuous field; label the
distinction and retain the direct quantum RGB field as a reference.

Such reuse could reduce measurement count at a given feature-grid density, but
the cached features are classical and contain finite information. Finer detail
may require more coefficients, cells or shots. Check higher-resolution accuracy
against real withheld targets, code/cache storage, shot correlations across
rendered pixels, and boundary continuity. Simply resizing V11's output raster
does not establish this result. The candidate is unimplemented and no quality or
quantum resource improvement is assumed.

**Fourth: demonstrate scale on separate dataset and resolution axes.**

Use a short paired pilot to reject clear failures, then a longer/convergence
check on the viable field and matched QVF references. Scale image count first
within the dog class, then to tens of thousands of CIFAR training images across
classes (for example 45,000 training/5,000 development). Preserve excluded-image
protocols and reserve the official test set for a final evaluation. Do not reuse
old completion images for fitting merely because the dataset is expanded.

Test spatial resolution separately on native higher-resolution photographs,
using 64/128/256-size targets derived from those sources with a declared sampling
or downsampling operator. Upsampling CIFAR is not a higher-detail ground truth.
Split by original image before taking crops. Start with fixed qubits and explicit
code budgets; subsequently measure any quality/storage tradeoff from finer grids.
Match training compute within each comparison and report image visits and
convergence when comparing dataset sizes. Fixed update counts alone do not give
equal optimization coverage as datasets grow.

Report reconstruction, edges, held-out coordinates, and excluded-image code
fitting, together with code bytes/image, shared parameters, preparation/gates,
depth, circuit queries, total shots, simulator time and memory. Arbitrary-query
support is not evidence of super-resolution accuracy without withheld targets.
The direct field still needs a circuit evaluation per requested coordinate;
measurement work grows with rendered pixels and shots even at fixed qubits.
Do not claim constant-cost image generation or a hardware speedup from batched
GPU simulation. Sample coordinates during training and render in chunks.

An improvement over vanilla QVF must include a global-code control matched in
per-image storage, plus the established local attention reference. Report both
the original QVF comparison and the resource-matched one. Processing-removal and
small classical controls identify component contributions; entanglement-specific
claims need an entangler-only control. A useful quantum INR result can stand on
better quality, fitting or robustness at declared quantum resources without
winning a general classical-model competition.

The intended contribution is an experimentally supported combination of spatial
representation, circuit execution budget and scaling evidence. Recent
[3D-QISR](https://arxiv.org/abs/2601.05250v2) already investigates task-specific
quantum representations and reduced state-preparation complexity, so novelty
must be checked against QVF/QIREN and newer QINR work before framing a paper.
After the decoder is validated, train a small encoder to predict its spatial
codes for unseen images and measure inference time/quality. That resumes Project
A; diffusion follows reconstruction and inference validation.
