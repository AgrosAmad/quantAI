# Z4 method: spatial slots and shared quantum filters

This is Project Z's representation study, before an image encoder, VAE or
diffusion. Training results belong to the frozen protocol and verified report
in `results/20261110_z4_spatial/`; this document explains the experiment.

## What enters the circuit

Every image still has 336 fitted classical values: three grids of sizes 2x2,
4x4 and 8x8, each with four feature channels. At a requested row/column coordinate,
the spatial model samples the centre and four cardinal neighbours from EACH
grid. A neighbour is one grid cell away at its scale, so coarse slots cover
broader context than fine slots. Bilinear interpolation and border extension
define the samples at arbitrary coordinates.

The five slot roles are fixed:

```
                    wire 1: above

wire 0: left        wire 2: centre        wire 4: right

                    wire 3: below
```

The features are learned latent features, not raw pixel colours. Each slot has
12 sampled values, the centre query's 26 positional features, and two fixed
numbers indicating the slot's spatial role. A shared classical MLP takes those
40 values and returns five bounded rotation angles. It is evaluated separately
for each slot: it never concatenates different neighbours before the circuit.
The MLP has three hidden layers of width 128, ReLU activations, and an output
bounded by `pi * tanh`. There are also three shared four-to-four value projections,
one per scale. These parts are classical and trainable.

Thus the actual pipeline is:

```
fitted image grids + requested coordinate
  -> five multiscale feature slots
  -> the SAME small MLP applied separately to each slot
  -> five angle uploads per spatial qubit, interleaved with quantum processing
  -> measurements of five qubits
  -> RGB for the requested coordinate
```

A qubit is not literally a container holding twelve readable real numbers.
Its assigned slot controls rotations on that wire. After entangling operations,
information can be distributed across the joint state.

## Shared processing and the seven-qubit condition

Five data wires start in `|+>`. In each of five stages, one data-dependent RY
rotation is applied to each data wire. Then a shared two-qubit filter is applied
between centre wire 2 and each other wire in ascending order. The filter is:

1. RY on the first wire and RY on the second.
2. Controlled RY from first to second and controlled RY in the reverse direction.
3. The same four angles are reused for every pair within that stage.

Each stage ends with one learned RY per wire. Stages have different filter
parameters; all image coordinates and images share the same circuit parameters.
There is no pooling, reset or intermediate measurement. Order matters, and the
absolute positional features mean this is not an exactly rotation- or
translation-equivariant convolution.

At seven qubits, wires 5 and 6 start in `|0>` and interact with the centre through
the same filter. They receive no direct data upload. The five spatial inputs,
classical encoder and RGB readout stay fixed. This tests extra quantum workspace
without also enlarging the image neighbourhood.

The RGB rule at BOTH counts is `red = Z0 + Z3`, `green = Z1 + Z4`, `blue = Z2`.
Measurements are expectations or joint finite-shot estimates. There is no
classical RGB prediction bypass. Original QVF's seven-qubit normalization would
change this rule; Z4 deliberately holds it fixed.

Two terminal RY rotations on the unmeasured work wires have no effect on RGB.
The executed seven-qubit circuit retains them, so its counts are conservative.
The frozen training metadata field `effective_quantum_parameters` actually
counts stored entries: 45 at five qubits and 55 at seven. The independent report
corrects this to 45/53 output-relevant entries and explicitly records the two
redundant terminal entries. This is a metadata-label correction; training
sources, parameters and predictions are preserved.

## What the controls isolate

| Comparison | What changes |
| --- | --- |
| Attention + chain, 5 vs 7 | Larger quantum register, more gates and larger amplitude MLP output |
| Attention + star vs attention + chain | Logical circuit interaction graph, at fixed input representation |
| Spatial slots + star vs centre-only slots + star | Access to neighbouring latent features, with identical parameters and circuit |
| Spatial slots, 5 vs 7 | Two extra work wires, same five input slots and same classical encoder |
| Wider attention + chain, 5 vs standard attention + chain, 7 | Classical parameter-count control for amplitude-output expansion |

The attention branch is the existing multiscale attention retrieval followed
by a classical MLP producing positive, normalized amplitudes. Its chain variant
at five qubits exactly reproduces the completed Z3 shared local head.

All conditions use 336 values/image, paired initial codes, data/query streams,
10,000 updates, the pixel + 0.5 edge objective, and three seeds. They are not
matched in FLOPs, gates or wall time. In particular the slot MLP is evaluated
five times per coordinate; the attention amplitude MLP is evaluated once.
The width-148 five-qubit control has 56,336 classical parameters, compared with
56,220 for width-128 seven-qubit attention. The spatial models both have 38,977
classical parameters; five-qubit width-128 attention has 43,836.

Completion freezes the field and fits new codes from 256 observed pixels of
each of 32 excluded development images. It scores the 768 withheld pixels and
edges with both endpoints withheld. This is code-inference generalization,
not automatic image encoding. These development images have been used in
earlier architecture studies; they are not a fresh final test.

## Quantum cost and the scaling hypothesis

RY/CNOT counts below include explicit input preparation and are per coordinate.
Shared parameters reduce storage, not the number of gate applications.

| Input and head | Qubits | RY primitives | CNOT primitives |
| --- | ---: | ---: | ---: |
| Attention/amplitude + chain or star | 5 | 176 | 110 |
| Spatial/centre slots + star | 5 | 175 | 80 |
| Attention/amplitude + chain or star | 7 | 342 | 246 |
| Spatial/centre slots + star | 7 | 245 | 120 |

The positive-real amplitude loader uses 31 RY/30 CNOT at five qubits and
127 RY/126 CNOT at seven. The slot loader uses five fixed preparation rotations
and 25 data rotations, with no entangling preparation gates at either size.
This is the concrete representation-cost motivation for the spatial branch.
It does not guarantee better reconstruction or a useful image model at scale.

The star graph is logical. An all-to-all gate count is not a native device
execution claim. The audit also constructs an explicit, unoptimized line route
with SWAPs and restored wire order; those routing counts are not lower bounds.
Synthetic noise acts after every RY/CNOT including preparation, with a joint
two-wire Pauli channel after CNOT and independent readout flips. Only four
development images/seed0 are used for the gate-noise diagnostic. Finite-shot
evaluation uses all 32 completion images and three training seeds.

There is still one circuit query per coordinate. A full 32x32 image at 1,024
shots per coordinate takes 1,048,576 circuit executions. Larger images increase
that cost even with a fixed register. Larger latent grids also require classical
storage. The current bilinear sampler builds dense interpolation weights; a
sparse four-neighbour sampler is a future implementation optimization, not an
achieved speed result here. Simulation, hardware inference and hardware training
costs must remain distinct.

The intended research question is whether explicit spatial inputs and shared
quantum processing can offer a useful quality/cost tradeoff for a continuous
quantum INR. This experiment establishes neither a novel generic convolution
primitive nor quantum advantage over classical image models.

## Related mechanisms

- [Quantum convolutional neural networks](https://arxiv.org/abs/1810.03787)
  include convolution and pooling on quantum states; Z4 does not implement
  that full architecture.
- [Quanvolutional neural networks](https://arxiv.org/abs/1904.04767) apply quantum
  processing to local classical image regions; Z4 instead predicts a continuous
  RGB field from learned multiscale latent features with trained shared circuits.
- [Data re-uploading](https://arxiv.org/abs/1907.02085) motivates repeated
  data-dependent rotations interleaved with trained operations. Re-uploading
  alone is established prior work, not this project's novelty claim.
