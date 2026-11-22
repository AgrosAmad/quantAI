# Pilot: does the quantum part matter, and neighbours in superposition

Small Colab pilot by Eldana, run while waiting for server access.
It is a simplified, self-contained copy of the Project Z `spatial_reference`
model at 32×32. No repository source files were changed, and nothing here
replaces the Z protocol. It is a first signal, not a paper result.

## Questions

1. Does the circuit in the current spatial model do anything a small classical
   head could not, and does entanglement matter?
2. A new idea: instead of one data wire per spatial slot, use **5 content
   qubits + 3 address qubits**. The address is in superposition over the 8
   neighbours of a 3×3 window, so one circuit run loads all of them. Learned,
   content-controlled rotations on the address mix the branches between
   rounds, so neighbours can reinforce or cancel each other (interference).
   This is the spatial version of Z7's coherent scale routing. Qubits grow as
   `5 + ceil(log2(neighbours))`: 3×3 → 8, 5×5 → 10, 7×7 → 11.

## Models

All models share the same front end: 336-value notes grids (2×2, 4×4, 8×8,
4 channels), bilinear reads, a shared slot MLP, and the RGB readout
Z0+Z3, Z1+Z4, Z2.

| Model | Qubits | What changes |
|---|---|---|
| `quantum` | 7 | Current spatial model: 5 slots on 5 data wires + 2 work wires, star CRY block |
| `no_entanglement` | 7 | CRY gates removed (product state) |
| `classical_twin` | 0 | Circuit replaced by an 8-wide ReLU MLP on the same 25 angles (507 params) |
| `classical_twin_fixed` | 0 | Same size, LeakyReLU (the ReLU twin collapsed in seed 1) |
| `superposition` | 8 | 5 content + 3 address qubits, 3×3 window, coherent branch mixing |
| `superposition_no_interference` | 8 | Same 8 branches, no mixing; 14-parameter classical softmax over branch colours at the end |
| `superposition_dephased` | 8 | Same circuit, same parameters and same mixing during the circuit, but the address is dephased before and after every mixing step: branches exchange weight classically and cannot interfere |

`superposition` vs `superposition_dephased` is the cleanest test of whether
quantum interference itself contributes. Both start from identical parameters
for a given seed.

## Protocol

CIFAR-10 dogs, 1,024 training images, 10,000 updates, batch 16 images × 128
pixels (64 neighbouring pairs), Adam lr 1e-3 for model and codes, loss =
pixel MSE + 0.5 × edge MSE + 1e-4 × code norm. Completion: frozen decoder,
fresh codes fit to 256 visible pixels of 32 unseen test dogs for 1,000 steps,
scored on the 768 hidden pixels (mean per-image PSNR, masked edge MSE).
Three seeds (0, 1, 2).

## Results (completion PSNR, dB, higher is better)

| Model | Seed 0 | Seed 1 | Seed 2 | Mean |
|---|---:|---:|---:|---:|
| quantum | 20.76 | 21.01 | 21.04 | 20.94 |
| no_entanglement | 20.66 | 20.06 | 20.59 | 20.44 |
| classical_twin | 20.90 | 12.19* | 20.91 | 20.90 (seeds 0, 2) |
| classical_twin_fixed | not run | not run | 20.69 | — |
| superposition | 21.15 | 21.16 | 20.66 | 20.99 |
| superposition_no_interference | 20.76 | 20.95 | 20.74 | 20.82 |
| superposition_dephased | not run | not run | not finished | — |

\* Training collapsed (loss flat for all 10k steps, dying ReLU in the 8-wide
head). Not a valid score. `classical_twin_fixed` (LeakyReLU, same size) was
added for this reason, but only seed 2 finished.

Paired differences (same seed):

| Comparison | Seed 0 | Seed 1 | Seed 2 | Mean |
|---|---:|---:|---:|---:|
| no_entanglement − quantum | −0.10 | −0.95 | −0.45 | **−0.50** |
| classical_twin − quantum (valid seeds) | +0.14 | — | −0.14 | ≈ 0 |
| superposition − no_interference | +0.39 | +0.21 | −0.08 | +0.17 |
| superposition − quantum | +0.39 | +0.15 | −0.38 | +0.05 |

What we can and cannot say:

1. **Removing entanglement hurts in all three seeds** (mean −0.50 dB, and
   higher edge error). Within this circuit, the entangling gates matter.
2. **A tiny classical head matches the circuit** (+0.14 and −0.14 dB in the
   two valid seeds). So the circuit is useful compared with a product-state
   circuit, but not compared with a small MLP of similar size.
3. **The superposition gain is not robust.** It was ahead of its
   no-interference control in seeds 0 and 1, but slightly behind in seed 2.
   Mean +0.17 dB with one sign flip is not enough to claim interference helps.
   The dephased control, which is the cleanest test, did not finish because of
   free GPU limits.

Per-run numbers are in `pilot_results.csv`; each row points to the executed
notebook in `runs/` it came from.

## Status

- Seeds 0 and 1 were run with an earlier version of the notebook. The code of
  the five models in those runs is identical to the current notebook; later
  versions only added `classical_twin_fixed`, `superposition_dephased`, a
  `RUN` switch and extra printing.
- Seed 2 ran with the current notebook. The run stopped (GPU quota) during
  `superposition_dephased`, after the other six models had finished.
- Still missing: `superposition_dephased` (all seeds) and
  `classical_twin_fixed` (seeds 0 and 1). The dephased model takes about
  1.5–2 hours per seed on a T4, so it fits better on the server.

## Files

| File | What |
|---|---|
| `quantum_check_pilot.ipynb` | Current notebook (all 7 models, `RUN` switch) |
| `verify_circuits_numpy.py` | Independent NumPy check of every circuit |
| `pilot_results.csv` | Every completed run, one row per model and seed |
| `runs/seed{0,1,2}_executed.ipynb` | The executed notebooks the numbers came from |

## Verification

`verify_circuits_numpy.py` (NumPy only) compares every circuit's fast
simulation with an explicit simulation built from full gate matrices: the 7q
models, the coherent 8q model (256×256 state) and the dephased model (256×256
density matrix with explicit address dephasing). All agree to about 1e-15. It
also checks that the dephased and coherent models are identical when mixing is
off, and differ when it is on. The notebook repeats the last check in PyTorch.

```bash
python verify_circuits_numpy.py      # expected last line: ALL CHECKS PASSED
```

## How to run

Open `quantum_check_pilot.ipynb` in Colab or Kaggle with a T4 GPU.

1. Run once with `QUICK_TEST = True` (a few minutes) to check everything works
   and see the time per model.
2. Set `QUICK_TEST = False`.
   - `RUN = 1`: seed 2 for all seven models.
   - `RUN = 2`: seeds 0 and 1 for `classical_twin_fixed` and `superposition_dephased`.
3. Download the executed notebook; results are printed in it.

## Caveats

- Pilot scale: 32px, 1,024 images, 10k steps. Differences of 0.2–0.4 dB need
  all three seeds before drawing conclusions.
- `superposition` sees 9 spots and `quantum` sees 5, so that comparison mixes
  two changes. Use the no-interference and dephased controls instead.
- Saving is in qubits, not gates: loading more neighbours still costs more
  gates per round.
- Exact simulation only. No hardware, shot noise or advantage claim.

## Possible next steps

1. Finish the missing runs (`superposition_dephased` for three seeds,
   `classical_twin_fixed` for seeds 0 and 1), ideally with more seeds, since
   the effects are a few tenths of a dB.
2. If superposition still beats both controls, port it into `qvf_project_z`
   with the full Z protocol (4,744 images, 50k updates, 32px and 256px,
   verification and quantum audits).
3. The no-entanglement result could go straight into the "what does the
   quantum part do" discussion, together with the classical-twin tie.
