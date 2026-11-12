"""Train the official PennyLane model and the fast model side by side from the same start.

Usage:
    python check_training.py /path/to/official/src/Images/QVF_for_image.py

Both models get the same weights, the same latent codes and the same full batch (the first 20
images of the demo data), and are trained with the official loss and two Adam optimisers for
30 steps on the CPU. The loss curves should agree to float32 rounding. The official circuit
uses the corrected rotation indexing (see check_equivalence.py).
"""

import json
import sys
from pathlib import Path

import torch

from check_equivalence import load_official, make_fixed_layer
from qvf_fast import QVFImage, positional_embedding
from train import PRESETS, load_images, make_coords, model_inputs

HERE = Path(__file__).parent


def main():
    official = load_official(sys.argv[1])
    official.QuantumLayer = make_fixed_layer(official)
    device = torch.device("cpu")
    steps, n_img = 30, 20
    report = []
    for identity in (True, False):
        cfg = dict(PRESETS["demo"], use_identity_block=identity, train_samples_use=n_img)
        torch.manual_seed(0)
        images = load_images(cfg, HERE / "data")
        pixels = images.reshape(n_img, -1, 3)
        coords = make_coords(32, 32)

        ref = official.ImageRegressor(cfg, device)
        mine = QVFImage(cfg)
        mine.load_state_dict(ref.state_dict(), strict=True)
        z_ref = official.generate_latent_codes(n_img, cfg["latent_size"], device)
        z_mine = z_ref.detach().clone().requires_grad_(True)

        opt_ref = [torch.optim.Adam(ref.parameters(), lr=cfg["lr_model"]), torch.optim.Adam([z_ref], lr=cfg["lr_latent"])]
        opt_mine = [torch.optim.Adam(mine.parameters(), lr=cfg["lr_model"]), torch.optim.Adam([z_mine], lr=cfg["lr_latent"])]
        sigma = cfg["latent_reg"] ** 0.5
        y = pixels.reshape(-1, 3)
        losses = []
        for step in range(steps):
            # official pipeline: official positional embedding, input assembly and loss
            coord = official.positional_embedding(coords.unsqueeze(0).repeat(n_img, 1, 1), cfg["L_embed"], device)
            z = z_ref.unsqueeze(1).expand(-1, coord.shape[1], -1)
            x = torch.cat([z, coord], dim=-1).view(-1, z.shape[-1] + coord.shape[-1])
            for o in opt_ref:
                o.zero_grad()
            loss_ref, _, _ = official.mse_latent_loss(y, ref(x).float(), x[:, :cfg["latent_size"]], sigma)
            loss_ref.backward()
            for o in opt_ref:
                o.step()

            # this repository's pipeline
            x2 = model_inputs(z_mine, positional_embedding(coords, cfg["L_embed"], scale=cfg["pe_scale"]))
            for o in opt_mine:
                o.zero_grad()
            pred = mine(x2)
            loss_mine = torch.nn.functional.mse_loss(pred, y) + cfg["latent_reg"] * torch.mean(
                torch.linalg.norm(x2[:, :cfg["latent_size"]], dim=1))
            loss_mine.backward()
            for o in opt_mine:
                o.step()
            losses.append((loss_ref.item(), loss_mine.item()))

        rel = [abs(a - b) / abs(a) for a, b in losses]
        report.append({"identity_block": identity, "steps": steps, "images": n_img,
                       "first_loss_official": losses[0][0], "first_loss_fast": losses[0][1],
                       "last_loss_official": losses[-1][0], "last_loss_fast": losses[-1][1],
                       "max_relative_loss_difference": max(rel)})
        print(f"identity block {identity}: loss {losses[0][0]:.6f} -> {losses[-1][0]:.6f} (official), "
              f"{losses[0][1]:.6f} -> {losses[-1][1]:.6f} (fast), max relative difference {max(rel):.2e}", flush=True)
    out = HERE / "results" / "training_equivalence.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
