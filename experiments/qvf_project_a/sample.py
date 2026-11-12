"""Generate images from a saved, self-contained Project A diffusion checkpoint."""

import argparse

from .train import configure_runtime, generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, help="New directory; existing samples are not overwritten.")
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=["cpu", "cuda:0"], default="cuda:0")
    args = parser.parse_args()
    device = configure_runtime(args.device, args.seed)
    generate(args.checkpoint, args.output, device, args.count, args.seed)
    print(f"Samples saved to {args.output}")


if __name__ == "__main__":
    main()
