"""Fetch a pinned, published DINOv2-B / ViT-XL RAE (no diffusion weights)."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


DEFAULT_ASSETS = Path.home() / ".cache/quantai/rae"
SOURCE_URL = "https://github.com/bytetriper/RAE.git"
SOURCE_REVISION = "a4d18c4db766419cbe7cb8c02cd9f7ceb0ec9041"
ENCODER_ID = "facebook/dinov2-with-registers-base"
ENCODER_REVISION = "a1d738ccfa7ae170945f210395d99dde8adb1805"
COLLECTION_ID = "nyu-visionx/RAE-collections"
COLLECTION_REVISION = "1be4f03273523431f099a934da4cf1940dc6039f"
DECODER_FILE = "decoders/dinov2/wReg_base/ViTXL_n08/model.pt"
STATS_FILE = "stats/dinov2/wReg_base/imagenet1k/stat.pt"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare(root):
    # Install dependencies in the separate RAE venv, never into quantai.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import HfApi, hf_hub_download

    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    source = root / "official"
    if not source.exists():
        subprocess.run(["git", "clone", "--no-checkout", SOURCE_URL, str(source)], check=True)
        subprocess.run(["git", "-C", str(source), "checkout", "--detach", SOURCE_REVISION], check=True)
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != SOURCE_REVISION:
        raise ValueError("Existing RAE checkout differs from the pinned revision; use another assets directory.")
    subprocess.run(["git", "-C", str(source), "diff", "--exit-code", "HEAD"], check=True)
    manifest = {"source_url": SOURCE_URL, "source_revision": revision, "files": {}, "source_files": {}}
    for repo, rev, subdir, names in (
        (ENCODER_ID, ENCODER_REVISION, "encoder", ["config.json", "preprocessor_config.json", "model.safetensors"]),
        (COLLECTION_ID, COLLECTION_REVISION, "models", [DECODER_FILE, STATS_FILE]),
    ):
        info = HfApi().model_info(repo, revision=rev, files_metadata=True)
        metadata = {item.rfilename: item for item in info.siblings}
        for name in names:
            print(f"Downloading {repo}@{rev[:8]}/{name}", flush=True)
            path = Path(hf_hub_download(repo, name, revision=rev, local_dir=root / subdir))
            digest = sha256(path)
            upstream = metadata[name].lfs
            if upstream is not None and digest != upstream.sha256:
                raise ValueError(f"Upstream LFS checksum mismatch: {name}")
            manifest["files"][str(path.relative_to(root))] = {
                "repo": repo, "revision": rev, "filename": name,
                "sha256": digest, "bytes": path.stat().st_size,
                "upstream_lfs_sha256": upstream.sha256 if upstream else None,
            }
    for relative in ("src/stage1", "configs/decoder/ViTXL", "configs/stage1/pretrained"):
        for path in sorted((source / relative).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".json", ".yaml"}:
                manifest["source_files"][str(path.relative_to(source))] = sha256(path)
    write_json(root / "manifest.json", manifest)
    print(f"Verified assets: {root / 'manifest.json'}", flush=True)
    return manifest


def verify_assets(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["source_revision"] != SOURCE_REVISION:
        raise ValueError("Unexpected source revision")
    for name, item in manifest["files"].items():
        if sha256(root / name) != item["sha256"]:
            raise ValueError(f"Asset changed: {name}")
    for name, expected in manifest["source_files"].items():
        if sha256(root / "official" / name) != expected:
            raise ValueError(f"Upstream source changed: {name}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    prepare(parser.parse_args().assets)
