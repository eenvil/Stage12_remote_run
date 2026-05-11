from __future__ import annotations

"""
stage12_single_relight.py

Single-image adapter for Stage4.

Place this file here:

    D:\relighting\Stage4\external\Stage12_remote_run\stage12_single_relight.py

It wraps the existing Stage12 batch worker:

    stage12_stream_worker_fast.py

by:
  1. creating a temporary one-image tar.gz shard
  2. running stage12_stream_worker_fast.py with --limit 1
  3. finding the generated scene folder
  4. copying the outputs Stage4 needs into:
       --scene-out
       --stage2-out

Expected output contract for Stage4:

    scene_out/
      albedo.png
      roughness.png
      metallic.png
      camera.json         if available
      original image refs if available

    stage2_out/
      render_optimized.png
      composite_optimized.png

Important:
- Stage 2 still optimizes source-image lighting first, then renders the final
  Stage4 buffers with --env-rgb when the worker exposes --target-env-rgb.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Iterable, List, Optional


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TEXTURE_EXTS = {".png", ".jpg", ".jpeg", ".exr", ".hdr"}


def log(*args) -> None:
    print("[stage12_single]", *args, flush=True)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def copy_if_exists(src: Optional[Path], dst: Path) -> bool:
    if src is None or not src.exists():
        return False
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return True


def find_first(root: Path, patterns: Iterable[str]) -> Optional[Path]:
    if not root.exists():
        return None
    for pat in patterns:
        hits = sorted(root.rglob(pat))
        if hits:
            return hits[0]
    return None


def find_largest_image(root: Path, name_hints: List[str]) -> Optional[Path]:
    """
    Find an image whose name contains one of the hints.
    Prefer larger files because final renders are usually larger than thumbnails.
    """
    if not root.exists():
        return None

    candidates: List[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in TEXTURE_EXTS:
            continue
        low = p.name.lower()
        if any(h in low for h in name_hints):
            candidates.append(p)

    if not candidates:
        return None

    candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
    return candidates[0]


def find_latest_scene_dir(output_root: Path, before_time: float) -> Optional[Path]:
    """
    Find most likely scene output folder from Stage12 worker.
    Usually:
        output/scene_000001
    """
    if not output_root.exists():
        return None

    dirs = [p for p in output_root.rglob("*") if p.is_dir() and p.name.startswith("scene_")]
    if not dirs:
        dirs = [p for p in output_root.iterdir() if p.is_dir()]

    # Prefer dirs modified after this script started.
    recent = [p for p in dirs if p.stat().st_mtime >= before_time - 5]
    search_dirs = recent if recent else dirs
    if not search_dirs:
        return None

    search_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return search_dirs[0]


def make_single_image_tar(input_image: Path, tar_path: Path, scene_id: str) -> None:
    ensure_dir(tar_path.parent)
    suffix = input_image.suffix.lower()
    if suffix not in IMAGE_EXTS:
        raise ValueError(f"Unsupported image extension: {input_image}")

    arcname = f"{scene_id}{suffix}"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(input_image, arcname=arcname)

    log("created temp shard:", tar_path)


def stage12_python_cmd(repo_root: Path) -> List[str]:
    venv_python = repo_root / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return [str(venv_python)]

    uv = shutil.which("uv")
    if uv:
        return [uv, "run", "python"]

    return [sys.executable]


def run_stage12_worker(
    repo_root: Path,
    tar_path: Path,
    args: argparse.Namespace,
    worker_extra_args: List[str],
) -> None:
    worker = repo_root / "stage12_stream_worker_fast.py"
    if not worker.exists():
        raise FileNotFoundError(f"Missing worker: {worker}")

    cmd = [
        *stage12_python_cmd(repo_root),
        str(worker),
        "--work-root", str(tar_path.parent / "worker_work"),
        "--logs-root", str(tar_path.parent / "worker_logs"),
        "--mesh-stride", str(args.mesh_stride),
        "--device", args.device,
        "--mitsuba-variant", args.mitsuba_variant,
        "--spp", str(args.spp),
        "--final-spp", str(args.final_spp),
        "--iters", str(args.iters),
        "--limit", "1",
        "--overwrite",
        "--keep-stage1-assets",
        "--target-env-rgb", *map(str, args.env_rgb),
        "--tarballs", str(tar_path),
    ]
    if args.point_light_position is not None:
        cmd.extend(["--target-point-position", *map(str, args.point_light_position)])
    if args.point_light_rgb is not None:
        cmd.extend(["--target-point-rgb", *map(str, args.point_light_rgb)])

    worker_light_mode = stage4_light_mode_to_worker(args.light_mode)
    if worker_light_mode is not None:
        cmd.extend(["--light-mode", worker_light_mode])

    if args.save_debug_renders:
        cmd.append("--save-debug-renders")
    if args.keep_temp:
        cmd.append("--keep-temp")

    cmd.extend(worker_extra_args)

    log("running Stage12 worker")
    log("cmd:", " ".join(cmd))
    subprocess.run(cmd, cwd=repo_root, check=True)


def stage4_light_mode_to_worker(light_mode: str) -> Optional[str]:
    """
    Stage4 exposes a target-light API; Stage12's worker exposes source-light
    reconstruction modes. Keep the richer source-light model by default and use
    --target-env-rgb for the actual requested relight.
    """
    normalized = light_mode.lower()
    if normalized == "env":
        return "envmap_points"
    if normalized in {"envmap_points", "constant"}:
        return normalized
    log("WARNING: unsupported light_mode for Stage12 source optimization; using envmap_points:", light_mode)
    return "envmap_points"


def copy_material_outputs(found_scene: Path, scene_out: Path, input_image: Path) -> None:
    """
    Copy/normalize scene outputs needed by Stage4.

    Tries many likely names because Stage12 output names evolved during development.
    """
    ensure_dir(scene_out)

    # Original source copy helps glass-aware debugging.
    copy_if_exists(input_image, scene_out / "source.png")

    # Camera
    cam = find_first(found_scene, ["camera.json"])
    if cam:
        copy_if_exists(cam, scene_out / "camera.json")

    # Meshes/assets: useful for debugging/future render reuse.
    for pat in ["*.ply", "*.glb", "*.obj", "*.mtl", "*.xml"]:
        for p in sorted(found_scene.rglob(pat)):
            rel = p.relative_to(found_scene)
            dst = scene_out / rel
            copy_if_exists(p, dst)

    # Material buffers.
    albedo = find_largest_image(found_scene, ["albedo", "basecolor", "base_color", "diffuse"])
    roughness = find_largest_image(found_scene, ["roughness", "rough"])
    metallic = find_largest_image(found_scene, ["metallic", "metalness", "metal"])

    if not copy_if_exists(albedo, scene_out / "albedo.png"):
        log("WARNING: could not find albedo image in", found_scene)

    if not copy_if_exists(roughness, scene_out / "roughness.png"):
        log("WARNING: could not find roughness image in", found_scene)
        # Stage3 requires a path. Use black fallback if missing.
        make_gray_fallback(scene_out / "roughness.png", like=scene_out / "albedo.png", value=128)

    if not copy_if_exists(metallic, scene_out / "metallic.png"):
        log("WARNING: could not find metallic image in", found_scene)
        # Most indoor objects are non-metal; black is a sane fallback.
        make_gray_fallback(scene_out / "metallic.png", like=scene_out / "albedo.png", value=0)


def make_gray_fallback(out_path: Path, like: Path, value: int = 0) -> None:
    from PIL import Image

    ensure_dir(out_path.parent)

    if like.exists():
        with Image.open(like) as img:
            size = img.size
    else:
        size = (512, 512)

    Image.new("L", size, int(value)).save(out_path)


def copy_stage2_outputs(found_scene: Path, stage2_out: Path) -> None:
    """
    Copy/normalize Stage2 render outputs.

    Required:
        render_optimized.png
        composite_optimized.png
    """
    ensure_dir(stage2_out)

    render = find_largest_image(found_scene, [
        "render_optimized",
        "optimized_render",
        "render_optimize",
        "render",
        "mitsuba",
        "final_render",
    ])

    composite = find_largest_image(found_scene, [
        "composite_optimized",
        "optimized_composite",
        "composite",
        "final_composite",
    ])

    # Avoid accidentally choosing the same file when possible.
    if composite is not None and render is not None and composite == render:
        alternatives = []
        for p in found_scene.rglob("*"):
            if p.is_file() and p.suffix.lower() in TEXTURE_EXTS:
                low = p.name.lower()
                if "composite" in low and p != render:
                    alternatives.append(p)
        if alternatives:
            alternatives.sort(key=lambda p: p.stat().st_size, reverse=True)
            composite = alternatives[0]

    if not copy_if_exists(render, stage2_out / "render_optimized.png"):
        raise FileNotFoundError(f"Could not find render output under {found_scene}")

    if not copy_if_exists(composite, stage2_out / "composite_optimized.png"):
        log("WARNING: could not find composite output; falling back to render as composite.")
        shutil.copy2(stage2_out / "render_optimized.png", stage2_out / "composite_optimized.png")


def write_manifest(
    manifest_path: Path,
    args: argparse.Namespace,
    found_scene: Path,
    scene_out: Path,
    stage2_out: Path,
) -> None:
    manifest = {
        "input_image": str(args.input_image),
        "scene_id": args.scene_id,
        "found_stage12_scene": str(found_scene),
        "scene_out": str(scene_out),
        "stage2_out": str(stage2_out),
        "light_mode": args.light_mode,
        "env_rgb": args.env_rgb,
        "point_light_position": args.point_light_position,
        "point_light_rgb": args.point_light_rgb,
        "note": (
            "Stage2 optimizes source-image lighting, then render_optimized and "
            "composite_optimized are rendered with this target env_rgb and optional "
            "point light."
        ),
    }
    ensure_dir(manifest_path.parent)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()

    # Contract expected by Stage4 wrapper.
    ap.add_argument("--input-image", required=True, type=Path)
    ap.add_argument("--scene-id", required=True)
    ap.add_argument("--scene-out", required=True, type=Path)
    ap.add_argument("--stage2-out", required=True, type=Path)
    ap.add_argument("--light-mode", default="env")
    ap.add_argument("--env-rgb", nargs=3, type=float, default=[1.0, 1.0, 1.0])
    ap.add_argument("--point-light-position", nargs=3, type=float, default=None)
    ap.add_argument("--point-light-rgb", nargs=3, type=float, default=None)
    ap.add_argument("--spp", type=int, default=32)
    ap.add_argument("--final-spp", type=int, default=128)

    # Existing Stage12 knobs.
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--mesh-stride", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mitsuba-variant", default="cuda_ad_rgb")
    ap.add_argument("--output-root", type=Path, default=None, help="Stage12 worker output root. Default: repo/output")
    ap.add_argument("--temp-root", type=Path, default=None)
    ap.add_argument("--save-debug-renders", action="store_true")
    ap.add_argument("--keep-temp", action="store_true")

    # Pass-through args if needed.
    ap.add_argument(
        "--worker-extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args appended to stage12_stream_worker_fast.py after --",
    )

    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent
    input_image = args.input_image.resolve()
    scene_out = args.scene_out.resolve()
    stage2_out = args.stage2_out.resolve()
    output_root = (args.output_root or (repo_root / "output")).resolve()

    if not input_image.exists():
        raise FileNotFoundError(input_image)

    if args.light_mode != "env":
        log("WARNING: Stage4 target light mode is expected to be env; received:", args.light_mode)

    log("scene_id:", args.scene_id)
    log("input_image:", input_image)
    log("env_rgb:", args.env_rgb)
    log("output_root:", output_root)

    start_time = time.time()

    if args.temp_root:
        temp_root = ensure_dir(args.temp_root.resolve())
        tar_path = temp_root / f"{args.scene_id}.tar.gz"
        make_single_image_tar(input_image, tar_path, args.scene_id)
        cleanup_temp = False
    else:
        tmpdir = tempfile.TemporaryDirectory(prefix="stage12_single_")
        temp_root = Path(tmpdir.name)
        tar_path = temp_root / f"{args.scene_id}.tar.gz"
        make_single_image_tar(input_image, tar_path, args.scene_id)
        cleanup_temp = True

    try:
        # Remove previous normalized outputs.
        if scene_out.exists():
            shutil.rmtree(scene_out)
        if stage2_out.exists():
            shutil.rmtree(stage2_out)
        ensure_dir(scene_out)
        ensure_dir(stage2_out)

        # Run existing worker.
        worker_extra = list(args.worker_extra_args)
        if worker_extra and worker_extra[0] == "--":
            worker_extra = worker_extra[1:]

        run_stage12_worker(repo_root, tar_path, args, worker_extra)

        found_scene = find_latest_scene_dir(output_root, start_time)
        if found_scene is None:
            raise FileNotFoundError(f"Could not find Stage12 scene dir under {output_root}")

        log("found Stage12 scene:", found_scene)

        copy_material_outputs(found_scene, scene_out, input_image)
        copy_stage2_outputs(found_scene, stage2_out)
        write_manifest(scene_out / "stage12_single_manifest.json", args, found_scene, scene_out, stage2_out)

        log("DONE")
        log("scene_out:", scene_out)
        log("stage2_out:", stage2_out)

    finally:
        if cleanup_temp:
            try:
                tmpdir.cleanup()  # type: ignore[name-defined]
            except Exception:
                pass


if __name__ == "__main__":
    main()
