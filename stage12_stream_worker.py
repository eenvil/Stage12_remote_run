#!/usr/bin/env python3
"""
Stage12 streaming production worker
===================================

Streaming design:

  for each image in downloads/ or extracted tarballs:
      1. run PixelPerfect/MoGe depth for this one image
      2. run MVInverse in-process with model loaded once
      3. build one temporary Stage-2 package under work/current/<stem>/stage_pkg
      4. run Stage 2 immediately
      5. keep only compact final output
      6. delete temp package and PixelPerfect temporary outputs

This avoids the bad two-phase behavior:

  Stage 1 all images -> many huge OBJ files -> Stage 2 later

Instead, the huge UV OBJ exists only for the current image.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ---------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def mkdirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def safe_rmtree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def safe_unlink(path: Path) -> None:
    try:
        if path.exists() and path.is_file():
            path.unlink()
    except Exception as e:
        print(f"[WARN] could not remove {path}: {e}")


def shlex_quote(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"


def run_cmd(cmd: list[str | Path], *, cwd: Path, log_path: Path) -> None:
    cmd_str = " ".join(str(x) for x in cmd)
    header = f"\n[{utc_now()}] > {cmd_str}\n{'-' * 90}\n"
    print(header, end="")

    with log_path.open("a", encoding="utf-8") as log:
        log.write(header)
        log.flush()

        env = os.environ.copy()
        # Windows console safety. On Linux this is harmless.
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")

        proc = subprocess.Popen(
            [str(x) for x in cmd],
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)

        ret = proc.wait()
        footer = f"\n[{utc_now()}] exit={ret}\n"
        print(footer, end="")
        log.write(footer)

    if ret != 0:
        raise RuntimeError(f"Command failed with exit code {ret}: {cmd_str}")


def find_images(paths: Iterable[Path]) -> list[Path]:
    images: list[Path] = []
    for base in paths:
        if base.is_file() and base.suffix.lower() in IMG_EXTS:
            images.append(base)
        elif base.is_dir():
            for p in base.rglob("*"):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    images.append(p)
    return sorted(set(images), key=lambda p: str(p).lower())


def extract_tarball(tar_path: Path, extract_root: Path, log_path: Path) -> Path:
    stem = tar_path.name
    for suffix in [".tar.gz", ".tgz", ".tar"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break

    out_dir = extract_root / stem
    done_marker = out_dir / ".extract_done"
    if done_marker.exists():
        print(f"[SKIP] already extracted: {tar_path} -> {out_dir}")
        return out_dir

    safe_rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[EXTRACT] {tar_path} -> {out_dir}")
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{utc_now()}] extracting {tar_path} -> {out_dir}\n")

    mode = "r:gz" if tar_path.name.endswith((".tar.gz", ".tgz")) else "r:"
    with tarfile.open(tar_path, mode) as tar:
        root_resolved = out_dir.resolve()
        for member in tar.getmembers():
            target = (out_dir / member.name).resolve()
            if not str(target).startswith(str(root_resolved)):
                raise RuntimeError(f"Unsafe tar member path: {member.name}")
        tar.extractall(out_dir)

    done_marker.write_text(utc_now(), encoding="utf-8")
    return out_dir


def download_if_needed(url: str | None, dst: Path, repo_root: Path, log_path: Path) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        print(f"[OK] checkpoint exists: {dst}")
        return
    if not url:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[DOWNLOAD] {url} -> {dst}")
    run_cmd(
        ["bash", "-lc", f"wget -c -O {shlex_quote(str(dst))} {shlex_quote(url)}"],
        cwd=repo_root,
        log_path=log_path,
    )


def ensure_checkpoints(args: argparse.Namespace, repo_root: Path, log_path: Path) -> None:
    ckpt_dir = repo_root / "checkpoints"
    mkdirs(ckpt_dir)

    download_if_needed(args.ppd_moge_url, ckpt_dir / "moge2.pt", repo_root, log_path)
    download_if_needed(args.ppd_model_url, ckpt_dir / "ppd_moge.pth", repo_root, log_path)
    download_if_needed(args.ppd_da2_url, ckpt_dir / "depth_anything_v2_vitl.pth", repo_root, log_path)
    download_if_needed(args.ppd_da2_model_url, ckpt_dir / "ppd.pth", repo_root, log_path)


# ---------------------------------------------------------------------
# PixelPerfect runner
# ---------------------------------------------------------------------

def run_pixelperfect_for_image(
    *,
    repo_root: Path,
    image: Path,
    args: argparse.Namespace,
    log_path: Path,
) -> dict[str, Path]:
    """
    Calls root-level run_point_cloud.py.

    Temporary root-level outputs created by run_point_cloud.py:
      depth_mesh/<stem>.ply
      depth_pcd/<stem>_camera.json
      background/<stem>_*.png/json

    They are copied into work/current/<stem>/stage_pkg and deleted after success.
    """
    stem = image.stem
    script = repo_root / "run_point_cloud.py"
    if not script.exists():
        raise FileNotFoundError(f"Missing {script}")

    cmd: list[str | Path] = [
        sys.executable,
        script,
        "--img_path", image,
        "--outdir", repo_root / "depth_vis",
        "--semantics_model", args.semantics_model,
        "--sampling_steps", str(args.sampling_steps),
        "--save_pcd",
        "--save_mesh",
        "--mesh_stride", str(args.mesh_stride),
        "--max_depth_jump", str(args.max_depth_jump),
        "--relative_depth_jump", str(args.relative_depth_jump),
        "--max_geometry_depth", str(args.max_geometry_depth),
        "--background_percentile", str(args.background_percentile),
        "--save_background",
        "--no_depth_vis",
    ]

    run_cmd(cmd, cwd=repo_root, log_path=log_path)

    outputs = {
        "mesh_ply": repo_root / "depth_mesh" / f"{stem}.ply",
        "camera": repo_root / "depth_pcd" / f"{stem}_camera.json",
        "pcd": repo_root / "depth_pcd" / f"{stem}.ply",
        "background": repo_root / "background" / f"{stem}_background.png",
        "geometry_mask": repo_root / "background" / f"{stem}_geometry_mask.png",
        "background_mask": repo_root / "background" / f"{stem}_background_mask.png",
        "background_meta": repo_root / "background" / f"{stem}_background_meta.json",
    }

    if not outputs["mesh_ply"].exists():
        raise FileNotFoundError(outputs["mesh_ply"])
    if not outputs["camera"].exists():
        raise FileNotFoundError(outputs["camera"])
    if not outputs["geometry_mask"].exists():
        raise FileNotFoundError(outputs["geometry_mask"])

    return outputs


def cleanup_pixelperfect_temp(repo_root: Path, stem: str) -> None:
    for p in [
        repo_root / "depth_mesh" / f"{stem}.ply",
        repo_root / "depth_pcd" / f"{stem}.ply",
        repo_root / "depth_pcd" / f"{stem}_camera.json",
        repo_root / "background" / f"{stem}_background.png",
        repo_root / "background" / f"{stem}_geometry_mask.png",
        repo_root / "background" / f"{stem}_background_mask.png",
        repo_root / "background" / f"{stem}_background_meta.json",
        repo_root / "depth_vis" / f"{stem}.png",
    ]:
        safe_unlink(p)


# ---------------------------------------------------------------------
# MVInverse in-process runner
# ---------------------------------------------------------------------

def load_mvinverse_once(repo_root: Path, ckpt: str, device: str):
    """
    Imports root-level single_image_batch_inference.py and uses its loader.
    """
    sys.path.insert(0, str(repo_root))
    mod = importlib.import_module("single_image_batch_inference")
    print("[MVI] loading model once")
    model = mod.load_model(ckpt, __import__("torch").device(device))
    return mod, model


def run_mvinverse_one(
    *,
    mvi_mod,
    mvi_model,
    image: Path,
    device: str,
    out_dir: Path,
) -> dict[str, Path]:
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    torch_device = torch.device(device)
    img = mvi_mod.load_single_image(image).unsqueeze(0).to(torch_device)

    use_cuda_amp = torch_device.type == "cuda"
    with torch.no_grad():
        if use_cuda_amp:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                res = mvi_model(img[None])
        else:
            res = mvi_model(img[None])

    def save_rgb_tensor(t, path: Path) -> None:
        arr = (t.detach().float().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(arr).save(path)

    albedo = res["albedo"][0, 0]
    metallic = res["metallic"][0, 0]
    roughness = res["roughness"][0, 0]
    normal = res["normal"][0, 0]
    shading = res["shading"][0, 0]

    paths = {
        "albedo": out_dir / "albedo.png",
        "metallic": out_dir / "metallic.png",
        "roughness": out_dir / "roughness.png",
        "normal": out_dir / "normal.png",
        "irradiance": out_dir / "irradiance.png",
    }

    save_rgb_tensor(albedo, paths["albedo"])
    Image.fromarray((metallic.squeeze(-1).detach().float().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)).save(paths["metallic"])
    Image.fromarray((roughness.squeeze(-1).detach().float().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)).save(paths["roughness"])
    save_rgb_tensor(normal * 0.5 + 0.5, paths["normal"])
    save_rgb_tensor(shading, paths["irradiance"])

    return paths


# ---------------------------------------------------------------------
# Build temporary Stage 2 package
# ---------------------------------------------------------------------

def build_temp_stage2_package(
    *,
    image: Path,
    ppd: dict[str, Path],
    maps: dict[str, Path],
    pkg_dir: Path,
) -> Path:
    """
    Create the only temporary package Mitsuba needs.

    Required:
      scene_uv.obj
      camera.json
      reference.png
      geometry_mask.png
      albedo.png
      roughness.png
      metallic.png
    """
    import trimesh

    stem = image.stem
    safe_rmtree(pkg_dir)
    pkg_dir.mkdir(parents=True, exist_ok=True)

    # Copy camera/masks/reference/material maps.
    shutil.copy2(ppd["camera"], pkg_dir / "camera.json")
    Image.open(image).convert("RGB").save(pkg_dir / "reference.png")

    shutil.copy2(ppd["geometry_mask"], pkg_dir / "geometry_mask.png")
    if ppd.get("background") and ppd["background"].exists():
        shutil.copy2(ppd["background"], pkg_dir / "background.png")
    if ppd.get("background_mask") and ppd["background_mask"].exists():
        shutil.copy2(ppd["background_mask"], pkg_dir / "background_mask.png")
    if ppd.get("background_meta") and ppd["background_meta"].exists():
        shutil.copy2(ppd["background_meta"], pkg_dir / "background_meta.json")

    for key, src in maps.items():
        if src.exists():
            shutil.copy2(src, pkg_dir / src.name)

    # Load mesh and project UVs from camera metadata.
    mesh_obj = ppd["mesh_ply"]
    scene_or_mesh = trimesh.load(str(mesh_obj), process=False)
    if isinstance(scene_or_mesh, trimesh.Scene):
        geoms = list(scene_or_mesh.geometry.values())
        if not geoms:
            raise RuntimeError(f"No geometry in {mesh_obj}")
        mesh = trimesh.util.concatenate(geoms)
    else:
        mesh = scene_or_mesh

    with open(ppd["camera"], "r", encoding="utf-8") as f:
        meta = json.load(f)

    K = np.asarray(meta["intrinsic"], dtype=np.float32)
    resize_W = int(meta["resize_W"])
    resize_H = int(meta["resize_H"])
    flip = np.asarray(meta.get("saved_ply_flip", [1.0, -1.0, -1.0]), dtype=np.float32)

    verts_saved = np.asarray(mesh.vertices, dtype=np.float32)
    verts_cam = verts_saved / flip[None, :]

    x = verts_cam[:, 0]
    y = verts_cam[:, 1]
    z = verts_cam[:, 2]
    valid = z > 1e-6

    u = K[0, 0] * x / z + K[0, 2]
    v = K[1, 1] * y / z + K[1, 2]

    inside = (
        valid
        & (u >= 0) & (u <= resize_W - 1)
        & (v >= 0) & (v <= resize_H - 1)
    )

    px = np.clip(np.round(u).astype(np.int32), 0, resize_W - 1)
    py = np.clip(np.round(v).astype(np.int32), 0, resize_H - 1)

    uv = np.zeros((len(verts_saved), 2), dtype=np.float32)
    uv[:, 0] = np.clip(u / max(1, resize_W - 1), 0.0, 1.0)
    uv[:, 1] = np.clip(1.0 - (v / max(1, resize_H - 1)), 0.0, 1.0)

    # Better vertex colors and vertex normals for fallback/debug.
    albedo_path = pkg_dir / "albedo.png"
    if albedo_path.exists():
        alb = np.asarray(Image.open(albedo_path).convert("RGB").resize((resize_W, resize_H), Image.BILINEAR), dtype=np.uint8)
        vc = np.zeros((len(verts_saved), 3), dtype=np.uint8)
        vc[inside] = alb[py[inside], px[inside]]
        try:
            old_vc = np.asarray(mesh.visual.vertex_colors)
            if old_vc is not None and len(old_vc) == len(verts_saved):
                vc[~inside] = old_vc[~inside, :3]
        except Exception:
            pass
        alpha = np.full((len(vc), 1), 255, dtype=np.uint8)
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=np.concatenate([vc, alpha], axis=1))

    normal_path = pkg_dir / "normal.png"
    if normal_path.exists():
        nmap = np.asarray(Image.open(normal_path).convert("RGB").resize((resize_W, resize_H), Image.BILINEAR), dtype=np.float32) / 255.0
        sampled = nmap[py[inside], px[inside]] * 2.0 - 1.0
        normals = np.zeros((len(verts_saved), 3), dtype=np.float32)
        normals[inside] = sampled
        try:
            geom_normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
            if geom_normals.shape == normals.shape:
                normals[~inside] = geom_normals[~inside]
        except Exception:
            pass
        normals *= flip[None, :]
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        bad = lengths[:, 0] < 1e-8
        normals[bad] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        lengths[bad] = 1.0
        normals = normals / lengths
        try:
            mesh.vertex_normals = normals
        except Exception:
            pass
        try:
            mesh._cache["vertex_normals"] = normals
        except Exception:
            pass

    # Export UV OBJ for Mitsuba bitmap albedo/roughness/metallic.
    if albedo_path.exists():
        tex_img = Image.open(albedo_path).convert("RGB")
        tex_mesh = mesh.copy()
        tex_mesh.visual = trimesh.visual.TextureVisuals(
            uv=uv,
            image=tex_img,
            material=trimesh.visual.material.SimpleMaterial(
                image=tex_img,
                diffuse=[255, 255, 255, 255],
            ),
        )
        tex_mesh.export(str(pkg_dir / "scene_uv.obj"))
    else:
        # Fallback: no UV texture, but Stage 2 can use PLY vertex color.
        mesh.export(str(pkg_dir / "scene.ply"))

    # Always keep a PLY fallback in temp only. It is deleted after Stage 2.
    try:
        mesh.export(str(pkg_dir / "scene.ply"))
    except Exception:
        pass

    manifest = {
        "schema": "stage12_temp_package_v1",
        "name": stem,
        "temporary": True,
        "mesh": "scene_uv.obj" if (pkg_dir / "scene_uv.obj").exists() else "scene.ply",
        "has_uv": (pkg_dir / "scene_uv.obj").exists(),
        "camera_metadata": "camera.json",
        "reference_image": "reference.png",
        "geometry_mask": "geometry_mask.png",
        "sidecar_maps": {
            "albedo": "albedo.png" if (pkg_dir / "albedo.png").exists() else None,
            "roughness": "roughness.png" if (pkg_dir / "roughness.png").exists() else None,
            "metallic": "metallic.png" if (pkg_dir / "metallic.png").exists() else None,
            "normal": "normal.png" if (pkg_dir / "normal.png").exists() else None,
            "irradiance": "irradiance.png" if (pkg_dir / "irradiance.png").exists() else None,
        },
    }
    (pkg_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return pkg_dir


# ---------------------------------------------------------------------
# Stage 2 runner and final-output compaction
# ---------------------------------------------------------------------

def run_stage2(
    *,
    repo_root: Path,
    pkg_dir: Path,
    args: argparse.Namespace,
    log_path: Path,
) -> Path:
    out_dir = repo_root / "output" / pkg_dir.name
    if out_dir.exists() and args.overwrite:
        safe_rmtree(out_dir)

    cmd: list[str | Path] = [
        sys.executable,
        repo_root / "stage2" / "main.py",
        "--input-root", pkg_dir.parent,
        "--scene", pkg_dir.name,
        "--output-root", repo_root / "output",
        "--variant", args.mitsuba_variant,
        "--spp", str(args.spp),
        "--final-spp", str(args.final_spp),
        "--iters", str(args.iters),
        "--lr", str(args.lr),
        "--material-mode", args.material_mode,
        "--material-model", args.material_model,
        "--normal-mode", args.normal_mode,
        "--light-mode", args.light_mode,
        "--envmap-width", str(args.envmap_width),
        "--envmap-height", str(args.envmap_height),
        "--point-grid-x", str(args.point_grid_x),
        "--point-grid-y", str(args.point_grid_y),
        "--point-layers", str(args.point_layers),
        "--point-init", str(args.point_init),
        "--light-reg", str(args.light_reg),
        "--no-copy-input-assets",
    ]

    if args.overwrite:
        cmd.append("--overwrite")
    if args.save_debug_renders:
        cmd.append("--save-debug-renders")
    if args.save_exr:
        cmd.append("--save-exr")
    if args.save_pairs:
        cmd.append("--save-pairs")
    if args.keep_stage2_work:
        cmd.append("--keep-work")

    run_cmd(cmd, cwd=repo_root, log_path=log_path)

    manifest = out_dir / "manifest.json"
    if not manifest.exists():
        raise RuntimeError(f"Stage 2 did not produce manifest: {manifest}")

    compact_final_output(out_dir, keep_render_backend=args.keep_render_backend, keep_stage1_assets=args.keep_stage1_assets)
    return out_dir


def compact_final_output(out_dir: Path, *, keep_render_backend: bool, keep_stage1_assets: bool) -> None:
    """
    Stage 2 currently writes/copies some production-unwanted files.
    Remove them here, after manifest exists.
    """
    removed = []

    if not keep_render_backend:
        for name in ["scene_render_backend.obj", "scene_render_backend.ply"]:
            p = out_dir / name
            if p.exists():
                p.unlink()
                removed.append(name)

    if not keep_stage1_assets:
        for name in [
            "camera.json",
            "reference.png",
            "geometry_mask.png",
            "background.png",
            "background_mask.png",
            "background_meta.json",
            "albedo.png",
            "normal.png",
            "roughness.png",
            "metallic.png",
            "irradiance.png",
            "scene.glb",
        ]:
            p = out_dir / name
            if p.exists():
                p.unlink()
                removed.append(name)

    # Patch manifest so it does not claim the huge backend is kept.
    manifest_path = out_dir / "manifest.json"
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not keep_render_backend:
            data["mitsuba_geometry_backend"] = None
            data.setdefault("production_cleanup", {})["render_backend_removed"] = True
        if not keep_stage1_assets:
            data.setdefault("production_cleanup", {})["stage1_assets_removed"] = True
        if removed:
            data.setdefault("production_cleanup", {})["removed_files"] = removed
        manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[WARN] Could not patch final manifest: {e}")

    if removed:
        print(f"[OK] compacted final output, removed {len(removed)} file(s)")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Stage12 streaming production worker")

    p.add_argument("--repo-root", type=Path, default=Path(__file__).parent.resolve())
    p.add_argument("--downloads-root", type=Path, default=None)
    p.add_argument("--work-root", type=Path, default=None)
    p.add_argument("--logs-root", type=Path, default=None)

    p.add_argument("--images", type=Path, nargs="*", default=None, help="Explicit image paths. If omitted, images/tarballs are read from downloads/.")
    p.add_argument("--tarballs", type=Path, nargs="*", default=None, help="Explicit tarballs. If omitted, *.tar/*.tar.gz under downloads/ are used.")

    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="Process only first N images; 0 = all.")

    # Checkpoint URLs.
    p.add_argument("--ppd-moge-url", type=str, default=None)
    p.add_argument("--ppd-model-url", type=str, default=None)
    p.add_argument("--ppd-da2-url", type=str, default=None)
    p.add_argument("--ppd-da2-model-url", type=str, default=None)

    # PixelPerfect.
    p.add_argument("--semantics-model", choices=["MoGe2", "DA2"], default="MoGe2")
    p.add_argument("--sampling-steps", type=int, default=20)
    p.add_argument("--mesh-stride", type=int, default=1)
    p.add_argument("--max-depth-jump", type=float, default=0.08)
    p.add_argument("--relative-depth-jump", type=float, default=0.08)
    p.add_argument("--max-geometry-depth", type=float, default=100.0)
    p.add_argument("--background-percentile", type=float, default=0.98)

    # MVInverse.
    p.add_argument("--mvi-ckpt", type=str, default="maddog241/mvinverse")
    p.add_argument("--device", type=str, default="cuda")

    # Stage 2.
    p.add_argument("--mitsuba-variant", type=str, default="cuda_ad_rgb")
    p.add_argument("--spp", type=int, default=32)
    p.add_argument("--final-spp", type=int, default=128)
    p.add_argument("--iters", type=int, default=80)
    p.add_argument("--lr", type=float, default=0.08)
    p.add_argument("--material-mode", choices=["auto", "texture", "vertex_color", "mean_albedo", "gray"], default="auto")
    p.add_argument("--material-model", choices=["principled", "diffuse"], default="principled")
    p.add_argument("--normal-mode", choices=["auto", "none", "tangent_map"], default="auto")
    p.add_argument("--light-mode", choices=["envmap_points", "constant"], default="envmap_points")
    p.add_argument("--envmap-width", type=int, default=16)
    p.add_argument("--envmap-height", type=int, default=8)
    p.add_argument("--point-grid-x", type=int, default=4)
    p.add_argument("--point-grid-y", type=int, default=3)
    p.add_argument("--point-layers", type=int, default=1)
    p.add_argument("--point-init", type=float, default=0.01)
    p.add_argument("--light-reg", type=float, default=1e-4)

    # Output policy.
    p.add_argument("--keep-temp", action="store_true", help="Keep work/current/<stem> after success.")
    p.add_argument("--keep-render-backend", action="store_true", help="Keep scene_render_backend.obj/.ply in final output.")
    p.add_argument("--keep-stage1-assets", action="store_true", help="Keep copied Stage 1 maps/reference/masks in final output.")
    p.add_argument("--save-debug-renders", action="store_true")
    p.add_argument("--save-exr", action="store_true")
    p.add_argument("--save-pairs", action="store_true")
    p.add_argument("--keep-stage2-work", action="store_true")

    return p.parse_args()


def collect_inputs(args: argparse.Namespace, repo_root: Path, downloads_root: Path, work_root: Path, log_path: Path) -> list[Path]:
    sources: list[Path] = []

    if args.tarballs:
        tarballs = [p.resolve() for p in args.tarballs]
    else:
        tarballs = sorted(downloads_root.glob("*.tar")) + sorted(downloads_root.glob("*.tar.gz")) + sorted(downloads_root.glob("*.tgz"))

    if tarballs:
        extract_root = work_root / "input"
        mkdirs(extract_root)
        for tar in tarballs:
            sources.append(extract_tarball(tar, extract_root, log_path))

    if args.images:
        sources.extend([p.resolve() for p in args.images])
    else:
        # Also accept loose images directly under downloads/.
        sources.append(downloads_root)

    images = find_images(sources)
    if args.limit and args.limit > 0:
        images = images[: args.limit]
    return images


def main() -> None:
    args = parse_args()

    repo_root = args.repo_root.resolve()
    downloads_root = (args.downloads_root or (repo_root / "downloads")).resolve()
    work_root = (args.work_root or (repo_root / "work")).resolve()
    logs_root = (args.logs_root or (repo_root / "logs")).resolve()

    current_root = work_root / "current"
    log_path = logs_root / "run.log"
    progress_path = logs_root / "progress.jsonl"
    failed_path = logs_root / "failed.jsonl"

    mkdirs(downloads_root, work_root, current_root, logs_root, repo_root / "output")

    ensure_checkpoints(args, repo_root, log_path)

    images = collect_inputs(args, repo_root, downloads_root, work_root, log_path)
    if not images:
        raise RuntimeError(f"No images found. Put images or tarballs under {downloads_root}, or pass --images/--tarballs.")

    print(f"Found {len(images)} image(s).")
    append_jsonl(progress_path, {"time": utc_now(), "event": "worker_start", "count": len(images)})

    mvi_mod, mvi_model = load_mvinverse_once(repo_root, args.mvi_ckpt, args.device)

    ok = 0
    fail = 0
    skip = 0

    for idx, image in enumerate(images, start=1):
        stem = image.stem
        final_manifest = repo_root / "output" / stem / "manifest.json"

        if final_manifest.exists() and not args.overwrite:
            print(f"[SKIP] [{idx}/{len(images)}] {stem}: final manifest exists")
            skip += 1
            append_jsonl(progress_path, {"time": utc_now(), "event": "skip_existing", "image": str(image), "stem": stem})
            continue

        t0 = time.time()
        current_dir = current_root / stem
        pkg_dir = current_dir / stem
        mvi_tmp = current_dir / "mvi_maps"

        append_jsonl(progress_path, {"time": utc_now(), "event": "image_start", "index": idx, "total": len(images), "image": str(image)})

        try:
            print(f"\n{'=' * 100}\n[{idx}/{len(images)}] {image}\n{'=' * 100}")
            safe_rmtree(current_dir)
            mkdirs(current_dir)

            ppd_outputs = run_pixelperfect_for_image(repo_root=repo_root, image=image, args=args, log_path=log_path)

            maps = run_mvinverse_one(
                mvi_mod=mvi_mod,
                mvi_model=mvi_model,
                image=image,
                device=args.device,
                out_dir=mvi_tmp,
            )

            build_temp_stage2_package(image=image, ppd=ppd_outputs, maps=maps, pkg_dir=pkg_dir)

            final_out = run_stage2(repo_root=repo_root, pkg_dir=pkg_dir, args=args, log_path=log_path)

            cleanup_pixelperfect_temp(repo_root, stem)
            if not args.keep_temp:
                safe_rmtree(current_dir)

            dt = time.time() - t0
            ok += 1
            append_jsonl(progress_path, {
                "time": utc_now(),
                "event": "image_success",
                "index": idx,
                "total": len(images),
                "image": str(image),
                "stem": stem,
                "seconds": dt,
                "output": str(final_out),
            })
            print(f"[OK] success {stem} in {dt:.1f}s -> {final_out}")

        except Exception as e:
            dt = time.time() - t0
            fail += 1
            err = {
                "time": utc_now(),
                "event": "image_failed",
                "index": idx,
                "total": len(images),
                "image": str(image),
                "stem": stem,
                "seconds": dt,
                "error": repr(e),
            }
            append_jsonl(failed_path, err)
            append_jsonl(progress_path, err)
            print(f"[ERR] failed {stem}: {e}", file=sys.stderr)

            # Keep current_dir for debugging failed images.
            continue

    summary = {
        "time": utc_now(),
        "event": "worker_done",
        "total": len(images),
        "success": ok,
        "failed": fail,
        "skipped": skip,
    }
    append_jsonl(progress_path, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
