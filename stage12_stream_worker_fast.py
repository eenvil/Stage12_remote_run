#!/usr/bin/env python3
"""
Stage12 streaming production worker
===================================

Streaming design:

  for each image in downloads/ or extracted tarballs:
      1. run MVInverse in-process with model loaded once
      2. run PixelPerfect/MoGe in-process with model loaded once
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
# PixelPerfect / MoGe in-process runner
# ---------------------------------------------------------------------

def load_pixelperfect_once(repo_root: Path, args: argparse.Namespace):
    """Load PixelPerfectDepth and MoGe once, instead of spawning run_point_cloud.py per image."""
    import torch
    from ppd.utils.set_seed import set_seed
    from ppd.moge.model.v2 import MoGeModel
    from ppd.models.ppd import PixelPerfectDepth

    set_seed(666)
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")

    ckpt_dir = repo_root / "checkpoints"
    if args.semantics_model == "MoGe2":
        semantics_pth = ckpt_dir / "moge2.pt"
        model_pth = ckpt_dir / "ppd_moge.pth"
    else:
        semantics_pth = ckpt_dir / "depth_anything_v2_vitl.pth"
        model_pth = ckpt_dir / "ppd.pth"

    print(f"[PPD] loading MoGe once from {ckpt_dir / 'moge2.pt'}")
    moge = MoGeModel.from_pretrained(str(ckpt_dir / "moge2.pt")).to(device).eval()

    print(f"[PPD] loading PixelPerfectDepth once from {model_pth}")
    ppd_model = PixelPerfectDepth(
        semantics_model=args.semantics_model,
        semantics_pth=str(semantics_pth),
        sampling_steps=args.sampling_steps,
    )
    ppd_model.load_state_dict(torch.load(str(model_pth), map_location="cpu"), strict=False)
    ppd_model = ppd_model.to(device).eval()

    return {
        "device": device,
        "moge": moge,
        "ppd_model": ppd_model,
    }


def run_pixelperfect_one_inprocess(
    *,
    ppd_state: dict[str, Any],
    image: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """
    In-process equivalent of run_point_cloud.py for one image.

    Returns an in-memory mesh plus camera/background data. No root-level
    depth_mesh/depth_pcd/background folders are written.
    """
    import cv2
    import open3d as o3d
    import torch
    from ppd.utils.align_depth_func import recover_metric_depth_ransac
    from ppd.utils.depth2pcd import depth2pcd

    device = ppd_state["device"]
    ppd_model = ppd_state["ppd_model"]
    moge = ppd_state["moge"]

    image_bgr = cv2.imread(str(image))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image}")

    print(f"[PPD] in-process depth/mesh: {image.name}")

    with torch.inference_mode():
        depth, resize_image = ppd_model.infer_image(image_bgr)
        depth_np_rel = depth.squeeze().detach().cpu().numpy()

        resize_H, resize_W = resize_image.shape[:2]
        moge_image = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
        moge_tensor = torch.tensor(moge_image / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
        moge_depth, mask, intrinsic = moge.infer(moge_tensor)

        # MoGe versions differ: some return torch tensors, others return NumPy arrays.
        # Normalize to torch tensors here so the original run_point_cloud.py logic works.
        moge_depth = torch.as_tensor(moge_depth, dtype=torch.float32, device=device)
        mask = torch.as_tensor(mask, dtype=torch.bool, device=device)
        intrinsic = torch.as_tensor(intrinsic, dtype=torch.float32, device=device)

        if torch.any(mask):
            moge_depth[~mask] = moge_depth[mask].max()
        else:
            raise RuntimeError(f"MoGe valid mask is empty for {image}")

        metric_depth = recover_metric_depth_ransac(depth_np_rel, moge_depth, mask)

        intrinsic = intrinsic.clone()
        intrinsic[0, 0] *= resize_W
        intrinsic[1, 1] *= resize_H
        intrinsic[0, 2] *= resize_W
        intrinsic[1, 2] *= resize_H

    # Keep pointcloud creation optional-compatible but do not save it.
    # Some upstream functions can force useful synchronization/validation.
    _ = depth2pcd(
        metric_depth,
        intrinsic,
        color=cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB),
        input_mask=mask,
        ret_pcd=True,
    )

    K_full = intrinsic.detach().cpu().numpy() if torch.is_tensor(intrinsic) else np.asarray(intrinsic)
    depth_full = metric_depth.detach().cpu().numpy() if torch.is_tensor(metric_depth) else np.asarray(metric_depth)
    mask_full = mask.detach().cpu().numpy() if torch.is_tensor(mask) else np.asarray(mask)
    rgb_full = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)

    base_mask = mask_full.astype(bool) & np.isfinite(depth_full) & (depth_full > 0)
    valid_depths = depth_full[base_mask]

    if valid_depths.size > 0:
        bg_percentile = min(max(float(args.background_percentile), 0.0), 1.0)
        adaptive_far = float(np.quantile(valid_depths, bg_percentile))
        hard_cap = float(args.max_geometry_depth)
        far_cut = min(hard_cap, adaptive_far) if hard_cap > 0 else adaptive_far
    else:
        adaptive_far = float("nan")
        hard_cap = float(args.max_geometry_depth)
        far_cut = hard_cap if hard_cap > 0 else float("inf")

    geometry_mask_full = base_mask & (depth_full < far_cut)
    background_mask_full = ~geometry_mask_full

    print(
        f"[PPD] far_cut={far_cut:.3f} percentile={float(args.background_percentile):.3f} "
        f"geometry={int(geometry_mask_full.sum())}/{geometry_mask_full.size} "
        f"({100.0 * geometry_mask_full.mean():.1f}%)"
    )

    background_rgb = rgb_full.copy()
    background_rgb[geometry_mask_full] = 255

    # Build decimated grid mesh, matching run_point_cloud.py behavior.
    K_mesh = K_full.copy()
    depth_mesh = depth_full
    geometry_mask_mesh = geometry_mask_full
    rgb_mesh = rgb_full

    stride = max(1, int(args.mesh_stride))
    if stride > 1:
        depth_mesh = depth_mesh[::stride, ::stride]
        geometry_mask_mesh = geometry_mask_mesh[::stride, ::stride]
        rgb_mesh = rgb_mesh[::stride, ::stride]
        K_mesh = K_mesh.copy()
        K_mesh[0, 0] /= stride
        K_mesh[1, 1] /= stride
        K_mesh[0, 2] /= stride
        K_mesh[1, 2] /= stride

    Hm, Wm = depth_mesh.shape
    fx, fy = K_mesh[0, 0], K_mesh[1, 1]
    cx, cy = K_mesh[0, 2], K_mesh[1, 2]

    ys, xs = np.meshgrid(np.arange(Hm), np.arange(Wm), indexing="ij")
    z = depth_mesh.astype(np.float32)
    x = (xs.astype(np.float32) - cx) * z / fx
    y = (ys.astype(np.float32) - cy) * z / fy

    verts = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    verts *= np.array([1, -1, -1], dtype=np.float32)
    colors = rgb_mesh.reshape(-1, 3).astype(np.float64) / 255.0
    valid = geometry_mask_mesh.astype(bool)

    idx = np.arange(Hm * Wm, dtype=np.int32).reshape(Hm, Wm)
    i00, i01 = idx[:-1, :-1], idx[:-1, 1:]
    i10, i11 = idx[1:, :-1], idx[1:, 1:]

    z00, z01 = depth_mesh[:-1, :-1], depth_mesh[:-1, 1:]
    z10, z11 = depth_mesh[1:, :-1], depth_mesh[1:, 1:]
    v00, v01 = valid[:-1, :-1], valid[:-1, 1:]
    v10, v11 = valid[1:, :-1], valid[1:, 1:]

    abs_jump = float(args.max_depth_jump)
    rel_jump = float(args.relative_depth_jump)
    local_z = np.maximum.reduce([z00, z01, z10, z11]).astype(np.float32)
    threshold = np.maximum(abs_jump, rel_jump * local_z)

    tri1_ok = (
        v00 & v01 & v10
        & (np.abs(z00 - z01) < threshold)
        & (np.abs(z00 - z10) < threshold)
        & (np.abs(z01 - z10) < threshold)
    )
    tri2_ok = (
        v01 & v11 & v10
        & (np.abs(z01 - z11) < threshold)
        & (np.abs(z10 - z11) < threshold)
        & (np.abs(z01 - z10) < threshold)
    )

    faces1 = np.stack([i00[tri1_ok], i10[tri1_ok], i01[tri1_ok]], axis=1)
    faces2 = np.stack([i01[tri2_ok], i10[tri2_ok], i11[tri2_ok]], axis=1)
    faces = np.concatenate([faces1, faces2], axis=0).astype(np.int32)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    mesh.compute_vertex_normals()

    print(f"[PPD] mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")

    bg_meta = {
        "resize_W": int(resize_W),
        "resize_H": int(resize_H),
        "background_percentile": float(args.background_percentile),
        "max_geometry_depth": float(args.max_geometry_depth),
        "adaptive_far_depth": None if not np.isfinite(adaptive_far) else float(adaptive_far),
        "far_cut_depth": None if not np.isfinite(far_cut) else float(far_cut),
        "geometry_pixel_ratio": float(geometry_mask_full.mean()),
        "note": "geometry_mask=finite mesh region; background_mask=sky/far/non-geometry region for stage-2 compositing",
    }

    camera = {
        "resize_W": int(resize_W),
        "resize_H": int(resize_H),
        "intrinsic": K_full.tolist(),
        "saved_ply_flip": [1.0, -1.0, -1.0],
        "max_geometry_depth": float(args.max_geometry_depth),
        "background_percentile": float(args.background_percentile),
        "background_files": {
            "background_rgb": "background.png",
            "geometry_mask": "geometry_mask.png",
            "background_mask": "background_mask.png",
            "meta": "background_meta.json",
        },
    }

    # Release transient CUDA tensors, but keep both models resident.
    del depth, moge_depth, mask, intrinsic, moge_tensor, metric_depth
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "mesh_o3d": mesh,
        "camera": camera,
        "background_rgb": background_rgb,
        "geometry_mask": (geometry_mask_full.astype(np.uint8) * 255),
        "background_mask": (background_mask_full.astype(np.uint8) * 255),
        "background_meta": bg_meta,
        "resize_rgb": rgb_full,
    }


# ---------------------------------------------------------------------
# MVInverse in-process runner
# ---------------------------------------------------------------------
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
    ppd: dict[str, Any],
    maps: dict[str, Path],
    pkg_dir: Path,
) -> Path:
    """
    Create the only temporary package Mitsuba needs.

    This fast version consumes the in-memory PixelPerfect result and writes
    only the per-image Stage-2 handoff package. No persistent depth_mesh/
    depth_pcd/background folders are needed.
    """
    import open3d as o3d
    import trimesh

    stem = image.stem
    safe_rmtree(pkg_dir)
    pkg_dir.mkdir(parents=True, exist_ok=True)

    # Camera / reference / masks / background.
    (pkg_dir / "camera.json").write_text(json.dumps(ppd["camera"], indent=2), encoding="utf-8")
    Image.open(image).convert("RGB").save(pkg_dir / "reference.png")
    Image.fromarray(ppd["geometry_mask"]).save(pkg_dir / "geometry_mask.png")
    Image.fromarray(ppd["background_rgb"]).save(pkg_dir / "background.png")
    Image.fromarray(ppd["background_mask"]).save(pkg_dir / "background_mask.png")
    (pkg_dir / "background_meta.json").write_text(json.dumps(ppd["background_meta"], indent=2), encoding="utf-8")

    for _key, src in maps.items():
        if src.exists():
            shutil.copy2(src, pkg_dir / src.name)

    mesh_o3d = ppd["mesh_o3d"]
    verts_saved = np.asarray(mesh_o3d.vertices, dtype=np.float32)
    faces = np.asarray(mesh_o3d.triangles, dtype=np.int64)
    if len(verts_saved) == 0 or len(faces) == 0:
        raise RuntimeError(f"Empty mesh for {image}")

    try:
        vc = np.asarray(mesh_o3d.vertex_colors)
        if vc is not None and len(vc) == len(verts_saved):
            vertex_colors = np.clip(vc * 255.0, 0, 255).astype(np.uint8)
        else:
            vertex_colors = None
    except Exception:
        vertex_colors = None

    mesh = trimesh.Trimesh(
        vertices=verts_saved,
        faces=faces,
        vertex_colors=vertex_colors,
        process=False,
    )

    meta = ppd["camera"]
    K = np.asarray(meta["intrinsic"], dtype=np.float32)
    resize_W = int(meta["resize_W"])
    resize_H = int(meta["resize_H"])
    flip = np.asarray(meta.get("saved_ply_flip", [1.0, -1.0, -1.0]), dtype=np.float32)

    verts_cam = verts_saved / flip[None, :]
    x, y, z = verts_cam[:, 0], verts_cam[:, 1], verts_cam[:, 2]
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

    albedo_path = pkg_dir / "albedo.png"
    if albedo_path.exists():
        alb = np.asarray(Image.open(albedo_path).convert("RGB").resize((resize_W, resize_H), Image.BILINEAR), dtype=np.uint8)
        vc2 = np.zeros((len(verts_saved), 3), dtype=np.uint8)
        vc2[inside] = alb[py[inside], px[inside]]
        if vertex_colors is not None:
            vc2[~inside] = vertex_colors[~inside, :3]
        alpha = np.full((len(vc2), 1), 255, dtype=np.uint8)
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=np.concatenate([vc2, alpha], axis=1))

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
        mesh.export(str(pkg_dir / "scene.ply"))

    # PLY fallback, still temporary and deleted after Stage 2.
    try:
        # Open3D writes a clean PLY quickly and preserves vertex colors.
        o3d.io.write_triangle_mesh(str(pkg_dir / "scene.ply"), mesh_o3d, write_vertex_colors=True)
    except Exception:
        try:
            mesh.export(str(pkg_dir / "scene.ply"))
        except Exception:
            pass

    manifest = {
        "schema": "stage12_temp_package_v2_inprocess_ppd",
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
    ppd_state = load_pixelperfect_once(repo_root, args)

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

            # Run MVInverse first while the input image is still the only large input.
            # Then run PixelPerfect/MoGe in-process with its model already resident.
            maps = run_mvinverse_one(
                mvi_mod=mvi_mod,
                mvi_model=mvi_model,
                image=image,
                device=args.device,
                out_dir=mvi_tmp,
            )

            ppd_outputs = run_pixelperfect_one_inprocess(
                ppd_state=ppd_state,
                image=image,
                args=args,
            )

            build_temp_stage2_package(image=image, ppd=ppd_outputs, maps=maps, pkg_dir=pkg_dir)

            final_out = run_stage2(repo_root=repo_root, pkg_dir=pkg_dir, args=args, log_path=log_path)
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
