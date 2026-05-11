#!/usr/bin/env python3
"""
Stage 2 - Mitsuba constant RGB environment-light estimation (UV-aware)
=====================================================================

Input package layout, produced by Stage 1:
  asset/input/<picture_name>/
    scene_uv.obj             # preferred Mitsuba backend when available
    scene.glb                # canonical Blender-checked asset
    scene.ply                # fallback Mitsuba render backend
    camera.json
    manifest.json
    reference.png
    geometry_mask.png
    background.png           # optional
    background_mask.png      # optional
    albedo.png / roughness.png / metallic.png / normal.png / irradiance.png

Production output layout:
  asset/output/<picture_name>/
    render_optimized.png
    composite_optimized.png
    residual.png
    scene_mitsuba.json
    optimize_config.json
    optimization_log.json
    manifest.json
    scene_render_backend.obj/.ply
    compact copied Stage-1 assets: camera/reference/masks/material maps
    light/initial_light.json
    light/optimized_light.json

Optional debug/training outputs, disabled by default:
  render_initial.png, render_material_preview.png, render EXR files, _work/, encoder_pairs/, renderer_pairs/, scene.glb

This version prefers the Stage-1 UV OBJ backend:
  scene_uv.obj > scene.ply > scene.glb

So if Stage 1 exported image-space UVs, Mitsuba can directly use:
  albedo.png, roughness.png, metallic.png
as real bitmap textures with a principled BSDF.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, None)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


def read_rgb(path: Path, size: tuple[int, int] | None = None, linear: bool = False) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize(size, Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return srgb_to_linear(arr) if linear else arr


def read_mask(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    img = Image.open(path).convert("L")
    if size is not None:
        img = img.resize(size, Image.NEAREST)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return (arr > 0.5).astype(np.float32)[..., None]


def write_png_srgb(path: Path, rgb_srgb: np.ndarray) -> None:
    rgb_srgb = np.clip(rgb_srgb, 0.0, 1.0)
    Image.fromarray((rgb_srgb * 255.0 + 0.5).astype(np.uint8)).save(path)


def copy_if_exists(src: Path, dst: Path) -> str | None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return dst.name
    return None


def load_mesh_with_trimesh(path: Path):
    """Load a mesh from PLY/GLB/OBJ without remeshing."""
    import trimesh

    loaded = trimesh.load(str(path), process=False)
    if isinstance(loaded, trimesh.Scene):
        geoms = list(loaded.geometry.values())
        if not geoms:
            raise RuntimeError(f"No geometry found in {path}")
        if len(geoms) == 1:
            return geoms[0]
        return trimesh.util.concatenate(geoms)
    return loaded


def normalize_vertex_colors_for_mitsuba(mesh):
    import numpy as np
    import trimesh

    try:
        vc = np.asarray(mesh.visual.vertex_colors)
        if vc is not None and len(vc) == len(mesh.vertices):
            vc = vc[:, :3].astype(np.float32)
            if vc.max() > 1.0:
                vc = vc / 255.0
            alpha = np.ones((len(vc), 1), dtype=np.float32)
            mesh.visual = trimesh.visual.ColorVisuals(
                mesh=mesh,
                vertex_colors=np.concatenate([vc, alpha], axis=1),
            )
    except Exception as e:
        print(f"[WARN] Could not normalize vertex colors: {e}")
    return mesh


def prepare_mitsuba_backend_shape(package_dir: Path, work_dir: Path) -> tuple[Path, str, str, bool]:
    """
    Prefer the Stage-1 UV mesh when available.

    Returns:
      backend_path, backend_source, shape_type, has_uv

    Priority:
      scene_uv.obj > scene.ply > scene.glb
    """
    uv_obj = package_dir / "scene_uv.obj"
    ply = package_dir / "scene.ply"
    glb = package_dir / "scene.glb"

    if uv_obj.exists():
        print(f"Using Stage 1 UV OBJ backend source: {uv_obj.name}")
        return uv_obj, "stage1_scene_uv_obj", "obj", True

    if ply.exists():
        src = ply
        src_kind = "stage1_scene_ply"
        print(f"Using Stage 1 PLY backend source: {ply.name}")
    elif glb.exists():
        src = glb
        src_kind = "converted_from_stage1_glb"
        print("[WARN] scene_uv.obj / scene.ply not found. Using scene.glb fallback backend source.")
    else:
        raise FileNotFoundError(f"No scene_uv.obj, scene.ply, or scene.glb found in {package_dir}")

    mesh = load_mesh_with_trimesh(src)
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise RuntimeError(f"Could not read valid mesh from {src}")

    mesh = normalize_vertex_colors_for_mitsuba(mesh)

    out_ply = work_dir / "scene_mitsuba_backend.ply"
    mesh.export(str(out_ply))
    print(f"Prepared Mitsuba backend PLY: {out_ply} ({len(mesh.vertices)} vertices, {len(mesh.faces)} faces)")
    return out_ply, src_kind, "ply", False


def load_camera(camera_path: Path) -> dict[str, Any]:
    with open(camera_path, "r", encoding="utf-8") as f:
        cam = json.load(f)
    if "intrinsic" not in cam:
        raise KeyError(f"camera.json missing 'intrinsic': {camera_path}")
    return cam


def mitsuba_sensor_from_camera(mi, camera: dict[str, Any], spp: int, width: int | None = None, height: int | None = None) -> dict[str, Any]:
    K = np.asarray(camera["intrinsic"], dtype=np.float32)
    w = int(width or camera.get("resize_W") or camera.get("width"))
    h = int(height or camera.get("resize_H") or camera.get("height"))
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    x_fov = math.degrees(2.0 * math.atan(w / (2.0 * fx)))
    pp_x = (cx - 0.5 * w) / w
    pp_y = (cy - 0.5 * h) / h

    T = mi.ScalarTransform4f
    return {
        "type": "perspective",
        "fov_axis": "x",
        "fov": x_fov,
        "principal_point_offset_x": pp_x,
        "principal_point_offset_y": pp_y,
        "to_world": T.look_at(origin=[0, 0, 0], target=[0, 0, -1], up=[0, 1, 0]),
        "sampler": {"type": "independent", "sample_count": int(spp)},
        "film": {
            "type": "hdrfilm",
            "width": w,
            "height": h,
            "pixel_format": "rgb",
            "rfilter": {"type": "box"},
        },
    }


def estimate_material_reflectance(pkg: Path, size: tuple[int, int], mask_np: np.ndarray, mode: str) -> list[float]:
    """Return a stable RGB reflectance for fallback diffuse/light fitting modes."""
    if mode == "gray":
        return [0.7, 0.7, 0.7]
    if mode == "mean_albedo" and (pkg / "albedo.png").exists():
        alb = read_rgb(pkg / "albedo.png", size=size, linear=True)
        m = mask_np[..., 0] > 0.5
        if np.any(m):
            rgb = alb[m].mean(axis=0)
        else:
            rgb = alb.reshape(-1, 3).mean(axis=0)
        rgb = np.clip(rgb, 0.05, 0.95)
        return [float(x) for x in rgb]
    return [0.7, 0.7, 0.7]


def estimate_scalar_map(pkg: Path, filename: str, size: tuple[int, int], mask_np: np.ndarray, default: float, lo: float, hi: float) -> float:
    """Estimate one stable scalar material parameter from an image-space map."""
    path = pkg / filename
    if not path.exists():
        return float(default)
    img = Image.open(path).convert("L").resize(size, Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    m = mask_np[..., 0] > 0.5
    if np.any(m):
        val = float(np.median(arr[m]))
    else:
        val = float(np.median(arr))
    return float(np.clip(val, lo, hi))


def render_to_numpy(mi, dr, image) -> np.ndarray:
    arr = np.array(image, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    return arr


def dr_scalar_to_float(x):
    import numpy as np
    import drjit as dr
    dr.eval(x)
    return float(np.asarray(x).reshape(-1)[0])



def build_principled_bsdf(
    pkg: Path,
    use_uv_textures: bool,
    use_vertex_color: bool,
    reflectance_rgb: list[float],
    roughness: float,
    metallic: float,
    normal_mode: str = "auto",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Build a Mitsuba principled BSDF and a metadata dict describing where
    each parameter came from.

    Important:
      Mitsuba's normalmap plugin expects a tangent/local-space normal map.
      Many inverse-rendering pipelines output camera-space normals instead.
      Therefore:
        - auto: do not use normal.png as a bitmap normal map
        - tangent_map: force-wrap the BSDF with Mitsuba normalmap
    """
    if use_uv_textures and (pkg / "albedo.png").exists():
        base_color = {
            "type": "bitmap",
            "filename": str((pkg / "albedo.png").resolve()),
            "raw": False,
        }
        albedo_mode = "bitmap_albedo_uv"
    elif use_vertex_color:
        base_color = {
            "type": "mesh_attribute",
            "name": "vertex_color",
        }
        albedo_mode = "vertex_color"
    else:
        base_color = {
            "type": "rgb",
            "value": reflectance_rgb,
        }
        albedo_mode = "constant_rgb"

    base_bsdf = {
        "type": "principled",
        "base_color": base_color,
    }

    roughness_mode = "scalar"
    if use_uv_textures and (pkg / "roughness.png").exists():
        base_bsdf["roughness"] = {
            "type": "bitmap",
            "filename": str((pkg / "roughness.png").resolve()),
            "raw": True,
        }
        roughness_mode = "bitmap_roughness_uv"
    else:
        base_bsdf["roughness"] = float(roughness)

    metallic_mode = "scalar"
    if use_uv_textures and (pkg / "metallic.png").exists():
        base_bsdf["metallic"] = {
            "type": "bitmap",
            "filename": str((pkg / "metallic.png").resolve()),
            "raw": True,
        }
        metallic_mode = "bitmap_metallic_uv"
    else:
        base_bsdf["metallic"] = float(metallic)

    # Default: keep mesh vertex normals. Stage 1 already projects normals to
    # vertex normals, which is often the correct choice for camera-space normals.
    nested_bsdf = base_bsdf
    normal_mode_effective = "mesh_vertex_normals"

    use_tangent_normal = (
        normal_mode == "tangent_map"
        and use_uv_textures
        and (pkg / "normal.png").exists()
    )
    if use_tangent_normal:
        nested_bsdf = {
            "type": "normalmap",
            "normalmap": {
                "type": "bitmap",
                "filename": str((pkg / "normal.png").resolve()),
                "raw": True,
            },
            "bsdf": base_bsdf,
        }
        normal_mode_effective = "bitmap_tangent_normal_uv"
    elif normal_mode == "tangent_map" and not ((pkg / "normal.png").exists() and use_uv_textures):
        normal_mode_effective = "requested_tangent_map_but_missing_uv_or_normal_png"

    wrapped = {"type": "twosided", "bsdf": nested_bsdf}
    meta = {
        "albedo_mode": albedo_mode,
        "roughness_mode": roughness_mode,
        "metallic_mode": metallic_mode,
        "normal_mode": normal_mode_effective,
    }
    return wrapped, meta




def make_initial_envmap(path: Path, width: int, height: int, rgb: list[float]) -> Path:
    """Write a tiny latitude-longitude HDR environment map for optimization."""
    import numpy as np
    import mitsuba as mi

    path.parent.mkdir(parents=True, exist_ok=True)
    env = np.zeros((height, width, 3), dtype=np.float32)
    env[:, :, 0] = float(rgb[0])
    env[:, :, 1] = float(rgb[1])
    env[:, :, 2] = float(rgb[2])
    mi.util.write_bitmap(str(path), env)
    return path


def mesh_bounds_for_lights(mesh_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read mesh bounds for placing a simple point-light grid."""
    import trimesh
    loaded = trimesh.load(str(mesh_path), process=False)
    if isinstance(loaded, trimesh.Scene):
        geoms = list(loaded.geometry.values())
        if not geoms:
            return np.array([-1, -1, -3], dtype=np.float32), np.array([1, 1, -1], dtype=np.float32)
        mesh = trimesh.util.concatenate(geoms)
    else:
        mesh = loaded
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    if verts.size == 0:
        return np.array([-1, -1, -3], dtype=np.float32), np.array([1, 1, -1], dtype=np.float32)
    return verts.min(axis=0), verts.max(axis=0)


def build_point_light_positions(mesh_path: Path, grid_x: int, grid_y: int, layers: int = 1) -> list[list[float]]:
    """
    Fixed point-light positions around the visible camera frustum.

    Stage 1 geometry is OpenGL-style: camera at origin looking along -Z.
    We keep positions fixed and optimize only RGB intensity.
    """
    bmin, bmax = mesh_bounds_for_lights(mesh_path)
    center = 0.5 * (bmin + bmax)
    extent = np.maximum(bmax - bmin, 1e-3)
    radius = float(np.linalg.norm(extent))

    # Make sure we cover the object slightly wider than its bounding box.
    x_pad = 0.35 * max(float(extent[0]), 0.25 * radius)
    y_pad = 0.35 * max(float(extent[1]), 0.25 * radius)
    xs = np.linspace(float(bmin[0] - x_pad), float(bmax[0] + x_pad), max(1, grid_x))
    ys = np.linspace(float(bmin[1] - y_pad), float(bmax[1] + y_pad), max(1, grid_y))

    # Put lights between camera and scene, plus optional deeper layers.
    # Since scene z is negative, bmax[2] is usually closest to camera.
    near_z = min(-0.05, float(bmax[2] + 0.15 * radius))
    scene_z = float(center[2])
    zs = np.linspace(near_z, scene_z, max(1, layers))

    positions: list[list[float]] = []
    for z in zs:
        for y in ys:
            for x in xs:
                positions.append([float(x), float(y), float(z)])
    return positions


def find_envmap_data_key(params) -> str | None:
    for k in params.keys():
        kl = k.lower()
        if "env_light" in kl and kl.endswith(".data"):
            return k
    for k in params.keys():
        kl = k.lower()
        if "env_light" in kl and "data" in kl:
            return k
    return None


def find_point_intensity_keys(params) -> list[str]:
    keys = []
    for k in params.keys():
        kl = k.lower()
        if kl.startswith("point_") and ("intensity.value" in kl or kl.endswith(".intensity.value")):
            keys.append(k)
    if not keys:
        for k in params.keys():
            kl = k.lower()
            if kl.startswith("point_") and "intensity" in kl:
                keys.append(k)
    return sorted(keys)


def save_envmap_from_params(path: Path, env_data, env_w: int, env_h: int) -> bool:
    """Try to save optimized Mitsuba envmap data to EXR."""
    import numpy as np
    import mitsuba as mi

    try:
        arr = np.array(env_data, dtype=np.float32)
        arr = arr.reshape((env_h, env_w, 3))
        mi.util.write_bitmap(str(path), arr)
        return True
    except Exception as e:
        print(f"[WARN] Could not save optimized envmap: {e}")
        return False



def make_scene_dict(
    mi,
    mesh_path: Path,
    shape_type: str,
    camera: dict[str, Any],
    spp: int,
    integrator: str,
    init_rgb: list[float],
    pkg: Path,
    reflectance_rgb: list[float] | None = None,
    use_vertex_color: bool = False,
    use_uv_textures: bool = False,
    roughness: float = 0.55,
    metallic: float = 0.0,
    material_model: str = "principled",
    normal_mode: str = "auto",
    light_mode: str = "envmap_points",
    envmap_path: Path | None = None,
    point_positions: list[list[float]] | None = None,
    point_init: float = 0.01,
) -> tuple[dict[str, Any], dict[str, Any]]:
    sensor = mitsuba_sensor_from_camera(mi, camera, spp=spp)

    if material_model == "principled":
        shape_bsdf, bsdf_meta = build_principled_bsdf(
            pkg=pkg,
            use_uv_textures=use_uv_textures,
            use_vertex_color=use_vertex_color,
            reflectance_rgb=reflectance_rgb or [0.7, 0.7, 0.7],
            roughness=roughness,
            metallic=metallic,
            normal_mode=normal_mode,
        )
    else:
        if use_vertex_color:
            reflectance = {"type": "mesh_attribute", "name": "vertex_color"}
            albedo_mode = "vertex_color"
        else:
            reflectance = {"type": "rgb", "value": reflectance_rgb or [0.7, 0.7, 0.7]}
            albedo_mode = "constant_rgb"
        shape_bsdf = {
            "type": "twosided",
            "bsdf": {
                "type": "diffuse",
                "reflectance": reflectance,
            },
        }
        bsdf_meta = {
            "albedo_mode": albedo_mode,
            "roughness_mode": "unused_in_diffuse",
            "metallic_mode": "unused_in_diffuse",
            "normal_mode": "unused_in_diffuse",
        }

    scene = {
        "type": "scene",
        "integrator": {"type": integrator, "max_depth": 4},
        "sensor": sensor,
        "shape": {
            "type": shape_type,
            "filename": str(mesh_path),
            "bsdf": shape_bsdf,
        },
    }

    if light_mode == "constant":
        scene["env_light"] = {
            "type": "constant",
            "radiance": {"type": "rgb", "value": init_rgb},
        }
    else:
        if envmap_path is None:
            raise ValueError("envmap_path is required for light_mode='envmap_points'")
        scene["env_light"] = {
            "type": "envmap",
            "filename": str(envmap_path),
            "scale": 1.0,
            "mis_compensation": True,
        }

        for i, pos in enumerate(point_positions or []):
            scene[f"point_{i:03d}"] = {
                "type": "point",
                "position": pos,
                "intensity": {
                    "type": "rgb",
                    "value": [float(point_init), float(point_init), float(point_init)],
                },
            }

    return scene, bsdf_meta


def find_radiance_key(params) -> str:
    keys = list(params.keys())
    candidates = [k for k in keys if k.endswith("radiance.value")]
    if not candidates:
        candidates = [k for k in keys if "radiance" in k.lower()]
    if not candidates:
        raise KeyError("Could not find a differentiable radiance parameter. Available keys:\n" + "\n".join(keys))
    for k in candidates:
        if "env_light" in k or "emitter" in k:
            return k
    return candidates[0]


def safe_remove_path(path: Path) -> None:
    """Remove a file or directory if it exists. Used only after a successful manifest write."""
    try:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    except Exception as e:
        print(f"[WARN] Could not remove {path}: {e}")


def copy_stage2_input_assets(pkg: Path, out: Path, include_glb: bool = False) -> dict[str, str]:
    """Copy only compact assets needed to make Stage 2 output self-contained.

    Avoids the old encoder_pairs/renderer_pairs duplication. Stage 1 package can be
    deleted after this succeeds.
    """
    copied: dict[str, str] = {}
    names = [
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
    ]
    if include_glb:
        names.append("scene.glb")
    for name in names:
        copied_name = copy_if_exists(pkg / name, out / name)
        if copied_name is not None:
            copied[Path(name).stem] = copied_name
    return copied


def cleanup_stage2_after_success(out: Path, keep_work: bool, keep_debug_renders: bool, keep_exr: bool, save_pairs: bool) -> None:
    """Production cleanup for Stage 2.

    Keeps final PNGs, JSON metadata, final light files, and compact copied assets.
    Removes temporary work folders, optional training-pair duplicates, and bulky EXR/debug renders.
    """
    if not keep_work:
        safe_remove_path(out / "_work")
    if not save_pairs:
        safe_remove_path(out / "encoder_pairs")
        safe_remove_path(out / "renderer_pairs")
    if not keep_debug_renders:
        for name in [
            "render_initial.png",
            "render_material_preview.png",
        ]:
            safe_remove_path(out / name)
    if not keep_exr:
        for name in [
            "render_initial.exr",
            "render_optimized.exr",
            "render_material_preview.exr",
        ]:
            safe_remove_path(out / name)
    print("[OK] Stage 2 cleanup complete")


def main() -> None:
    parser = argparse.ArgumentParser("Stage 2 Mitsuba constant RGB light estimation (UV-aware)")
    parser.add_argument("--input-root", type=Path, default=Path("asset/input"))
    parser.add_argument("--output-root", type=Path, default=Path("asset/output"))
    parser.add_argument("--scene", type=str, default=None, help="Process one package name under asset/input/. Omit to process all folders.")
    parser.add_argument("--variant", type=str, default="cuda_ad_rgb", help="Mitsuba variant, e.g. cuda_ad_rgb or llvm_ad_rgb")
    parser.add_argument("--spp", type=int, default=32, help="Samples per pixel for optimization renders")
    parser.add_argument("--final-spp", type=int, default=128, help="Samples per pixel for saved final render")
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--init-light", type=float, nargs=3, default=[1.0, 1.0, 1.0])
    parser.add_argument("--integrator", type=str, default="prb", help="Use prb for differentiable rendering; fallback to path manually if needed.")
    parser.add_argument("--material-mode", choices=["auto", "texture", "vertex_color", "mean_albedo", "gray"], default="auto",
                        help="Albedo source. auto = UV bitmap if scene_uv.obj+albedo.png exist, else vertex_color, else mean_albedo.")
    parser.add_argument("--material-model", choices=["principled", "diffuse"], default="principled", help="Mitsuba BSDF model. Option A uses principled.")
    parser.add_argument("--roughness", type=float, default=None, help="Override scalar roughness fallback. Default: median of roughness.png, else 0.55.")
    parser.add_argument("--metallic", type=float, default=None, help="Override scalar metallic fallback. Default: median of metallic.png, else 0.0.")
    parser.add_argument("--normal-mode", choices=["auto", "none", "tangent_map"], default="auto",
                        help="How to use normal.png. auto/none keep mesh vertex normals; tangent_map wraps the BSDF with Mitsuba normalmap using normal.png. Use tangent_map only if normal.png is a tangent/local-space normal map.")
    parser.add_argument("--light-mode", choices=["envmap_points", "constant"], default="envmap_points",
                        help="envmap_points optimizes a low-res envmap plus a fixed point-light grid. constant keeps the older constant-RGB environment baseline.")
    parser.add_argument("--envmap-width", type=int, default=16, help="Low-res optimized latitude-longitude envmap width.")
    parser.add_argument("--envmap-height", type=int, default=8, help="Low-res optimized latitude-longitude envmap height.")
    parser.add_argument("--point-grid-x", type=int, default=4, help="Point-light grid columns.")
    parser.add_argument("--point-grid-y", type=int, default=3, help="Point-light grid rows.")
    parser.add_argument("--point-layers", type=int, default=1, help="Number of point-light depth layers.")
    parser.add_argument("--point-init", type=float, default=0.01, help="Initial point-light RGB intensity.")
    parser.add_argument("--light-reg", type=float, default=1e-4, help="Small L2 regularizer on light values.")
    parser.add_argument(
        "--target-env-rgb",
        type=float,
        nargs=3,
        default=None,
        help="Optional target relight constant environment RGB. When set, saved optimized render/composite use this light after original-light optimization.",
    )
    parser.add_argument(
        "--target-point-position",
        type=float,
        nargs=3,
        default=None,
        help="Optional target relight point-light XYZ position in Stage 2 scene coordinates.",
    )
    parser.add_argument(
        "--target-point-rgb",
        type=float,
        nargs=3,
        default=None,
        help="Optional target relight point-light RGB intensity.",
    )
    parser.add_argument("--preview-light", type=float, nargs=3, default=[1.0, 1.0, 1.0], help="White/material preview constant environment RGB used for render_material_preview.png.")
    parser.add_argument("--save-debug-renders", action="store_true", help="Keep render_initial.png and render_material_preview.png. Disabled by default for production cleanup.")
    parser.add_argument("--save-exr", action="store_true", help="Keep render EXR files. Disabled by default because they are large; optimized_envmap.exr is still kept as a final light asset.")
    parser.add_argument("--save-pairs", action="store_true", help="Write encoder_pairs/ and renderer_pairs/. Disabled by default to avoid duplicate files in production output.")
    parser.add_argument("--keep-work", action="store_true", help="Keep output/<scene>/_work for debugging. Disabled by default.")
    parser.add_argument("--no-copy-input-assets", dest="copy_input_assets", action="store_false", help="Do not copy compact Stage 1 assets into Stage 2 output. Not recommended if Stage 1 packages will be deleted.")
    parser.add_argument("--copy-canonical-glb", action="store_true", help="Also copy scene.glb into Stage 2 output. Disabled by default to avoid duplicate geometry.")
    parser.set_defaults(copy_input_assets=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    import mitsuba as mi
    import drjit as dr

    try:
        mi.set_variant(args.variant)
    except Exception as e:
        if args.variant == "cuda_ad_rgb":
            print(f"[WARN] Could not set cuda_ad_rgb ({e}). Falling back to llvm_ad_rgb.")
            mi.set_variant("llvm_ad_rgb")
        else:
            raise

    if args.scene:
        input_dirs = [args.input_root / args.scene]
    else:
        input_dirs = sorted([p for p in args.input_root.iterdir() if p.is_dir()])

    if not input_dirs:
        sys.exit(f"No input packages found under {args.input_root}")

    for pkg in input_dirs:
        name = pkg.name
        out = args.output_root / name
        if out.exists() and not args.overwrite:
            print(f"[SKIP] {name}: output exists, skipping. Use --overwrite to re-run.")
            continue
        if out.exists() and args.overwrite:
            shutil.rmtree(out)

        print(f"\n{'=' * 80}\nStage 2 constant RGB light: {name}\n{'=' * 80}")
        out.mkdir(parents=True, exist_ok=True)
        (out / "light").mkdir(exist_ok=True)
        if args.save_pairs:
            (out / "encoder_pairs").mkdir(exist_ok=True)
            (out / "renderer_pairs").mkdir(exist_ok=True)
        work_dir = out / "_work"
        work_dir.mkdir(exist_ok=True)

        camera_path = pkg / "camera.json"
        ref_path = pkg / "reference.png"
        mask_path = pkg / "geometry_mask.png"
        bg_path = pkg / "background.png"

        if not camera_path.exists():
            raise FileNotFoundError(f"Missing camera.json in {pkg}")
        if not ref_path.exists():
            raise FileNotFoundError(f"Missing reference.png in {pkg}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing geometry_mask.png in {pkg}")

        camera = load_camera(camera_path)
        width = int(camera.get("resize_W") or camera.get("width"))
        height = int(camera.get("resize_H") or camera.get("height"))
        size = (width, height)

        mesh_path, backend_source, shape_type, shape_has_uv = prepare_mitsuba_backend_shape(pkg, work_dir)
        scene_glb = pkg / "scene.glb"

        target_np = read_rgb(ref_path, size=size, linear=True)
        mask_np = read_mask(mask_path, size=size)
        if mask_np.mean() < 1e-4:
            print("[WARN] geometry_mask is almost empty; using full-image mask for this run.")
            mask_np = np.ones_like(mask_np, dtype=np.float32)

        reflectance_rgb = estimate_material_reflectance(pkg, size=size, mask_np=mask_np, mode=("mean_albedo" if args.material_mode in ["auto", "texture", "vertex_color"] else args.material_mode))
        roughness = float(args.roughness) if args.roughness is not None else estimate_scalar_map(pkg, "roughness.png", size=size, mask_np=mask_np, default=0.55, lo=0.02, hi=1.0)
        metallic = float(args.metallic) if args.metallic is not None else estimate_scalar_map(pkg, "metallic.png", size=size, mask_np=mask_np, default=0.0, lo=0.0, hi=1.0)

        if args.material_mode == "auto":
            use_uv_textures = shape_has_uv and (pkg / "albedo.png").exists()
            use_vertex_color = (not use_uv_textures)
        elif args.material_mode == "texture":
            use_uv_textures = shape_has_uv and (pkg / "albedo.png").exists()
            if not use_uv_textures:
                print("[WARN] texture mode requested, but scene_uv.obj or albedo.png is missing. Falling back.")
            use_vertex_color = not use_uv_textures
        elif args.material_mode == "vertex_color":
            use_uv_textures = False
            use_vertex_color = True
        else:
            use_uv_textures = False
            use_vertex_color = False

        print(f"Backend: {backend_source} | shape_type={shape_type} | shape_has_uv={shape_has_uv}")
        if use_uv_textures:
            print(f"Material mode: UV bitmap albedo + {args.material_model} BSDF, roughness={roughness:.3f}, metallic={metallic:.3f}, normal_mode={args.normal_mode}")
        elif use_vertex_color:
            print(f"Material mode: vertex_color albedo + {args.material_model} BSDF, roughness={roughness:.3f}, metallic={metallic:.3f}, normal_mode={args.normal_mode}")
        else:
            print(f"Material mode: {args.material_mode} + {args.material_model} BSDF, reflectance={reflectance_rgb}, roughness={roughness:.3f}, metallic={metallic:.3f}, normal_mode={args.normal_mode}")

        target = mi.TensorXf(target_np)
        mask = mi.TensorXf(mask_np)

        envmap_path = work_dir / "initial_envmap.exr"
        point_positions: list[list[float]] = []
        if args.light_mode == "envmap_points":
            make_initial_envmap(envmap_path, args.envmap_width, args.envmap_height, list(args.init_light))
            point_positions = build_point_light_positions(
                mesh_path=mesh_path,
                grid_x=args.point_grid_x,
                grid_y=args.point_grid_y,
                layers=args.point_layers,
            )
            print(f"Light model: optimized {args.envmap_width}x{args.envmap_height} envmap + {len(point_positions)} fixed point lights")
        else:
            print("Light model: constant RGB environment baseline")

        scene_dict, bsdf_meta = make_scene_dict(
            mi=mi,
            mesh_path=mesh_path,
            shape_type=shape_type,
            camera=camera,
            spp=args.spp,
            integrator=args.integrator,
            init_rgb=list(args.init_light),
            pkg=pkg,
            reflectance_rgb=reflectance_rgb,
            use_vertex_color=use_vertex_color,
            use_uv_textures=use_uv_textures,
            roughness=roughness,
            metallic=metallic,
            material_model=args.material_model,
            normal_mode=args.normal_mode,
            light_mode=args.light_mode,
            envmap_path=envmap_path if args.light_mode == "envmap_points" else None,
            point_positions=point_positions,
            point_init=args.point_init,
        )

        try:
            scene = mi.load_dict(scene_dict)
        except Exception as e:
            if args.integrator != "path":
                print(f"[WARN] Could not load integrator '{args.integrator}' ({e}). Falling back to 'path'.")
                scene_dict, bsdf_meta = make_scene_dict(
                    mi=mi,
                    mesh_path=mesh_path,
                    shape_type=shape_type,
                    camera=camera,
                    spp=args.spp,
                    integrator="path",
                    init_rgb=list(args.init_light),
                    pkg=pkg,
                    reflectance_rgb=reflectance_rgb,
                    use_vertex_color=use_vertex_color,
                    use_uv_textures=use_uv_textures,
                    roughness=roughness,
                    metallic=metallic,
                    material_model=args.material_model,
                    normal_mode=args.normal_mode,
                    light_mode=args.light_mode,
                    envmap_path=envmap_path if args.light_mode == "envmap_points" else None,
                    point_positions=point_positions,
                    point_init=args.point_init,
                )
                scene = mi.load_dict(scene_dict)
            else:
                raise

        # Optional debug preview: render material under a simple white constant environment.
        if args.save_debug_renders:
            preview_scene_dict, _preview_meta = make_scene_dict(
                mi=mi,
                mesh_path=mesh_path,
                shape_type=shape_type,
                camera=camera,
                spp=args.final_spp,
                integrator="path",
                init_rgb=list(args.preview_light),
                pkg=pkg,
                reflectance_rgb=reflectance_rgb,
                use_vertex_color=use_vertex_color,
                use_uv_textures=use_uv_textures,
                roughness=roughness,
                metallic=metallic,
                material_model=args.material_model,
                normal_mode=args.normal_mode,
                light_mode="constant",
                envmap_path=None,
                point_positions=None,
                point_init=args.point_init,
            )
            preview_scene = mi.load_dict(preview_scene_dict)
            preview_img = mi.render(preview_scene, spp=args.final_spp, seed=12345)
            mi.util.write_bitmap(str(out / "render_material_preview.png"), preview_img)
            if args.save_exr:
                mi.util.write_bitmap(str(out / "render_material_preview.exr"), preview_img)

        params = mi.traverse(scene)

        light_keys: list[str] = []
        env_key: str | None = None
        point_keys: list[str] = []

        if args.light_mode == "envmap_points":
            env_key = find_envmap_data_key(params)
            point_keys = find_point_intensity_keys(params)
            if env_key is not None:
                light_keys.append(env_key)
            light_keys.extend(point_keys)
            if not light_keys:
                raise KeyError("Could not find envmap or point-light parameters to optimize.")
            print("Optimizing light parameters:")
            for k in light_keys:
                print(f"  - {k}")
        else:
            light_key = find_radiance_key(params)
            light_keys = [light_key]
            print(f"Optimizing parameter: {light_key}")

        if args.save_debug_renders or args.save_pairs:
            initial_img = mi.render(scene, spp=args.final_spp)
            mi.util.write_bitmap(str(out / "render_initial.png"), initial_img)
            if args.save_exr:
                mi.util.write_bitmap(str(out / "render_initial.exr"), initial_img)

        opt = mi.ad.Adam(lr=args.lr)
        for k in light_keys:
            opt[k] = params[k]
        log: list[dict[str, Any]] = []

        for it in range(args.iters):
            params.update(opt)
            image = mi.render(scene, params=params, spp=args.spp, seed=it)
            diff = (image - target) * mask
            loss = dr.mean(dr.square(diff))

            # Small light-energy regularization prevents a few point lights from
            # exploding while the envmap becomes black.
            if args.light_reg > 0:
                reg = 0.0
                for k in light_keys:
                    reg = reg + dr.mean(dr.square(opt[k]))
                loss = loss + float(args.light_reg) * reg

            dr.backward(loss)
            opt.step()

            # Keep all recovered lighting non-negative and bounded.
            for k in light_keys:
                opt[k] = dr.clip(opt[k], 0.0, 50.0)
            params.update(opt)

            if it % 10 == 0 or it == args.iters - 1:
                loss_value = dr_scalar_to_float(loss)
                summary: dict[str, Any] = {"iter": it, "loss": loss_value}
                if args.light_mode == "envmap_points":
                    try:
                        if env_key is not None:
                            env_arr = np.array(params[env_key], dtype=np.float32).reshape(-1, 3)
                            summary["env_mean_rgb"] = env_arr.mean(axis=0).tolist()
                            summary["env_max"] = float(env_arr.max())
                        if point_keys:
                            point_vals = [np.array(params[k], dtype=np.float32).reshape(-1).tolist() for k in point_keys]
                            point_np = np.asarray(point_vals, dtype=np.float32)
                            summary["point_mean_rgb"] = point_np.mean(axis=0).tolist()
                            summary["point_max"] = float(point_np.max())
                    except Exception:
                        pass
                    print(f"iter {it:04d} | loss={loss_value:.6g} | env_mean={summary.get('env_mean_rgb')} | point_max={summary.get('point_max')}")
                else:
                    try:
                        light_rgb = np.array(params[light_keys[0]], dtype=np.float32).reshape(-1).tolist()
                    except Exception:
                        light_rgb = []
                    summary["light_rgb"] = light_rgb
                    print(f"iter {it:04d} | loss={loss_value:.6g} | light={light_rgb}")
                log.append(summary)

        if args.light_mode == "envmap_points":
            optimized_envmap_name = None
            if env_key is not None:
                env_out = out / "light" / "optimized_envmap.exr"
                if save_envmap_from_params(env_out, params[env_key], args.envmap_width, args.envmap_height):
                    optimized_envmap_name = "optimized_envmap.exr"

            point_lights = []
            for i, (pos, key) in enumerate(zip(point_positions, point_keys)):
                try:
                    rgb = np.array(params[key], dtype=np.float32).reshape(-1).tolist()
                except Exception:
                    rgb = [0.0, 0.0, 0.0]
                point_lights.append({
                    "id": i,
                    "position": pos,
                    "rgb_intensity": rgb,
                    "mitsuba_parameter": key,
                })

            initial_light = {
                "type": "envmap_plus_point_grid",
                "initial_envmap": str(envmap_path.name),
                "envmap_width": args.envmap_width,
                "envmap_height": args.envmap_height,
                "point_grid": [args.point_grid_x, args.point_grid_y, args.point_layers],
                "point_init": args.point_init,
            }
            optimized_light = {
                "type": "envmap_plus_point_grid",
                "envmap": optimized_envmap_name,
                "envmap_mitsuba_parameter": env_key,
                "point_lights": point_lights,
            }
            if envmap_path.exists() and (args.save_debug_renders or args.save_exr):
                shutil.copy2(envmap_path, out / "light" / "initial_envmap.exr")
            (out / "light" / "optimized_point_lights.json").write_text(json.dumps(point_lights, indent=2), encoding="utf-8")
        else:
            try:
                optimized_rgb = np.array(params[light_keys[0]], dtype=np.float32).reshape(-1).tolist()
            except Exception:
                optimized_rgb = list(args.init_light)
            initial_light = {"type": "constant_rgb_environment", "radiance": list(args.init_light)}
            optimized_light = {"type": "constant_rgb_environment", "radiance": optimized_rgb, "mitsuba_parameter": light_keys[0]}

        (out / "light" / "initial_light.json").write_text(json.dumps(initial_light, indent=2), encoding="utf-8")
        (out / "light" / "optimized_light.json").write_text(json.dumps(optimized_light, indent=2), encoding="utf-8")
        (out / "optimization_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")

        render_light_model = optimized_light
        if args.target_env_rgb is not None or args.target_point_position is not None:
            # Stage 4 asks for a target relight buffer. The optimization above
            # still estimates the source image lighting; this final pass swaps in
            # user-requested target emitters.
            target_env_rgb = list(args.target_env_rgb) if args.target_env_rgb is not None else [0.0, 0.0, 0.0]
            target_scene_dict, _target_bsdf_meta = make_scene_dict(
                mi=mi,
                mesh_path=mesh_path,
                shape_type=shape_type,
                camera=camera,
                spp=args.final_spp,
                integrator="path",
                init_rgb=target_env_rgb,
                pkg=pkg,
                reflectance_rgb=reflectance_rgb,
                use_vertex_color=use_vertex_color,
                use_uv_textures=use_uv_textures,
                roughness=roughness,
                metallic=metallic,
                material_model=args.material_model,
                normal_mode=args.normal_mode,
                light_mode="constant",
                envmap_path=None,
                point_positions=None,
                point_init=args.point_init,
            )
            target_point = None
            if args.target_point_position is not None:
                target_point_rgb = list(args.target_point_rgb) if args.target_point_rgb is not None else [10.0, 10.0, 10.0]
                target_point = {
                    "type": "point",
                    "position": list(args.target_point_position),
                    "intensity": {
                        "type": "rgb",
                        "value": target_point_rgb,
                    },
                }
                target_scene_dict["target_point"] = target_point
            target_scene = mi.load_dict(target_scene_dict)
            final_img = mi.render(target_scene, spp=args.final_spp, seed=999)
            render_light_model = {
                "type": "target_environment_plus_point",
                "radiance": target_env_rgb,
                "point_light": target_point,
                "source_optimized_light": optimized_light,
            }
            (out / "light" / "target_light.json").write_text(json.dumps(render_light_model, indent=2), encoding="utf-8")
        else:
            final_img = mi.render(scene, params=params, spp=args.final_spp, seed=999)

        mi.util.write_bitmap(str(out / "render_optimized.png"), final_img)
        if args.save_exr:
            mi.util.write_bitmap(str(out / "render_optimized.exr"), final_img)

        final_np_linear = render_to_numpy(mi, dr, final_img)
        final_srgb = linear_to_srgb(final_np_linear)
        ref_srgb = read_rgb(ref_path, size=size, linear=False)
        residual = np.abs(final_srgb - ref_srgb)
        write_png_srgb(out / "residual.png", residual / max(1e-6, float(np.percentile(residual, 99))))

        if bg_path.exists():
            bg_srgb = read_rgb(bg_path, size=size, linear=False)
            m = mask_np
            composite = final_srgb * m + bg_srgb * (1.0 - m)
            write_png_srgb(out / "composite_optimized.png", composite)
        else:
            shutil.copy2(out / "render_optimized.png", out / "composite_optimized.png")

        render_backend_name = "scene_render_backend.obj" if mesh_path.suffix.lower() == ".obj" else "scene_render_backend.ply"

        scene_json = {
            "integrator": scene_dict["integrator"],
            "sensor": {
                "type": "perspective",
                "width": width,
                "height": height,
                "fov_axis": "x",
                "fov": scene_dict["sensor"]["fov"],
                "principal_point_offset_x": scene_dict["sensor"].get("principal_point_offset_x", 0.0),
                "principal_point_offset_y": scene_dict["sensor"].get("principal_point_offset_y", 0.0),
                "look_at": {"origin": [0, 0, 0], "target": [0, 0, -1], "up": [0, 1, 0]},
            },
            "shape": {
                "type": shape_type,
                "filename": render_backend_name,
                "bsdf": {
                    "model": args.material_model,
                    "albedo_mode": bsdf_meta["albedo_mode"],
                    "roughness_mode": bsdf_meta["roughness_mode"],
                    "metallic_mode": bsdf_meta["metallic_mode"],
                    "normal_mode": bsdf_meta["normal_mode"],
                    "roughness_fallback_scalar": roughness,
                    "metallic_fallback_scalar": metallic,
                },
            },
            "emitter": render_light_model,
        }
        (out / "scene_mitsuba.json").write_text(json.dumps(scene_json, indent=2), encoding="utf-8")

        optimize_config = {
            "stage": "envmap_plus_point_grid" if args.light_mode == "envmap_points" else "constant_rgb_environment",
            "variant": mi.variant(),
            "spp": args.spp,
            "final_spp": args.final_spp,
            "iters": args.iters,
            "lr": args.lr,
            "optimized_parameters": light_keys,
            "target_env_rgb": list(args.target_env_rgb) if args.target_env_rgb is not None else None,
            "target_point_position": list(args.target_point_position) if args.target_point_position is not None else None,
            "target_point_rgb": list(args.target_point_rgb) if args.target_point_rgb is not None else None,
            "material_mode": args.material_mode,
            "material_model": args.material_model,
            "shape_type": shape_type,
            "shape_has_uv": shape_has_uv,
            "use_uv_textures": use_uv_textures,
            "requested_normal_mode": args.normal_mode,
            "effective_normal_mode": bsdf_meta["normal_mode"],
            "use_vertex_color": use_vertex_color,
            "material_reflectance_rgb": reflectance_rgb,
            "roughness": roughness,
            "metallic": metallic,
            "backend_source": backend_source,
            "next_stage": "user_controllable_relighting",
        }
        (out / "optimize_config.json").write_text(json.dumps(optimize_config, indent=2), encoding="utf-8")

        if args.save_pairs:
            enc = out / "encoder_pairs"
            ren = out / "renderer_pairs"
            copy_if_exists(ref_path, enc / "input_reference.png")
            copy_if_exists(pkg / "albedo.png", enc / "input_albedo.png")
            copy_if_exists(pkg / "normal.png", enc / "input_normal.png")
            copy_if_exists(pkg / "roughness.png", enc / "input_roughness.png")
            copy_if_exists(pkg / "metallic.png", enc / "input_metallic.png")
            copy_if_exists(mask_path, enc / "input_geometry_mask.png")
            shutil.copy2(out / "light" / "optimized_light.json", enc / "target_light.json")

            if (out / "render_initial.png").exists():
                shutil.copy2(out / "render_initial.png", ren / "prerender.png")
            copy_if_exists(out / "render_material_preview.png", ren / "material_preview.png")
            shutil.copy2(out / "composite_optimized.png", ren / "inference_picture.png")
            copy_if_exists(ref_path, ren / "target_reference.png")
            copy_if_exists(mask_path, ren / "mask.png")
            copy_if_exists(bg_path, ren / "background.png")
            shutil.copy2(camera_path, ren / "camera.json")
            shutil.copy2(out / "light" / "optimized_light.json", ren / "light_condition.json")

        copied_input_assets = {}
        if args.copy_input_assets:
            copied_input_assets = copy_stage2_input_assets(pkg, out, include_glb=args.copy_canonical_glb)

        copy_if_exists(mesh_path, out / render_backend_name)

        manifest = {
            "stage": 2,
            "name": name,
            "input_package": str(pkg),
            "canonical_geometry": "scene.glb" if (args.copy_canonical_glb and scene_glb.exists()) else None,
            "mitsuba_geometry_backend": render_backend_name,
            "backend_source": backend_source,
            "material_mode": args.material_mode,
            "material_model": args.material_model,
            "shape_has_uv": shape_has_uv,
            "use_uv_textures": use_uv_textures,
            "requested_normal_mode": args.normal_mode,
            "effective_normal_mode": bsdf_meta["normal_mode"],
            "outputs": {
                "render_optimized": "render_optimized.png",
                "composite_optimized": "composite_optimized.png",
                "residual": "residual.png",
                "scene_mitsuba": "scene_mitsuba.json",
                "optimization_log": "optimization_log.json",
                "render_initial": "render_initial.png" if (out / "render_initial.png").exists() else None,
                "render_material_preview": "render_material_preview.png" if (out / "render_material_preview.png").exists() else None,
            },
            "copied_input_assets": copied_input_assets,
            "training_pairs": {
                "encoder": "encoder_pairs/",
                "renderer": "renderer_pairs/",
            } if args.save_pairs else None,
            "light_model": render_light_model,
            "source_optimized_light_model": optimized_light if args.target_env_rgb is not None else None,
            "light_mode": args.light_mode,
            "preview_light": list(args.preview_light),
            "note": "Envmap + point-grid light reconstruction baseline. Preferred backend order: scene_uv.obj > scene.ply > scene.glb. If UV exists, Mitsuba uses bitmap albedo/roughness/metallic with a principled BSDF. normal.png is only used as a Mitsuba normal map when --normal-mode tangent_map is set, because Mitsuba expects tangent/local-space normals. debug renders and training-pair folders are optional; production default keeps only compact final outputs and deletes _work after success.",
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        cleanup_stage2_after_success(
            out=out,
            keep_work=args.keep_work,
            keep_debug_renders=args.save_debug_renders,
            keep_exr=args.save_exr,
            save_pairs=args.save_pairs,
        )
        print(f"[OK] Stage 2 output written: {out}")


if __name__ == "__main__":
    main()
