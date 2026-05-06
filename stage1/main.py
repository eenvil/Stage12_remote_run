"""
Scene Reconstruction Pipeline  -  main.py
==========================================

Directory layout expected:
  ./
  +-- main.py                       <- this file (orchestrator)
  +-- main_checkpoint.py            <- resumable variant of main.py
  +-- pyproject.toml                <- uv project file
  +-- asset/
  |   +-- input/                    <- source images (flat or nested folders)
  |   +-- pointcloud/               <- step 1 output: <stem>.ply + <stem>_camera.json
  |   +-- mesh/                     <- step 2 output: <stem>.ply / .obj / .glb
  |   +-- materials/                <- step 3 output (MVInverse intrinsic maps)
  |   |   +-- <stem>/                 <- final per-image maps consumed downstream
  |   |   |   +-- albedo.png
  |   |   |   +-- normals.png
  |   |   |   +-- roughness.png
  |   |   |   +-- metallicity.png
  |   |   |   +-- irradiance.png       <- from MVInverse "shading" head
  |   |   +-- _mvi_input/<stem>/      <- staging dir (one image per call)
  |   |   +-- _mvi_raw/<stem>/        <- raw MVInverse outputs (000_albedo.png, ...)
  |   +-- background/               <- PixelPerfect background/geometry split
  |   +-- output/                   <- step 4 output: per-stem stage-1 packages
  |       +-- <stem>/
  |           +-- scene.glb / scene.ply / scene_uv.obj
  |           +-- albedo.png / roughness.png / metallic.png / normal.png / irradiance.png
  |           +-- background.png / geometry_mask.png / background_mask.png
  |           +-- camera.json / reference.png
  |           +-- scene_mitsuba.json / optimize_config.json
  |           +-- manifest.json
  +-- pixel-perfect-depth/          <- step 1 backend (depth + camera intrinsics)
  +-- MVInverse/                    <- step 3 backend (intrinsic decomposition)
      +-- inference.py              <- driven via subprocess from this file

Usage
-----
  # Run on all images in asset/input/ (recurses into subfolders)
  python main.py

  # Run on a specific image
  python main.py --image asset/input/photo.jpg

  # Skip a branch if you already have outputs from a previous run
  python main.py --skip_depth
  python main.py --skip_mvi

  # Default uses fast image-grid mesh from PixelPerfect depth
  python main.py

  # Default also runs cleanup + Laplacian smoothing on the selected mesh
  python main.py --smooth_iterations 30 --smooth_lambda 0.8

  # Optional old Poisson fallback
  python main.py --use_poisson --mesh_depth 8

  # Try other post-processors when Laplacian shrinkage becomes a problem
  python main.py --smooth_method taubin
  python main.py --smooth_method arap
  python main.py --vertex_cluster_voxel 0.01
"""

import argparse
import os
import sys
import shutil
import subprocess
from pathlib import Path


# ------------------------------------------------------------------
# Paths
# main.py lives in stage1/, repo root is one level above.
# ------------------------------------------------------------------
STAGE1_DIR = Path(__file__).parent.resolve()
ROOT = STAGE1_DIR.parent.resolve()

PPD_DIR = ROOT / "pixel-perfect-depth"
MVI_DIR = ROOT / "mvinverse"

ASSET_DIR  = ROOT / "asset"
INPUT_DIR  = ASSET_DIR / "input"
PCD_DIR    = ASSET_DIR / "pointcloud"
MESH_DIR   = ASSET_DIR / "mesh"
MAT_DIR    = ASSET_DIR / "materials"
OUTPUT_DIR = ASSET_DIR / "output"
BG_DIR     = ASSET_DIR / "background"

# MVInverse checkpoint (HuggingFace repo ID or a local .pt path).
# `inference.py` accepts either: it tries os.path.exists() first, then HF Hub.
MVI_CKPT   = "maddog241/mvinverse"

# Supported input extensions
IMG_EXTS   = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Modality keys exposed to the rest of the pipeline.
# These names are the *contract* that run_combine() and the manifest depend on,
# so we keep them stable even though MVInverse uses slightly different terms.
# Mapping to MVInverse outputs:
#   normals      <- MVInverse "normal"     (re-encoded RGB = n*0.5+0.5)
#   albedo       <- MVInverse "albedo"
#   roughness    <- MVInverse "roughness"
#   metallicity  <- MVInverse "metallic"
#   irradiance   <- MVInverse "shading"    (closest semantic analog)
MODALITIES = ["normals", "albedo", "irradiance", "roughness", "metallicity"]
MVI_OUTPUT_NAME_MAP = {
    "normals":     "normal",
    "albedo":      "albedo",
    "irradiance":  "shading",
    "roughness":   "roughness",
    "metallicity": "metallic",
}


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def ensure_dirs():
    for d in (INPUT_DIR, PCD_DIR, MESH_DIR, MAT_DIR, BG_DIR, OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)


def run(cmd: list, cwd: Path = None):
    """Run a subprocess, inheriting stdout/stderr so progress is visible."""
    cmd_str = " ".join(str(c) for c in cmd)
    print(f"\n>  {cmd_str}")
    print("-" * 70)
    result = subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None)
    if result.returncode != 0:
        sys.exit(f"\n[ERR]  Command failed (exit {result.returncode})")


def find_images(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMG_EXTS)


def header(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")



def original_stem_from_mesh(mesh_path: Path) -> str:
    """Return the logical source stem used by depth/camera/background assets."""
    stem = mesh_path.stem
    for suffix in ("_post", "_smoothed", "_smooth"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


# ------------------------------------------------------------------
# Mesh cleanup / smoothing utilities
# ------------------------------------------------------------------

def cleanup_triangle_mesh(mesh, keep_largest_component: bool = True):
    """Basic topology cleanup before smoothing."""
    import numpy as np

    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    try:
        mesh.remove_non_manifold_edges()
    except Exception as e:
        print(f"   [WARN] remove_non_manifold_edges failed: {e}")
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()

    if keep_largest_component and len(mesh.triangles) > 0:
        try:
            clusters, counts, areas = mesh.cluster_connected_triangles()
            clusters = np.asarray(clusters)
            counts = np.asarray(counts)
            if len(counts) > 1:
                largest = int(np.argmax(counts))
                remove_mask = clusters != largest
                mesh.remove_triangles_by_mask(remove_mask)
                mesh.remove_unreferenced_vertices()
                print(f"   Kept largest connected component ({counts[largest]:,} triangles)")
        except Exception as e:
            print(f"   [WARN] Connected-component cleanup failed: {e}")

    return mesh


def estimate_voxel_size_from_mesh(mesh, fraction: float = 0.01) -> float:
    bbox = mesh.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    diag = float((extent[0] ** 2 + extent[1] ** 2 + extent[2] ** 2) ** 0.5)
    return max(1e-6, diag * fraction)


def get_boundary_vertex_indices(mesh):
    import numpy as np

    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    if triangles.size == 0:
        return np.array([], dtype=np.int32)

    edge_count = {}
    for tri in triangles:
        a, b, c = map(int, tri)
        for u, v in ((a, b), (b, c), (c, a)):
            if u > v:
                u, v = v, u
            edge_count[(u, v)] = edge_count.get((u, v), 0) + 1

    boundary_vertices = set()
    for (u, v), count in edge_count.items():
        if count == 1:
            boundary_vertices.add(u)
            boundary_vertices.add(v)

    return np.asarray(sorted(boundary_vertices), dtype=np.int32)


def smooth_triangle_mesh(mesh, args):
    """
    Mesh post-processing with cleanup + optional vertex clustering + smoothing.

    Supported methods:
      - laplacian: default, good denoising but can shrink at high iteration counts
      - taubin: reduces shrinkage
      - simple: simple averaging
      - arap: boundary-constrained as-rigid-as-possible smoothing
      - none: cleanup only
    """
    import open3d as o3d
    import numpy as np

    method = getattr(args, "smooth_method", "laplacian")
    iterations = int(getattr(args, "smooth_iterations", 30))
    smooth_lambda = float(getattr(args, "smooth_lambda", 0.8))
    taubin_mu = float(getattr(args, "taubin_mu", -0.53))
    keep_largest = bool(getattr(args, "keep_largest_component", False))
    vertex_cluster_voxel = float(getattr(args, "vertex_cluster_voxel", 0.0))

    print("   Cleaning up mesh topology...")
    mesh = cleanup_triangle_mesh(mesh, keep_largest_component=keep_largest)

    if vertex_cluster_voxel > 0:
        voxel_size = vertex_cluster_voxel
    elif bool(getattr(args, "auto_vertex_cluster", False)):
        voxel_size = estimate_voxel_size_from_mesh(mesh, fraction=float(getattr(args, "auto_vertex_cluster_fraction", 0.003)))
    else:
        voxel_size = 0.0

    if voxel_size > 0:
        print(f"   Vertex clustering (quadric) voxel_size={voxel_size:.6f}...")
        try:
            mesh = mesh.simplify_vertex_clustering(
                voxel_size=voxel_size,
                contraction=o3d.geometry.SimplificationContraction.Quadric,
            )
            mesh = cleanup_triangle_mesh(mesh, keep_largest_component=keep_largest)
        except Exception as e:
            print(f"   [WARN] Vertex clustering failed: {e}")

    if method == "none":
        print("   Smoothing disabled (cleanup only).")
    elif method == "laplacian":
        print(f"   Laplacian smoothing: iterations={iterations}, lambda={smooth_lambda}")
        mesh = mesh.filter_smooth_laplacian(
            number_of_iterations=iterations,
            lambda_filter=smooth_lambda,
        )
    elif method == "simple":
        print(f"   Simple smoothing: iterations={iterations}")
        mesh = mesh.filter_smooth_simple(number_of_iterations=iterations)
    elif method == "taubin":
        print(f"   Taubin smoothing: iterations={iterations}, lambda={smooth_lambda}, mu={taubin_mu}")
        try:
            mesh = mesh.filter_smooth_taubin(
                number_of_iterations=iterations,
                lambda_filter=smooth_lambda,
                mu=taubin_mu,
            )
        except TypeError:
            mesh = mesh.filter_smooth_taubin(number_of_iterations=iterations)
    elif method == "arap":
        print(f"   ARAP smoothing: boundary vertices held fixed, max_iter={getattr(args, 'arap_max_iter', 50)}")
        try:
            original_positions = np.asarray(mesh.vertices).copy()
            boundary_ids = get_boundary_vertex_indices(mesh)
            if len(boundary_ids) == 0:
                print("   [WARN] No boundary vertices found; falling back to Taubin smoothing.")
                try:
                    mesh = mesh.filter_smooth_taubin(
                        number_of_iterations=iterations,
                        lambda_filter=smooth_lambda,
                        mu=taubin_mu,
                    )
                except TypeError:
                    mesh = mesh.filter_smooth_taubin(number_of_iterations=iterations)
            else:
                mesh = mesh.deform_as_rigid_as_possible(
                    constraint_vertex_indices=o3d.utility.IntVector(boundary_ids.tolist()),
                    constraint_vertex_positions=o3d.utility.Vector3dVector(original_positions[boundary_ids]),
                    max_iter=int(getattr(args, "arap_max_iter", 50)),
                )
        except Exception as e:
            print(f"   [WARN] ARAP smoothing failed: {e}")
            print("   Falling back to Taubin smoothing.")
            try:
                mesh = mesh.filter_smooth_taubin(
                    number_of_iterations=iterations,
                    lambda_filter=smooth_lambda,
                    mu=taubin_mu,
                )
            except TypeError:
                mesh = mesh.filter_smooth_taubin(number_of_iterations=iterations)
    else:
        raise ValueError(f"Unknown smooth method: {method}")

    mesh = cleanup_triangle_mesh(mesh, keep_largest_component=keep_largest)
    try:
        mesh.compute_vertex_normals()
    except Exception as e:
        print(f"   [WARN] compute_vertex_normals failed: {e}")
    return mesh


def postprocess_mesh_file(mesh_path: Path, args, suffix: str | None = None) -> Path:
    """Read a mesh file, cleanup/smooth it, and overwrite the same mesh path by default.

    This keeps Stage 1 output names compatible with Stage 2:
      asset/mesh/<stem>.ply
      asset/output/<stem>/scene.glb
      asset/output/<stem>/scene.ply

    Version control can track/recover the pre-smoothed mesh if needed.
    """
    import open3d as o3d

    header(f"Mesh Post-process  -  {mesh_path.name}")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        sys.exit(f"[ERR] Could not read mesh: {mesh_path}")

    print(f"   Input mesh: {len(mesh.vertices):,} vertices  {len(mesh.triangles):,} faces")
    mesh = smooth_triangle_mesh(mesh, args)
    print(f"   Output mesh: {len(mesh.vertices):,} vertices  {len(mesh.triangles):,} faces")

    if suffix:
        out_path = mesh_path.with_name(f"{mesh_path.stem}{suffix}{mesh_path.suffix}")
    else:
        out_path = mesh_path

    ok = o3d.io.write_triangle_mesh(str(out_path), mesh, write_vertex_normals=True, write_vertex_colors=True)
    if not ok:
        sys.exit(f"[ERR] Failed to write smoothed mesh: {out_path}")

    if out_path == mesh_path:
        print(f"[OK]  Post-processed mesh overwritten -> {out_path.relative_to(ROOT)}")
    else:
        print(f"[OK]  Post-processed mesh -> {out_path.relative_to(ROOT)}")

    return out_path


# ------------------------------------------------------------------
# Step 1  -  pixel-perfect-depth  ->  point cloud + fast grid mesh
# ------------------------------------------------------------------

def run_depth(image: Path, args=None) -> tuple[Path, Path | None]:
    """
    Calls pixel-perfect-depth/run_point_cloud.py.
    PPD writes:
      - depth_pcd/<stem>.ply          point cloud
      - depth_pcd/<stem>_camera.json  camera metadata
      - depth_mesh/<stem>.ply         fast image-grid mesh
    We copy them into asset/ so the rest of the pipeline is self-contained.
    """
    header(f"Step 1 / 4  -  Depth -> Point Cloud   [{image.name}]")

    run(
        [
            sys.executable,
            "run_point_cloud.py",   # relative - cwd is pixel-perfect-depth/
            "--img_path",        image,
            "--outdir",          PPD_DIR / "depth_vis",  # depth colourmap (not used further)
            "--semantics_model", "MoGe2",                # metric depth + intrinsics
            "--sampling_steps",  "20",
            "--save_pcd",                                 # emit the .ply
            "--save_mesh",
            "--mesh_stride", str(getattr(args, "mesh_stride", 1)),
            "--max_depth_jump", str(getattr(args, "max_depth_jump", 0.08)),
            "--relative_depth_jump", str(getattr(args, "relative_depth_jump", 0.08)),
            "--max_geometry_depth", str(getattr(args, "max_geometry_depth", 100.0)),
            "--background_percentile", str(getattr(args, "background_percentile", 0.98)),
            "--save_background",
            "--no_depth_vis",
        ],
        cwd=PPD_DIR,                # <- run from inside pixel-perfect-depth/, same as standalone
    )

    # PPD writes to its own subfolders relative to its cwd
    src = PPD_DIR / "depth_pcd" / f"{image.stem}.ply"
    dst = PCD_DIR / f"{image.stem}.ply"
    src_meta = PPD_DIR / "depth_pcd" / f"{image.stem}_camera.json"
    dst_meta = PCD_DIR / f"{image.stem}_camera.json"
    src_mesh = PPD_DIR / "depth_mesh" / f"{image.stem}.ply"
    dst_mesh = MESH_DIR / f"{image.stem}.ply"

    if not src.exists():
        sys.exit(
            f"\n[ERR]  Expected PLY not found at {src}\n"
            "   Check the pixel-perfect-depth output above for errors.\n"
            "   If moge2.pt is missing, see README for download instructions."
        )

    shutil.copy2(src, dst)
    print(f"\n[OK]  Point cloud  ->  {dst.relative_to(ROOT)}")

    if src_meta.exists():
        shutil.copy2(src_meta, dst_meta)
        print(f"[OK]  Camera metadata -> {dst_meta.relative_to(ROOT)}")
    else:
        print(f"[WARN]  Camera metadata not found: {src_meta}")

    fast_mesh: Path | None = None
    if src_mesh.exists():
        shutil.copy2(src_mesh, dst_mesh)
        fast_mesh = dst_mesh
        print(f"[OK]  Fast grid mesh -> {dst_mesh.relative_to(ROOT)}")
    else:
        print(f"[WARN]  Fast grid mesh not found: {src_mesh}")
    # fix
    copy_background_outputs_for_image(image)
    return dst, fast_mesh



def write_image_list(images: list[Path], name: str) -> Path:
    """Write absolute image paths to a temporary txt file for batch subprocesses."""
    list_path = ASSET_DIR / f"_{name}_image_list.txt"
    with open(list_path, "w", encoding="utf-8") as f:
        for img in images:
            f.write(str(img.resolve()) + "\n")
    return list_path


def copy_background_outputs_for_image(image: Path) -> dict[str, Path]:
    """Copy PixelPerfect background/mask outputs into asset/background/."""
    src_dir = PPD_DIR / "background"
    copied: dict[str, Path] = {}
    patterns = {
        "background": f"{image.stem}_background.png",
        "geometry_mask": f"{image.stem}_geometry_mask.png",
        "background_mask": f"{image.stem}_background_mask.png",
        "background_meta": f"{image.stem}_background_meta.json",
    }
    for key, name in patterns.items():
        src = src_dir / name
        dst = BG_DIR / name
        if src.exists():
            shutil.copy2(src, dst)
            copied[key] = dst
        else:
            print(f"[WARN] Background split file not found: {src}")
    if copied:
        pretty = ", ".join(k for k in copied)
        print(f"[OK] Background split -> {BG_DIR.relative_to(ROOT)} ({pretty})")
    return copied


def copy_depth_outputs_for_image(image: Path) -> tuple[Path, Path | None]:
    """Copy PixelPerfect outputs for one image into asset folders."""
    src = PPD_DIR / "depth_pcd" / f"{image.stem}.ply"
    dst = PCD_DIR / f"{image.stem}.ply"
    src_meta = PPD_DIR / "depth_pcd" / f"{image.stem}_camera.json"
    dst_meta = PCD_DIR / f"{image.stem}_camera.json"
    src_mesh = PPD_DIR / "depth_mesh" / f"{image.stem}.ply"
    dst_mesh = MESH_DIR / f"{image.stem}.ply"

    if not src.exists():
        sys.exit(f"\n[ERR] Expected PLY not found at {src}\n  Check the pixel-perfect-depth output above for errors.")

    shutil.copy2(src, dst)

    if src_meta.exists():
        shutil.copy2(src_meta, dst_meta)
    else:
        print(f"[WARN] Camera metadata not found: {src_meta}")

    fast_mesh: Path | None = None
    if src_mesh.exists():
        shutil.copy2(src_mesh, dst_mesh)
        fast_mesh = dst_mesh
    else:
        print(f"[WARN] Fast grid mesh not found: {src_mesh}")

    copy_background_outputs_for_image(image)

    return dst, fast_mesh


def run_depth_batch(
    images: list[Path],
    mesh_stride: int = 1,
    max_depth_jump: float = 0.08,
    relative_depth_jump: float = 0.08,
    max_geometry_depth: float = 100.0,
    background_percentile: float = 0.98,
) -> dict[str, tuple[Path, Path | None]]:
    """Batch PixelPerfect depth for all images in one subprocess."""
    header(f"Step 1 / 4 - Batch PixelPerfect depth  [{len(images)} image(s)]")

    list_path = write_image_list(images, "ppd")
    run(
        [
            sys.executable,
            "run_point_cloud.py",
            "--img_path", list_path,
            "--outdir", PPD_DIR / "depth_vis",
            "--semantics_model", "MoGe2",
            "--sampling_steps", "20",
            "--save_pcd",
            "--save_mesh",
            "--mesh_stride", str(mesh_stride),
            "--max_depth_jump", str(max_depth_jump),
            "--relative_depth_jump", str(relative_depth_jump),
            "--max_geometry_depth", str(max_geometry_depth),
            "--background_percentile", str(background_percentile),
            "--save_background",
            "--no_depth_vis",
        ],
        cwd=PPD_DIR,
    )

    outputs: dict[str, tuple[Path, Path | None]] = {}
    for image in images:
        ply, fast_mesh = copy_depth_outputs_for_image(image)
        outputs[image.stem] = (ply, fast_mesh)
        print(f"[OK] {image.name}: point cloud -> {ply.relative_to(ROOT)}")
        if fast_mesh:
            print(f"  fast mesh -> {fast_mesh.relative_to(ROOT)}")
    return outputs


# ------------------------------------------------------------------
# Step 2  -  Point cloud  ->  mesh  (Poisson surface reconstruction)
# ------------------------------------------------------------------

def run_meshing(ply: Path, depth: int = 9, normal_map: Path | None = None, args=None) -> Path:
    header(f"Step 2 / 4  -  Point Cloud -> Mesh   [{ply.name}]  depth={depth}")

    import open3d as o3d
    import numpy as np

    pcd = o3d.io.read_point_cloud(str(ply))
    print(f"   Loaded {len(pcd.points):,} points")

    # Estimate normals when missing
    if not pcd.has_normals():
        print("   Estimating normals...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(100)

    print("   Running Poisson surface reconstruction...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )

    # Trim low-density floater vertices (bottom 2%)
    densities = np.asarray(densities)
    keep = densities > np.quantile(densities, 0.02)
    mesh.remove_vertices_by_mask(~keep)
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    print(f"   Mesh: {len(mesh.vertices):,} vertices  {len(mesh.triangles):,} faces")

    # Transfer vertex colours from the point cloud (nearest-neighbour)
    if pcd.has_colors():
        tree = o3d.geometry.KDTreeFlann(pcd)
        verts = np.asarray(mesh.vertices)
        colors = np.zeros((len(verts), 3))
        for i, v in enumerate(verts):
            _, idx, _ = tree.search_knn_vector_3d(v, 1)
            colors[i] = np.asarray(pcd.colors)[idx[0]]
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    # -- Override vertex normals from Ouroboros normal map --------------------
    # The Poisson mesh normals are derived from the coarse geometry, which
    # causes staircase shading. The Ouroboros normal map encodes the true
    # high-frequency surface detail. We project each vertex to image-space,
    # sample the normal map, decode RGB->[-1,1] normals, and write them back.
    # This makes the blocky mesh shade smoothly without changing its geometry.
    if normal_map and normal_map.exists():
        print("   Applying Ouroboros normal map to vertex normals...")
        from PIL import Image as PILImage
        nimg = PILImage.open(normal_map).convert("RGB")
        W, H = nimg.size
        nmap = np.array(nimg).astype(np.float32) / 255.0  # (H, W, 3) in [0,1]

        verts = np.asarray(mesh.vertices)

        # Planar projection: same mapping used for albedo in run_combine
        x = verts[:, 0]
        y = verts[:, 1]
        x_n = (x - x.min()) / (x.max() - x.min() + 1e-9)
        y_n = (y - y.min()) / (y.max() - y.min() + 1e-9)
        px = np.clip((x_n * (W - 1)).astype(int), 0, W - 1)
        py = np.clip(((1 - y_n) * (H - 1)).astype(int), 0, H - 1)  # flip Y

        # Decode: RGB [0,1] -> normal [-1,1]  (standard tangent-space convention)
        sampled = nmap[py, px]                   # (N, 3)
        normals = sampled * 2.0 - 1.0            # (N, 3) in [-1, 1]

        # Normalise (guard against zero vectors from flat/clipped regions)
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        lengths = np.where(lengths < 1e-6, 1.0, lengths)
        normals = normals / lengths

        mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
        print(f"   Normal map applied  ({W}x{H}  ->  {len(verts):,} vertex normals)")
    else:
        # Fall back to smooth geometry-based normals (better than face normals)
        mesh.compute_vertex_normals()
        print("   Using geometry-based smooth vertex normals (no normal map supplied)")

    # Optional cleanup / smoothing after Poisson reconstruction.
    if args is not None and getattr(args, "smooth_mesh", True):
        print("   Applying mesh post-processing to Poisson mesh...")
        mesh = smooth_triangle_mesh(mesh, args)

    # Save as PLY - OBJ does not support vertex colors, they would be silently
    # dropped on load. PLY preserves them natively.
    mesh_path = MESH_DIR / f"{ply.stem}.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), mesh, write_vertex_colors=True)
    print(f"\n[OK]  Mesh  ->  {mesh_path.relative_to(ROOT)}")
    return mesh_path


# ------------------------------------------------------------------
# Step 3  -  MVInverse  ->  PBR material maps
# ------------------------------------------------------------------

# Staging dir for MVInverse inputs and raw outputs. MVInverse expects a
# directory of images (it is multi-view), so for single-image stage 1 we
# create a per-stem subfolder containing just that one image.
MVI_INPUT_STAGE  = MAT_DIR / "_mvi_input"
MVI_OUTPUT_STAGE = MAT_DIR / "_mvi_raw"


def ensure_mvi_single_runner_script() -> Path:
    """Create a local MVInverse runner that loads the model once, then processes
    unrelated images one-by-one without passing them as a multi-view sequence.
    """
    runner_path = MVI_DIR / "single_image_batch_inference.py"
    runner_code = 'import argparse\nimport os\nfrom pathlib import Path\n\nimport numpy as np\nimport torch\nfrom PIL import Image\nfrom torchvision import transforms\n\nfrom mvinverse.models.mvinverse import MVInverse\n\n\ndef load_model(ckpt_path_or_repo_id, device):\n    if os.path.exists(ckpt_path_or_repo_id):\n        print(f"Loading model from local checkpoint: {ckpt_path_or_repo_id}")\n        model = MVInverse().to(device).eval()\n        weight = torch.load(ckpt_path_or_repo_id, map_location=device, weights_only=False)\n        weight = weight["model"] if "model" in weight else weight\n        missing, unused = model.load_state_dict(weight, strict=False)\n        if len(missing) != 0:\n            print(f"Warning: Missing keys found: {missing}")\n        if len(unused) != 0:\n            print(f"Got unexpected keys: {unused}")\n    else:\n        print(f"Checkpoint not found locally. Attempting to download from Hugging Face Hub: {ckpt_path_or_repo_id}")\n        model = MVInverse.from_pretrained(ckpt_path_or_repo_id)\n        model = model.to(device).eval()\n    return model\n\n\ndef load_single_image(image_path: Path) -> torch.Tensor:\n    with Image.open(image_path) as img:\n        img = img.convert("RGB")\n        width, height = img.size\n\n        # Match upstream MVInverse inference.py behavior.\n        if max(width, height) > 1024:\n            scaling_factor = 1024 / max(width, height)\n            new_width, new_height = int(width * scaling_factor), int(height * scaling_factor)\n        else:\n            new_width, new_height = width, height\n\n        # MVInverse requires dimensions divisible by 14.\n        new_w = max(14, new_width // 14 * 14)\n        new_h = max(14, new_height // 14 * 14)\n        img = img.resize((new_w, new_h))\n\n    return transforms.ToTensor()(img)\n\n\ndef save_one_result(res, save_dir: Path):\n    save_dir.mkdir(parents=True, exist_ok=True)\n\n    albedo = res["albedo"][0, 0]\n    metallic = res["metallic"][0, 0]\n    roughness = res["roughness"][0, 0]\n    normal = res["normal"][0, 0]\n    shading = res["shading"][0, 0]\n\n    Image.fromarray((albedo.cpu().numpy() * 255).astype(np.uint8)).save(save_dir / "000_albedo.png")\n    Image.fromarray((metallic.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)).save(save_dir / "000_metallic.png")\n    Image.fromarray((roughness.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)).save(save_dir / "000_roughness.png")\n    Image.fromarray(((normal * 0.5 + 0.5).cpu().numpy() * 255).astype(np.uint8)).save(save_dir / "000_normal.png")\n    Image.fromarray((shading.cpu().numpy() * 255).astype(np.uint8)).save(save_dir / "000_shading.png")\n\n\ndef main():\n    parser = argparse.ArgumentParser(\n        description="MVInverse single-image batch runner: load model once, process unrelated images independently."\n    )\n    parser.add_argument("--image_list", required=True, help="Text file containing one absolute image path per line.")\n    parser.add_argument("--ckpt", default="maddog241/mvinverse")\n    parser.add_argument("--save_path", default="outputs")\n    parser.add_argument("--device", default="cuda")\n    args = parser.parse_args()\n\n    with open(args.image_list, "r", encoding="utf-8") as f:\n        image_paths = [Path(line.strip()) for line in f if line.strip()]\n\n    if not image_paths:\n        raise RuntimeError(f"No images listed in {args.image_list}")\n\n    print("Loading model once...")\n    device = torch.device(args.device)\n    model = load_model(args.ckpt, device)\n\n    save_root = Path(args.save_path)\n    use_cuda_amp = device.type == "cuda"\n\n    for idx, image_path in enumerate(image_paths, start=1):\n        print("-" * 70)\n        print(f"[{idx}/{len(image_paths)}] Single-image MVInverse: {image_path.name}")\n        img = load_single_image(image_path).unsqueeze(0).to(device)   # [1, 3, H, W]\n        print(f"Loaded 1 image, size is {tuple(img.shape[-2:])}")\n\n        with torch.no_grad():\n            if use_cuda_amp:\n                with torch.amp.autocast("cuda", dtype=torch.float16):\n                    res = model(img[None])  # [B=1, N=1, C, H, W]\n            else:\n                res = model(img[None])\n\n        out_dir = save_root / image_path.stem\n        save_one_result(res, out_dir)\n        print(f"Results saved to {out_dir}")\n\n    print("Done. MVInverse model was loaded once for all unrelated images.")\n\n\nif __name__ == "__main__":\n    main()\n'
    runner_path.write_text(runner_code, encoding="utf-8")
    return runner_path


def _stage_mvi_inputs(images: list[Path], group_name: str) -> Path:
    """Copy the given images into a flat staging dir for MVInverse.

    MVInverse derives its output subfolder name from `os.path.basename(data_path)`,
    so we use a deterministic group name to know exactly where outputs land.
    """
    stage = MVI_INPUT_STAGE / group_name
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True, exist_ok=True)
    for img in images:
        shutil.copy2(img, stage / img.name)
    return stage


def _harvest_mvi_outputs(raw_dir: Path, image_stems: list[Path | str]) -> dict[str, dict[str, Path]]:
    """Map MVInverse raw outputs (000_albedo.png, 000_normal.png, ...) to per-stem
    material folders MAT_DIR/<stem>/{albedo,normals,roughness,metallicity,irradiance}.png

    `image_stems` must match the alphabetical order MVInverse used to enumerate
    its input directory (it sorts filenames internally).
    """
    outputs: dict[str, dict[str, Path]] = {}
    for idx, stem_or_path in enumerate(image_stems):
        stem = Path(stem_or_path).stem if isinstance(stem_or_path, (Path, str)) else str(stem_or_path)
        stem_dir = MAT_DIR / stem
        stem_dir.mkdir(parents=True, exist_ok=True)
        per_stem: dict[str, Path] = {}
        for our_key, mvi_name in MVI_OUTPUT_NAME_MAP.items():
            src = raw_dir / f"{idx:03d}_{mvi_name}.png"
            if not src.exists():
                print(f"   [WARN]  MVInverse output not found for '{our_key}': {src.name}")
                continue
            dst = stem_dir / f"{our_key}.png"
            shutil.copy2(src, dst)
            per_stem[our_key] = dst
        outputs[stem] = per_stem
    return outputs


def run_ouroboros(image: Path) -> dict[str, Path]:
    """
    Run MVInverse on a single image and return a dict keyed by modality.

    Uses mvinverse/single_image_batch_inference.py instead of inference.py.
    This is safer for the current repo layout and matches the production plan:
    unrelated images are processed as N=1 inputs, not as one multi-view sequence.
    """
    header(f"Step 3 / 4  -  MVInverse intrinsic decomposition   [{image.name}]")

    infer_script = MVI_DIR / "single_image_batch_inference.py"
    if not infer_script.exists():
        sys.exit(
            f"\n[ERR] MVInverse single-image runner not found at:\n"
            f"   {infer_script}\n"
            f"Expected repo layout:\n"
            f"   <repo>/mvinverse/single_image_batch_inference.py\n"
        )

    image_list = write_image_list([image], f"mvi_single_{image.stem}")

    raw_root = MVI_OUTPUT_STAGE
    raw_root.mkdir(parents=True, exist_ok=True)

    raw_subdir = raw_root / image.stem
    if raw_subdir.exists():
        shutil.rmtree(raw_subdir)

    run(
        [
            sys.executable,
            infer_script.name,
            "--image_list", image_list,
            "--save_path", raw_root,
            "--ckpt", MVI_CKPT,
            "--device", "cuda",
        ],
        cwd=MVI_DIR,
    )

    harvested = _harvest_mvi_outputs(raw_subdir, [image])
    maps = harvested.get(image.stem, {})

    if maps:
        print(f"\n[OK] Material maps: {', '.join(maps.keys())}")
    else:
        print(f"   [ERR] No MVInverse outputs harvested under {raw_subdir}")

    return maps


def collect_ouro_maps_for_stem(stem: str) -> dict[str, Path]:
    """Find previously-generated intrinsic maps under asset/materials/<stem>/.

    Function name kept for back-compat. Looks for the canonical filenames
    written by `_harvest_mvi_outputs` first, then falls back to the keyword
    scan so old Ouroboros runs are still recognised.
    """
    stem_mat_dir = MAT_DIR / stem
    if not stem_mat_dir.exists():
        return {}
    maps: dict[str, Path] = {}
    # Preferred: exact filenames produced by the new MVInverse harvester.
    for mod in MODALITIES:
        candidate = stem_mat_dir / f"{mod}.png"
        if candidate.exists():
            maps[mod] = candidate
    if maps:
        return maps
    # Fallback: keyword scan across all PNGs (covers legacy Ouroboros layouts
    # and MVInverse raw 000_*.png outputs).
    all_pngs = list(stem_mat_dir.rglob("*.png"))
    aliases = {
        "normals":     ("normals", "normal"),
        "albedo":      ("albedo",),
        "irradiance":  ("irradiance", "shading"),
        "roughness":   ("roughness",),
        "metallicity": ("metallicity", "metallic"),
    }
    for mod, words in aliases.items():
        for p in all_pngs:
            name = p.name.lower()
            if any(w in name for w in words):
                maps[mod] = p
                break
    return maps


def run_ouroboros_batch(images: list[Path]) -> dict[str, dict[str, Path]]:
    """Run MVInverse for many unrelated images without cross-image mixing.

    Important:
      - We do NOT pass all images as one MVInverse multi-view sequence.
      - The helper process loads the model once, then runs each image as
        a separate N=1 inference call.
      - Output layout remains compatible with `_harvest_mvi_outputs`:
          asset/materials/_mvi_raw/<stem>/000_albedo.png, ...
    """
    header(f"Step 3 / 4 - MVInverse single-image batch  [{len(images)} unrelated image(s)]")

    infer_script = ensure_mvi_single_runner_script()
    image_list = write_image_list(images, "mvi_unrelated")

    raw_root = MVI_OUTPUT_STAGE
    raw_root.mkdir(parents=True, exist_ok=True)

    # Clear only the raw output folders for the images being processed.
    for image in images:
        raw_subdir = raw_root / image.stem
        if raw_subdir.exists():
            shutil.rmtree(raw_subdir)

    run(
        [
            sys.executable,
            infer_script.name,
            "--image_list", image_list,
            "--save_path", raw_root,
            "--ckpt", MVI_CKPT,
            "--device", "cuda",
        ],
        cwd=MVI_DIR,
    )

    outputs: dict[str, dict[str, Path]] = {}
    for image in images:
        raw_subdir = raw_root / image.stem
        harvested = _harvest_mvi_outputs(raw_subdir, [image])
        maps = harvested.get(image.stem, {})
        outputs[image.stem] = maps
        if maps:
            print(f"[OK] {image.name}: maps -> {', '.join(maps)}")
        else:
            print(f"[WARN] {image.name}: no MVInverse maps found under {raw_subdir}")

    return outputs


# ------------------------------------------------------------------
# Step 4  -  Combine mesh + PBR maps  ->  .glb
# ------------------------------------------------------------------

def run_combine(mesh_obj: Path, maps: dict[str, Path], reference_image: Path | None = None, args=None) -> Path:
    """
    Combines mesh + Ouroboros AOV maps into a .glb.

    Current stage-1 representation:
      - geometry: PixelPerfect fast image-grid mesh, or optional Poisson mesh
      - base color: Ouroboros albedo projected to vertex COLOR_0
      - shading normals: Ouroboros camera-space normals projected to vertex normals
      - roughness / metallicity / irradiance / normal maps: copied as sidecar files
      - Mitsuba-ready PLY mesh, Mitsuba scene seed, reference/masks, and optimizer config

    Note: the mesh still has no UVs, so roughness/metallic/normal PNGs are not
    embedded as true glTF texture slots yet. They are sidecar assets for stage 2.
    """
    header(f"Step 4 / 4  -  Combine -> GLB   [{mesh_obj.name}]")
    source_stem = original_stem_from_mesh(mesh_obj)

    try:
        import json
        import trimesh
        import numpy as np
        from PIL import Image as PILImage
    except ImportError:
        sys.exit(
            "\n[ERR]  Missing packages: trimesh / pillow\n"
            "   Fix:  uv add trimesh pillow pyglet"
        )

    # -- Load mesh ------------------------------------------------------------
    scene_or_mesh = trimesh.load(str(mesh_obj), process=False)
    if isinstance(scene_or_mesh, trimesh.Scene):
        geoms = list(scene_or_mesh.geometry.values())
        if not geoms:
            sys.exit("[ERR]  No geometry found in the mesh file.")
        mesh = geoms[0]
    else:
        mesh = scene_or_mesh

    print(f"   Mesh: {len(mesh.vertices):,} vertices")

    # -- Load PixelPerfect camera metadata ------------------------------------
    # The fast mesh / point cloud was exported after applying [1,-1,-1].
    # To project image-space Ouroboros maps back to vertices, undo that flip,
    # project with the same intrinsics, sample maps, then re-apply the flip to
    # normals so they live in the exported mesh coordinate system.
    meta_path = PCD_DIR / f"{source_stem}_camera.json"
    if not meta_path.exists():
        print(f"   [WARN]  Camera metadata not found: {meta_path}")
        print("   [WARN]  Cannot camera-project Ouro albedo/normals; keeping mesh colors/normals as-is")
    else:
        with open(meta_path, "r") as f:
            meta = json.load(f)

        K = np.asarray(meta["intrinsic"], dtype=np.float32)
        resize_W = int(meta["resize_W"])
        resize_H = int(meta["resize_H"])
        flip = np.asarray(meta.get("saved_ply_flip", [1.0, -1.0, -1.0]), dtype=np.float32)

        verts_saved = np.asarray(mesh.vertices, dtype=np.float32)
        verts_cam = verts_saved / flip[None, :]  # undo saved PLY flip

        x = verts_cam[:, 0]
        y = verts_cam[:, 1]
        z = verts_cam[:, 2]
        valid = z > 1e-6

        u = K[0, 0] * x / z + K[0, 2]
        v = K[1, 1] * y / z + K[1, 2]

        inside = (
            valid &
            (u >= 0) & (u <= resize_W - 1) &
            (v >= 0) & (v <= resize_H - 1)
        )

        px = np.clip(np.round(u).astype(np.int32), 0, resize_W - 1)
        py = np.clip(np.round(v).astype(np.int32), 0, resize_H - 1)
        # fix calculat uv
        # -- Image-space UVs for MoGe single-view mesh ------------------------
        # Because the mesh is reconstructed from the input view, the correct
        # first UV unwrap is just the camera projection:
        #   U = projected_pixel_x / image_width
        #   V = 1 - projected_pixel_y / image_height
        # Store it on the mesh object so we can export scene_uv.obj later.
        uv = np.zeros((len(verts_saved), 2), dtype=np.float32)
        uv[:, 0] = np.clip(u / max(1, resize_W - 1), 0.0, 1.0)
        uv[:, 1] = np.clip(1.0 - (v / max(1, resize_H - 1)), 0.0, 1.0)
        mesh.metadata["image_space_uv"] = uv
        
        
        # -- Override vertex colors from Ouroboros albedo ---------------------
        if maps.get("albedo") and maps["albedo"].exists():
            print("   Projecting Ouroboros albedo to vertex colors...")

            albedo_img = PILImage.open(maps["albedo"]).convert("RGB")
            albedo_img = albedo_img.resize((resize_W, resize_H), PILImage.BILINEAR)
            alb = np.asarray(albedo_img, dtype=np.uint8)

            vc = np.zeros((len(verts_saved), 3), dtype=np.uint8)
            vc[inside] = alb[py[inside], px[inside]]

            # Preserve old colors for vertices that project outside the image.
            try:
                old_vc = mesh.visual.vertex_colors
                if old_vc is not None and len(old_vc) == len(verts_saved):
                    vc[~inside] = np.asarray(old_vc)[~inside, :3]
            except Exception:
                pass

            alpha = np.full((len(vc), 1), 255, dtype=np.uint8)
            mesh.visual = trimesh.visual.ColorVisuals(
                mesh=mesh,
                vertex_colors=np.concatenate([vc, alpha], axis=1)
            )
            print(f"   [OK] using Ouro albedo as COLOR_0 ({inside.sum():,}/{len(verts_saved):,} vertices projected)")
        else:
            print("   [WARN]  No Ouro albedo found; keeping existing vertex colors")

        # -- Override vertex normals from Ouroboros normals -------------------
        # Ouroboros prompt says camera-space normals. Decode RGB [0,1] -> [-1,1],
        # sample by the same camera projection used for albedo, normalize, then
        # apply the PixelPerfect export flip so normals match the GLB mesh axes.
        if maps.get("normals") and maps["normals"].exists():
            print("   Projecting Ouroboros normal map to vertex normals...")

            nimg = PILImage.open(maps["normals"]).convert("RGB")
            nimg = nimg.resize((resize_W, resize_H), PILImage.BILINEAR)
            nmap = np.asarray(nimg, dtype=np.float32) / 255.0

            sampled = nmap[py[inside], px[inside]]
            sampled = sampled * 2.0 - 1.0

            normals = np.zeros((len(verts_saved), 3), dtype=np.float32)
            normals[inside] = sampled

            # For unprojected vertices, fall back to geometry normals if available.
            try:
                geom_normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
                if geom_normals.shape == normals.shape:
                    normals[~inside] = geom_normals[~inside]
            except Exception:
                pass

            # Match exported coordinate convention.
            normals *= flip[None, :]

            lengths = np.linalg.norm(normals, axis=1, keepdims=True)
            bad = lengths[:, 0] < 1e-8
            if np.any(bad):
                normals[bad] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                lengths[bad] = 1.0
            normals = normals / lengths

            # Trimesh may not expose a normal setter in every version, but glTF
            # export reads mesh.vertex_normals. Setting both paths makes this robust.
            try:
                mesh.vertex_normals = normals
            except Exception:
                pass
            try:
                mesh._cache["vertex_normals"] = normals
            except Exception:
                pass

            print(f"   [OK] using Ouro normals for smooth shading ({inside.sum():,}/{len(verts_saved):,} vertices projected)")
        else:
            print("   [WARN]  No Ouro normals found; keeping geometry normals")

    # -- Export self-contained stage-1 package --------------------------------
    # Each image gets its own folder:
    #   asset/output/<stem>/
    #     scene.glb
    #     camera.json
    #     manifest.json
    #     albedo.png / roughness.png / metallic.png / normal.png / irradiance.png
    #     background.png / geometry_mask.png / background_mask.png / background_meta.json
    package_dir = OUTPUT_DIR / source_stem
    package_dir.mkdir(parents=True, exist_ok=True)

    glb_path = package_dir / "scene.glb"
    mesh.export(str(glb_path))
    size_mb = glb_path.stat().st_size / 1024 / 1024
    print(f"\n[OK]  GLB  ->  {glb_path.relative_to(ROOT)}  ({size_mb:.1f} MB)")
    # fix: export a uv mesh
    # Also export a UV OBJ. This is the important path for bitmap roughness /
    # metallic / albedo in Mitsuba, because PLY is reliable for vertex colors
    # but not a good carrier for ordinary image texture coordinates.
    uv_obj_path = None
    try:
        uv = mesh.metadata.get("image_space_uv")
        if uv is not None and maps.get("albedo") and maps["albedo"].exists():
            tex_img = PILImage.open(maps["albedo"]).convert("RGB")
            tex_mesh = mesh.copy()
            tex_mesh.visual = trimesh.visual.TextureVisuals(
                uv=uv,
                image=tex_img,
                material=trimesh.visual.material.SimpleMaterial(
                    image=tex_img,
                    diffuse=[255, 255, 255, 255],
                ),
            )
            uv_obj_path = package_dir / "scene_uv.obj"
            tex_mesh.export(str(uv_obj_path))
            print(f"   UV textured OBJ -> {uv_obj_path.relative_to(ROOT)}")
    except Exception as e:
        print(f"   [WARN] Could not export UV textured OBJ: {e}")
        uv_obj_path = None
    
    # Mitsuba prefers its mesh loaders (PLY/OBJ/serialized) for stage-2 rendering.
    # Keep GLB for Blender/preview, and export PLY for Mitsuba.
    ply_path = package_dir / "scene.ply"
    try:
        mesh.export(str(ply_path))
        print(f"   Mitsuba mesh -> {ply_path.relative_to(ROOT)}")
    except Exception as e:
        print(f"   [WARN] Could not export scene.ply for Mitsuba: {e}")
        ply_path = None

    # -- Copy camera metadata beside GLB --------------------------------------
    camera_file = None
    cam_src = PCD_DIR / f"{source_stem}_camera.json"
    if cam_src.exists():
        cam_dst = package_dir / "camera.json"
        shutil.copy2(cam_src, cam_dst)
        camera_file = cam_dst.name
        print(f"   Camera metadata -> {cam_dst.relative_to(ROOT)}")
    else:
        print(f"   [WARN]  Camera metadata not found: {cam_src}")

    # -- Copy original input image as reference.png for render/loss target -----
    reference_file = None
    ref_src = reference_image if reference_image and reference_image.exists() else None
    if ref_src is None:
        for ext in IMG_EXTS:
            candidate = INPUT_DIR / f"{source_stem}{ext}"
            if candidate.exists():
                ref_src = candidate
                break
    if ref_src is not None:
        try:
            ref_img = PILImage.open(ref_src).convert("RGB")
            ref_dst = package_dir / "reference.png"
            ref_img.save(ref_dst)
            reference_file = ref_dst.name
            print(f"   Reference image -> {ref_dst.relative_to(ROOT)}")
        except Exception as e:
            print(f"   [WARN] Could not copy reference image {ref_src}: {e}")
    else:
        print("   [WARN] Reference image not found; stage-2 loss target will be missing")

    # -- Copy PBR sidecar maps into package -----------------------------------
    sidecars = {}
    sidecar_name = {
        "albedo": "albedo",
        "roughness": "roughness",
        "metallicity": "metallic",
        "normals": "normal",
        "irradiance": "irradiance",
    }

    for mod, src in maps.items():
        if not src.exists():
            continue
        clean = sidecar_name.get(mod, mod)
        dst = package_dir / f"{clean}.png"
        shutil.copy2(src, dst)
        sidecars[clean] = dst.name

    # -- Copy PixelPerfect geometry/background split into package -------------
    background_files = {}
    bg_source_names = {
        "background_image": f"{source_stem}_background.png",
        "geometry_mask": f"{source_stem}_geometry_mask.png",
        "background_mask": f"{source_stem}_background_mask.png",
        "background_meta": f"{source_stem}_background_meta.json",
    }
    bg_output_names = {
        "background_image": "background.png",
        "geometry_mask": "geometry_mask.png",
        "background_mask": "background_mask.png",
        "background_meta": "background_meta.json",
    }
    for key, source_name in bg_source_names.items():
        src = BG_DIR / source_name
        if src.exists():
            dst = package_dir / bg_output_names[key]
            shutil.copy2(src, dst)
            background_files[key] = dst.name

    # -- Mitsuba scene seed + optimization config -----------------------------
    # These are JSON recipes for stage 2. They avoid hard-coding paths later.
    mitsuba_scene_file = None
    optimize_config_file = None

    sensor = {
        "type": "perspective",
        "camera_metadata": camera_file,
        "resolution": None,
        "intrinsics": None,
        "coordinate_note": "Mesh is already exported in pixelperfect_export_flip_[1,-1,-1] coordinates.",
    }
    if camera_file is not None:
        try:
            with open(package_dir / camera_file, "r") as f:
                camera_meta = json.load(f)
            K_meta = camera_meta.get("intrinsic")
            if K_meta is not None:
                sensor["intrinsics"] = {
                    "fx": K_meta[0][0],
                    "fy": K_meta[1][1],
                    "cx": K_meta[0][2],
                    "cy": K_meta[1][2],
                }
            sensor["resolution"] = {
                "width": camera_meta.get("resize_W"),
                "height": camera_meta.get("resize_H"),
            }
            if K_meta is not None and camera_meta.get("resize_W"):
                import math
                fx = float(K_meta[0][0])
                width = float(camera_meta.get("resize_W"))
                if fx > 0:
                    sensor["fov_x_degrees"] = float(2.0 * math.degrees(math.atan(width / (2.0 * fx))))
        except Exception as e:
            print(f"   [WARN] Could not read camera metadata for Mitsuba scene seed: {e}")

    mitsuba_scene = {
        "schema": "mitsuba_stage2_scene_seed_v1",
        "name": source_stem,
        "preferred_mesh": "scene.ply" if ply_path is not None else "scene.glb",
        "mesh_shape": {
            "type": "ply" if ply_path is not None else "external",
            "filename": "scene.ply" if ply_path is not None else "scene.glb",
            "face_normals": False,
        },
        "sensor": sensor,
        "integrator": {
            "type": "prb",
            "max_depth": 8,
        },
        "emitter_initialization": {
            "type": "constant",
            "radiance": [1.0, 1.0, 1.0],
            "note": "Use irradiance.png only as a weak lighting prior/initialization, not as ground-truth lighting.",
        },
        "bsdf_initialization": {
            "type": "diffuse_or_principled_placeholder",
            "base_color": "vertex_color_from_ouro_albedo",
            "roughness_sidecar": sidecars.get("roughness"),
            "metallic_sidecar": sidecars.get("metallic"),
            "normal_sidecar": sidecars.get("normal"),
            "has_uv": False,
            "note": "No UVs yet; sidecar SVBRDF maps are for later baking or image-space losses.",
        },
        "targets": {
            "reference": reference_file,
            "geometry_mask": background_files.get("geometry_mask"),
            "background": background_files.get("background_image"),
            "background_mask": background_files.get("background_mask"),
        },
        "coordinate_system": "pixelperfect_export_flip_[1,-1,-1]",
    }
    mitsuba_scene_path = package_dir / "scene_mitsuba.json"
    with open(mitsuba_scene_path, "w") as f:
        json.dump(mitsuba_scene, f, indent=2)
    mitsuba_scene_file = mitsuba_scene_path.name

    optimize_config = {
        "schema": "stage2_optimization_config_v1",
        "loss": {
            "target": reference_file,
            "mask": background_files.get("geometry_mask"),
            "background_composite": background_files.get("background_image"),
            "use_l1": True,
            "use_multiscale": False,
        },
        "optimize": {
            "env_radiance": True,
            "key_light": False,
            "camera_pose": False,
            "mesh_vertices": False,
            "vertex_albedo": False,
            "roughness_scalar_or_texture": False,
            "background": False,
        },
        "initial_values": {
            "env_radiance": [1.0, 1.0, 1.0],
            "samples_per_pixel": 64,
        },
        "notes": [
            "Start by optimizing lighting only; keep geometry/material fixed.",
            "Use geometry_mask for mesh-render loss; composite or ignore background pixels.",
            "irradiance.png is a learned prior and should not be treated as ground truth.",
        ],
    }
    optimize_config_path = package_dir / "optimize_config.json"
    with open(optimize_config_path, "w") as f:
        json.dump(optimize_config, f, indent=2)
    optimize_config_file = optimize_config_path.name

    # -- Manifest for stage 2 / Blender / Mitsuba -----------------------------
    manifest = {
        "schema": "stage1_reconstruction_package_v1",
        "name": source_stem,
        "mesh": glb_path.name,
        "mitsuba_mesh": "scene.ply" if ply_path is not None else None,
        "camera_metadata": camera_file,
        "reference_image": reference_file,
        "mitsuba_scene": mitsuba_scene_file,
        "optimize_config": optimize_config_file,
        "albedo_mode": "vertex_color_from_ouro_albedo" if maps.get("albedo") else "mesh_existing_vertex_color",
        "normal_mode": "vertex_normals_from_ouro_normals" if maps.get("normals") else "geometry_normals",
        "has_uv": False,
        "has_background": bool(background_files),
        "background_files": background_files,
        "background_mode": "camera_background_layer_from_far_depth_split" if background_files else "none",
        "sidecar_maps": sidecars,
        "coordinate_system": "pixelperfect_export_flip_[1,-1,-1]",
        "recommended_stage2_use": {
            "render_mesh_blender": "scene.glb",
            "render_mesh_mitsuba": "scene.ply" if ply_path is not None else None,
            "base_color": "vertex COLOR_0 from Ouro albedo",
            "composite_background": background_files.get("background_image"),
            "geometry_mask": background_files.get("geometry_mask"),
            "background_mask": background_files.get("background_mask"),
        },
        "note": "Self-contained package for Blender + Mitsuba stage 2. GLB is for Blender/preview; PLY + scene_mitsuba.json is the preferred Mitsuba entry. No UVs yet; roughness/metallic/normal/irradiance are sidecar maps for baking, image-space losses, or later SVBRDF refinement. Background files define non-geometry far/sky layer for compositing/rendering.",
    }
    manifest_path = package_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    if sidecars:
        print(f"   PBR sidecar maps copied -> {package_dir.relative_to(ROOT)}/")
    if background_files:
        print(f"   Background/mask files copied -> {package_dir.relative_to(ROOT)}/")
    print(f"   Mitsuba scene seed -> {(package_dir / mitsuba_scene_file).relative_to(ROOT)}")
    print(f"   Optimization config -> {(package_dir / optimize_config_file).relative_to(ROOT)}")
    print(f"   Manifest -> {manifest_path.relative_to(ROOT)}")

    return glb_path



# ------------------------------------------------------------------
# Production cleanup utilities
# ------------------------------------------------------------------

def safe_remove_path(path: Path, root_guard: Path = ROOT) -> bool:
    """Remove one file/folder if it exists, with a simple project-root guard."""
    try:
        path = Path(path)
        if not path.exists():
            return False
        resolved = path.resolve()
        guard = root_guard.resolve()
        if resolved == guard or guard not in resolved.parents:
            print(f"   [WARN] Refusing to delete outside project root: {resolved}")
            return False
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        return True
    except Exception as e:
        print(f"   [WARN] Could not remove {path}: {e}")
        return False


def cleanup_intermediates_for_stem(stem: str, args=None):
    """Delete per-image intermediate files after the final output package exists.

    Production rule:
      - asset/output/<stem>/ is the artifact to keep.
      - asset/pointcloud, asset/mesh, asset/materials, asset/background are work/cache copies.
      - pixel-perfect-depth/depth_* and MVInverse staging/raw folders are also work/cache copies.

    This prevents the same map/mesh/camera/mask from living in both asset/* work
    folders and the final compact package.
    """
    if args is not None and not getattr(args, "cleanup_intermediates", True):
        return

    package_dir = OUTPUT_DIR / stem
    manifest = package_dir / "manifest.json"
    if not manifest.exists():
        print(f"   [WARN] Cleanup skipped for {stem}: final manifest does not exist yet")
        return

    header(f"Cleanup intermediates  -  {stem}")
    candidates: list[Path] = []

    # Asset-level duplicate/work copies.
    candidates += [
        PCD_DIR / f"{stem}.ply",
        PCD_DIR / f"{stem}_camera.json",
        MESH_DIR / f"{stem}.ply",
        MESH_DIR / f"{stem}.obj",
        MESH_DIR / f"{stem}.glb",
        MAT_DIR / stem,
    ]

    for suffix in [
        "_background.png",
        "_geometry_mask.png",
        "_background_mask.png",
        "_background_meta.json",
    ]:
        candidates.append(BG_DIR / f"{stem}{suffix}")

    # PixelPerfect internal outputs copied into asset/ or final package.
    candidates += [
        PPD_DIR / "depth_pcd" / f"{stem}.ply",
        PPD_DIR / "depth_pcd" / f"{stem}_camera.json",
        PPD_DIR / "depth_mesh" / f"{stem}.ply",
        PPD_DIR / "depth_vis" / f"{stem}.png",
        PPD_DIR / "depth_npy" / f"{stem}.npy",
    ]
    for suffix in [
        "_background.png",
        "_geometry_mask.png",
        "_background_mask.png",
        "_background_meta.json",
    ]:
        candidates.append(PPD_DIR / "background" / f"{stem}{suffix}")

    # MVInverse staging/raw outputs. Canonical maps were already copied into package.
    candidates += [
        MVI_INPUT_STAGE / stem,
        MVI_OUTPUT_STAGE / stem,
    ]

    removed = []
    for path in candidates:
        if safe_remove_path(path):
            removed.append(path)

    # Remove temporary batch list files when no longer needed. These are global,
    # so it is okay if the first completed image removes them after subprocesses finished.
    for list_file in [ASSET_DIR / "_ppd_image_list.txt", ASSET_DIR / "_mvi_unrelated_image_list.txt"]:
        if safe_remove_path(list_file):
            removed.append(list_file)

    if removed:
        print(f"   Removed {len(removed)} intermediate file/folder(s).")
    else:
        print("   Nothing to remove.")


def cleanup_empty_work_dirs(args=None):
    """Remove empty cache/staging directories after per-file cleanup."""
    if args is not None and not getattr(args, "cleanup_intermediates", True):
        return
    for d in [
        PPD_DIR / "depth_pcd",
        PPD_DIR / "depth_mesh",
        PPD_DIR / "depth_vis",
        PPD_DIR / "depth_npy",
        PPD_DIR / "background",
        MVI_INPUT_STAGE,
        MVI_OUTPUT_STAGE,
        PCD_DIR,
        MESH_DIR,
        BG_DIR,
    ]:
        try:
            if d.exists() and d.is_dir() and not any(d.iterdir()):
                d.rmdir()
        except Exception:
            pass

# ------------------------------------------------------------------
# Per-image orchestration
# ------------------------------------------------------------------

def process_image(image: Path, args) -> Path:
    print(f"\n\n{'#' * 70}")
    print(f"  IMAGE :  {image.name}")
    print(f"{'#' * 70}")

    stem = image.stem

    # -- Step 1: depth -> point cloud -----------------------------
    fast_mesh = MESH_DIR / f"{stem}.ply"
    if args.skip_depth:
        ply = PCD_DIR / f"{stem}.ply"
        if not ply.exists():
            sys.exit(f"[ERR]  --skip_depth set but {ply.relative_to(ROOT)} not found")
        if not fast_mesh.exists():
            fast_mesh = None
        print(f"[SKIP]  Skipping depth  ->  using {ply.relative_to(ROOT)}")
    else:
        ply, fast_mesh = run_depth(image, args)

    # -- Step 3: MVInverse intrinsic decomposition ----------------
    # Run BEFORE meshing so the normal map is available to smooth the mesh.
    if args.skip_mvi:
        maps = collect_ouro_maps_for_stem(stem)
        if not maps:
            stem_mat_dir = MAT_DIR / stem
            sys.exit(
                f"[ERR]  --skip_mvi set but no intrinsic maps found in "
                f"{stem_mat_dir.relative_to(ROOT)}\n"
                f"   Files there: {[p.name for p in stem_mat_dir.rglob('*') if p.is_file()] if stem_mat_dir.exists() else 'directory does not exist'}"
            )
        print(f"[SKIP]  Skipping MVInverse  ->  found: {', '.join(maps)}")
    else:
        maps = run_ouroboros(image)

    normal_map = maps.get("normals")   # may be None if Ouroboros skipped / failed

    # -- Step 2: choose mesh --------------------------------------
    # Default: use the fast image-grid mesh exported by PixelPerfect.
    # Optional: --use_poisson keeps the old slow Poisson reconstruction path.
    if args.skip_mesh:
        mesh_obj = MESH_DIR / f"{stem}.ply"
        if not mesh_obj.exists():
            mesh_obj = MESH_DIR / f"{stem}.obj"
        if not mesh_obj.exists():
            sys.exit(f"[ERR]  --skip_mesh set but {(MESH_DIR / stem).relative_to(ROOT)}.[ply|obj] not found")
        print(f"[SKIP]  Skipping meshing  ->  using {mesh_obj.relative_to(ROOT)}")
    elif args.use_poisson:
        mesh_obj = run_meshing(ply, depth=args.mesh_depth, normal_map=normal_map, args=args)
    else:
        if fast_mesh is None or not fast_mesh.exists():
            sys.exit(
                f"[ERR]  Fast grid mesh not found for {stem}.\n"
                "   Re-run without --skip_depth, or use --use_poisson as a fallback."
            )
        mesh_obj = fast_mesh
        print(f"  Using fast image-grid mesh -> {mesh_obj.relative_to(ROOT)}")

    if getattr(args, "smooth_mesh", True) and not args.use_poisson:
        mesh_obj = postprocess_mesh_file(mesh_obj, args)

    # -- Step 4: combine -> GLB ------------------------------------
    glb_path = run_combine(mesh_obj, maps, reference_image=image, args=args)
    cleanup_intermediates_for_stem(stem, args)
    cleanup_empty_work_dirs(args)
    return glb_path


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Scene reconstruction pipeline: image(s) -> Blender-ready .glb",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--image", type=str, default=None, help="Path to one input image. Omit to process every image in asset/input/.")
    parser.add_argument("--mesh_depth", type=int, default=9, help="Poisson reconstruction depth 8-11. Only used with --use_poisson. Default: 9")
    parser.add_argument("--use_poisson", action="store_true", help="Use the old slow Open3D Poisson reconstruction instead of the fast image-grid mesh.")
    parser.add_argument("--mesh_stride", type=int, default=1, help="Use every Nth depth pixel for the fast grid mesh. 1 = full resolution, 2 = lighter preview mesh.")
    parser.add_argument("--max_depth_jump", type=float, default=0.08, help="Minimum absolute depth-jump threshold for fast-mesh triangle filtering.")
    parser.add_argument("--relative_depth_jump", type=float, default=0.08, help="Also allow depth jumps up to this fraction of local depth; useful for tunnels/far geometry.")
    parser.add_argument("--max_geometry_depth", type=float, default=100.0, help="Hard safety cap: pixels farther than this are treated as background, not mesh geometry. Use <=0 to disable.")
    parser.add_argument("--background_percentile", type=float, default=1.0, help="Adaptive far-background cutoff percentile among valid depths. 0.98 removes farthest ~2% as background.")
    parser.add_argument("--no_batch", action="store_true", help="Disable batch inference and use the older per-image subprocess flow.")
    parser.add_argument("--no_cleanup", dest="cleanup_intermediates", action="store_false",
                        help="Keep asset/*, PixelPerfect, and MVInverse intermediate files for debugging. Default production behavior deletes them after each successful image.")
    parser.add_argument("--keep_intermediates", dest="cleanup_intermediates", action="store_false",
                        help="Alias for --no_cleanup.")
    parser.add_argument("--skip_depth", action="store_true", help="Skip pixel-perfect-depth; reuse existing .ply from asset/pointcloud/ and mesh from asset/mesh/.")
    parser.add_argument("--skip_mesh", action="store_true", help="Skip mesh selection/reconstruction; reuse existing .ply/.obj from asset/mesh/.")
    parser.add_argument("--skip_mvi", "--skip_ouro", dest="skip_mvi", action="store_true",
                        help="Skip MVInverse intrinsic decomposition; reuse existing maps from asset/materials/. "
                             "(--skip_ouro is kept as an alias for backward compatibility.)")
    parser.add_argument("--no_smooth_mesh", dest="smooth_mesh", action="store_false", help="Disable mesh cleanup / smoothing post-process.")
    parser.add_argument("--smooth_method", type=str, default="laplacian", choices=["laplacian", "taubin", "simple", "arap", "none"], help="Mesh post-process method. Default: laplacian")
    parser.add_argument("--smooth_iterations", type=int, default=30, help="Number of smoothing iterations. Default: 30")
    parser.add_argument("--smooth_lambda", type=float, default=0.8, help="Lambda for Laplacian/Taubin smoothing. Default: 0.8")
    parser.add_argument("--taubin_mu", type=float, default=-0.53, help="Mu for Taubin smoothing. Default: -0.53")
    parser.add_argument("--vertex_cluster_voxel", type=float, default=0.0, help="Optional pre-smoothing vertex clustering voxel size. 0 disables it.")
    parser.add_argument("--auto_vertex_cluster", action="store_true", help="Auto-pick a small vertex clustering voxel size from the mesh bounding box.")
    parser.add_argument("--auto_vertex_cluster_fraction", type=float, default=0.003, help="Fraction of bbox diagonal used for --auto_vertex_cluster. Default: 0.003")
    parser.add_argument("--keep_largest_component", action="store_true", help="Drop all but the largest connected mesh component. Disabled by default because valid background/object islands may be disconnected.")
    parser.add_argument("--arap_max_iter", type=int, default=50, help="Max iterations for ARAP smoothing. Default: 50")
    parser.set_defaults(smooth_mesh=True, keep_largest_component=False, cleanup_intermediates=True)
    args = parser.parse_args()

    ensure_dirs()

    if args.image:
        images = [Path(args.image).resolve()]
        if not images[0].exists():
            sys.exit(f"[ERR] Image not found: {images[0]}")
    else:
        images = find_images(INPUT_DIR)
        if not images:
            sys.exit(f"\n[ERR] No images found in {INPUT_DIR.relative_to(ROOT)}\n  Drop a .jpg or .png there, or pass --image path/to/file.jpg\n")

    print(f"\n  {len(images)} image(s) queued:  " f"{', '.join(i.name for i in images)}")

    use_batch = (not args.no_batch) and len(images) > 1

    if not use_batch:
        results = []
        for img in images:
            glb = process_image(img, args)
            results.append(glb)
    else:
        if args.skip_depth:
            depth_outputs = {}
            for image in images:
                ply = PCD_DIR / f"{image.stem}.ply"
                fast_mesh = MESH_DIR / f"{image.stem}.ply"
                if not ply.exists():
                    sys.exit(f"[ERR] --skip_depth set but {ply.relative_to(ROOT)} not found")
                depth_outputs[image.stem] = (ply, fast_mesh if fast_mesh.exists() else None)
            print("[SKIP] Skipping batch PixelPerfect depth")
        else:
            depth_outputs = run_depth_batch(
                images,
                mesh_stride=args.mesh_stride,
                max_depth_jump=args.max_depth_jump,
                relative_depth_jump=args.relative_depth_jump,
                max_geometry_depth=args.max_geometry_depth,
                background_percentile=args.background_percentile,
            )

        if args.skip_mvi:
            maps_outputs = {image.stem: collect_ouro_maps_for_stem(image.stem) for image in images}
            print("[SKIP] Skipping batch MVInverse")
        else:
            maps_outputs = run_ouroboros_batch(images)

        results = []
        for image in images:
            print(f"\n\n{'#' * 70}")
            print(f"  COMBINE : {image.name}")
            print(f"{'#' * 70}")

            ply, fast_mesh = depth_outputs[image.stem]
            maps = maps_outputs.get(image.stem, {})
            normal_map = maps.get("normals")

            if args.skip_mesh:
                mesh_obj = MESH_DIR / f"{image.stem}.ply"
                if not mesh_obj.exists():
                    mesh_obj = MESH_DIR / f"{image.stem}.obj"
                if not mesh_obj.exists():
                    sys.exit(f"[ERR] --skip_mesh set but {(MESH_DIR / image.stem).relative_to(ROOT)}.[ply|obj] not found")
                print(f"[SKIP] Skipping meshing -> using {mesh_obj.relative_to(ROOT)}")
            elif args.use_poisson:
                mesh_obj = run_meshing(ply, depth=args.mesh_depth, normal_map=normal_map, args=args)
            else:
                if fast_mesh is None or not fast_mesh.exists():
                    sys.exit(f"[ERR] Fast grid mesh not found for {image.stem}.\n  Re-run without --skip_depth, or use --use_poisson as a fallback.")
                mesh_obj = fast_mesh
                print(f" Using fast image-grid mesh -> {mesh_obj.relative_to(ROOT)}")

            if getattr(args, "smooth_mesh", True) and not args.use_poisson:
                mesh_obj = postprocess_mesh_file(mesh_obj, args)

            glb_path = run_combine(mesh_obj, maps, reference_image=image, args=args)
            cleanup_intermediates_for_stem(image.stem, args)
            cleanup_empty_work_dirs(args)
            results.append(glb_path)

    header("All done")
    for glb in results:
        print(f"  [OK]  {glb.relative_to(ROOT)}")
    print()
    print("  Open in Blender:  File -> Import -> glTF 2.0  (.glb / .gltf)")
    print()


if __name__ == "__main__":
    main()