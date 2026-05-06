import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
import torch.nn.functional as F
import open3d as o3d
from ppd.utils.set_seed import set_seed
from ppd.utils.align_depth_func import recover_metric_depth_ransac
from ppd.utils.depth2pcd import depth2pcd
from ppd.moge.model.v2 import MoGeModel 
from ppd.models.ppd import PixelPerfectDepth
import json
    

if __name__ == '__main__':
    set_seed(666) # set random seed
    parser = argparse.ArgumentParser(description='Pixel-Perfect Depth')
    parser.add_argument('--img_path', type=str, default='assets/examples/images')
    parser.add_argument('--input_size', type=int, default=[1024, 768])
    parser.add_argument('--outdir', type=str, default='depth_vis')
    parser.add_argument('--no_depth_vis', action='store_true', help='Do not write debug depth visualization PNGs. Recommended for production runs.')
    parser.add_argument('--semantics_model', type=str, default='DA2', choices=['MoGe2', 'DA2'])
    parser.add_argument('--sampling_steps', type=int, default=20)
    parser.add_argument('--apply_filter', action='store_false', default=True)
    parser.add_argument('--pred_only', action='store_true', help='only display/save the predicted depth (no input image)')
    parser.add_argument('--save_npy', action='store_true', help='save raw depth prediction as .npy file (float32, unnormalized)')
    parser.add_argument('--save_pcd', action='store_true', help='save point cloud as .ply file')
    parser.add_argument('--save_mesh', action='store_true', default=True, help='save fast image-grid mesh as .ply')
    parser.add_argument('--mesh_stride', type=int, default=1, help='use every Nth depth pixel for fast mesh; 1 = full resolution')
    parser.add_argument('--max_depth_jump', type=float, default=0.08, help='minimum absolute depth-jump threshold for fast mesh triangle filtering')
    parser.add_argument('--relative_depth_jump', type=float, default=0.08, help='also allow depth jumps up to this fraction of local depth; useful for tunnels/far geometry')
    parser.add_argument("--max_geometry_depth", type=float, default=100.0, help="Hard safety cap: pixels farther than this are treated as background, not mesh geometry. Use 0 or negative to disable hard cap.")
    parser.add_argument("--background_percentile", type=float, default=1.0, help="Adaptive far-background cutoff percentile among valid depths. 0.98 removes the farthest ~2 percent as background.")
    parser.add_argument("--save_background", action="store_true", default=True, help="Save background RGB, geometry mask, and background mask for stage-2 compositing/rendering.")
    args = parser.parse_args()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    if args.semantics_model == 'MoGe2':
        semantics_pth = 'checkpoints/moge2.pt'
        model_pth = 'checkpoints/ppd_moge.pth'
    else:
        semantics_pth = 'checkpoints/depth_anything_v2_vitl.pth'
        model_pth = 'checkpoints/ppd.pth'

    moge = MoGeModel.from_pretrained("checkpoints/moge2.pt").to(DEVICE).eval()

    model = PixelPerfectDepth(semantics_model=args.semantics_model, semantics_pth=semantics_pth, sampling_steps=args.sampling_steps)
    model.load_state_dict(torch.load(model_pth, map_location='cpu'), strict=False)
    model = model.to(DEVICE).eval()

    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)
        filenames = sorted(filenames)

    if not args.no_depth_vis:
        os.makedirs(args.outdir, exist_ok=True)

    cmap = matplotlib.colormaps.get_cmap('Spectral')

    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        
        image = cv2.imread(filename)
        H, W = image.shape[:2]
        depth, resize_image = model.infer_image(image)
        depth = depth.squeeze().cpu().numpy()

        # moge provide metric depth and intrinsic
        resize_H, resize_W = resize_image.shape[:2]
        moge_image = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
        moge_image = torch.tensor(moge_image / 255, dtype=torch.float32, device=DEVICE).permute(2, 0, 1) 
        moge_depth, mask, intrinsic = moge.infer(moge_image)
        moge_depth[~mask] = moge_depth[mask].max()
        
        # relative depth -> metric depth
        metric_depth = recover_metric_depth_ransac(depth, moge_depth, mask)
        intrinsic[0, 0] *= resize_W 
        intrinsic[1, 1] *= resize_H
        intrinsic[0, 2] *= resize_W
        intrinsic[1, 2] *= resize_H

        # metric depth -> point cloud
        pcd = depth2pcd(metric_depth, intrinsic, color=cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB), input_mask=mask, ret_pcd=True)
        if args.save_mesh:
            mesh_dir = "depth_mesh"
            os.makedirs(mesh_dir, exist_ok=True)

            K = intrinsic.detach().cpu().numpy() if torch.is_tensor(intrinsic) else np.asarray(intrinsic)
            depth_np = metric_depth.detach().cpu().numpy() if torch.is_tensor(metric_depth) else np.asarray(metric_depth)
            mask_np = mask.detach().cpu().numpy() if torch.is_tensor(mask) else np.asarray(mask)
            # -------------------------------------------------------
            # Use Ouroboros-style intrinsic albedo for vertex colors
            # (fallback to input RGB if albedo not available)
            # -------------------------------------------------------
            rgb = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)

            # -------------------------------------------------------
            # Adaptive background / sky split
            # -------------------------------------------------------
            # Sky or extremely far regions should not become mesh geometry.
            # Instead, keep them as a background layer for stage-2 compositing.
            base_mask = mask_np.astype(bool) & np.isfinite(depth_np) & (depth_np > 0)
            valid_depths = depth_np[base_mask]

            if valid_depths.size > 0:
                bg_percentile = float(args.background_percentile)
                bg_percentile = min(max(bg_percentile, 0.0), 1.0)
                adaptive_far = float(np.quantile(valid_depths, bg_percentile))

                hard_cap = float(args.max_geometry_depth)
                if hard_cap > 0:
                    far_cut = min(hard_cap, adaptive_far)
                else:
                    far_cut = adaptive_far
            else:
                adaptive_far = float("nan")
                hard_cap = float(args.max_geometry_depth)
                far_cut = hard_cap if hard_cap > 0 else float("inf")

            geometry_mask_full = base_mask & (depth_np < far_cut)
            background_mask_full = ~geometry_mask_full

            print(
                f"Adaptive background split: far_cut={far_cut:.3f} "
                f"(percentile={float(args.background_percentile):.3f}, "
                f"hard_cap={float(args.max_geometry_depth):.3f})"
            )
            print(
                f"Geometry pixels: {int(geometry_mask_full.sum())}/{geometry_mask_full.size} "
                f"({100.0 * geometry_mask_full.mean():.1f}%)"
            )

            if args.save_background:
                bg_dir = "background"
                os.makedirs(bg_dir, exist_ok=True)
                stem = os.path.splitext(os.path.basename(filename))[0]

                # White-out geometry, keep background pixels visible.
                # This is mainly a stage-2 reference layer, not a final panorama.
                background_rgb = rgb.copy()
                background_rgb[geometry_mask_full] = 255

                geometry_mask_u8 = (geometry_mask_full.astype(np.uint8) * 255)
                background_mask_u8 = (background_mask_full.astype(np.uint8) * 255)

                cv2.imwrite(os.path.join(bg_dir, stem + "_background.png"), cv2.cvtColor(background_rgb, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(bg_dir, stem + "_geometry_mask.png"), geometry_mask_u8)
                cv2.imwrite(os.path.join(bg_dir, stem + "_background_mask.png"), background_mask_u8)

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
                with open(os.path.join(bg_dir, stem + "_background_meta.json"), "w") as f:
                    json.dump(bg_meta, f, indent=2)

                print(f"Saved background split to {bg_dir}/{stem}_*.png/json")

            # Optional decimation. This is much faster/lighter for Blender preview.
            stride = max(1, int(args.mesh_stride))
            if stride > 1:
                depth_np = depth_np[::stride, ::stride]
                mask_np = mask_np[::stride, ::stride]
                geometry_mask_full = geometry_mask_full[::stride, ::stride]
                rgb = rgb[::stride, ::stride]
                K = K.copy()
                K[0, 0] /= stride
                K[1, 1] /= stride
                K[0, 2] /= stride
                K[1, 2] /= stride

            Hm, Wm = depth_np.shape
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]

            ys, xs = np.meshgrid(np.arange(Hm), np.arange(Wm), indexing="ij")
            z = depth_np.astype(np.float32)
            x = (xs.astype(np.float32) - cx) * z / fx
            y = (ys.astype(np.float32) - cy) * z / fy

            verts = np.stack([x, y, z], axis=-1).reshape(-1, 3)
            verts *= np.array([1, -1, -1], dtype=np.float32)  # Blender/Open3D convention
            # if you run Ouro after depth, albedo may not exist yet here.
            # so we generate mesh initially with RGB fallback.
            colors = rgb.reshape(-1, 3).astype(np.float64) / 255.0
            valid = geometry_mask_full.astype(bool)

            # Vectorized triangle creation from neighboring valid depth pixels.
            idx = np.arange(Hm * Wm, dtype=np.int32).reshape(Hm, Wm)
            i00 = idx[:-1, :-1]
            i01 = idx[:-1, 1:]
            i10 = idx[1:, :-1]
            i11 = idx[1:, 1:]

            z00 = depth_np[:-1, :-1]
            z01 = depth_np[:-1, 1:]
            z10 = depth_np[1:, :-1]
            z11 = depth_np[1:, 1:]

            v00 = valid[:-1, :-1]
            v01 = valid[:-1, 1:]
            v10 = valid[1:, :-1]
            v11 = valid[1:, 1:]

            # Relative depth-jump threshold.
            # A fixed threshold such as 0.08 m is too strict for tunnel / corridor
            # scenes because neighboring pixels can legitimately differ by a large
            # absolute depth when they are far from the camera. Use the larger of:
            #   1) an absolute minimum threshold, and
            #   2) a fraction of the local depth.
            abs_jump = float(args.max_depth_jump)
            rel_jump = float(args.relative_depth_jump)
            local_z = np.maximum.reduce([z00, z01, z10, z11]).astype(np.float32)
            threshold = np.maximum(abs_jump, rel_jump * local_z)

            tri1_ok = (
                v00 & v01 & v10 &
                (np.abs(z00 - z01) < threshold) &
                (np.abs(z00 - z10) < threshold) &
                (np.abs(z01 - z10) < threshold)
            )
            tri2_ok = (
                v01 & v11 & v10 &
                (np.abs(z01 - z11) < threshold) &
                (np.abs(z10 - z11) < threshold) &
                (np.abs(z01 - z10) < threshold)
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

            mesh_path = os.path.join(mesh_dir, os.path.splitext(os.path.basename(filename))[0] + ".ply")
            o3d.io.write_triangle_mesh(mesh_path, mesh, write_vertex_colors=True)
            print(f"Saved fast grid mesh: {mesh_path} ({len(mesh.vertices)} vertices, {len(mesh.triangles)} faces)")
        if args.apply_filter:
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=3.0)
            pcd = pcd.select_by_index(ind)
        
        if not args.no_depth_vis:
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
            depth_min = float(np.nanmin(depth))
            depth_max = float(np.nanmax(depth))
            denom = max(1e-8, depth_max - depth_min)
            vis_depth = (depth - depth_min) / denom * 255.0
            vis_depth = vis_depth.astype(np.uint8)
            vis_depth = (cmap(vis_depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            
            if args.pred_only:
                cv2.imwrite(os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'), vis_depth)
            else:
                split_region = np.ones((image.shape[0], 50, 3), dtype=np.uint8) * 255
                combined_result = cv2.hconcat([image, split_region, vis_depth])
                cv2.imwrite(os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'), combined_result)

        if args.save_npy:
            depth_npy_dir = 'depth_npy'
            os.makedirs(depth_npy_dir, exist_ok=True)
            npy_path = os.path.join(depth_npy_dir, os.path.splitext(os.path.basename(filename))[0] + '.npy')
            np.save(npy_path, depth)
        

        if args.save_pcd:
            depth_pcd_dir = 'depth_pcd'
            os.makedirs(depth_pcd_dir, exist_ok=True)
            pcd_path = os.path.join(depth_pcd_dir, os.path.splitext(os.path.basename(filename))[0] + '.ply')
            pcd.points = o3d.utility.Vector3dVector(
                np.asarray(pcd.points) * np.array([1, -1, -1], dtype=np.float32))
            o3d.io.write_point_cloud(pcd_path, pcd)
            meta_path = os.path.join(depth_pcd_dir, os.path.splitext(os.path.basename(filename))[0] + "_camera.json")

            meta = {
                "resize_W": int(resize_W),
                "resize_H": int(resize_H),
                "intrinsic": intrinsic.detach().cpu().numpy().tolist()
                if torch.is_tensor(intrinsic) else np.asarray(intrinsic).tolist(),
                "saved_ply_flip": [1.0, -1.0, -1.0],
                "max_geometry_depth": float(args.max_geometry_depth),
                "background_percentile": float(args.background_percentile),
                "background_files": {
                    "background_rgb": f"background/{os.path.splitext(os.path.basename(filename))[0]}_background.png",
                    "geometry_mask": f"background/{os.path.splitext(os.path.basename(filename))[0]}_geometry_mask.png",
                    "background_mask": f"background/{os.path.splitext(os.path.basename(filename))[0]}_background_mask.png",
                    "meta": f"background/{os.path.splitext(os.path.basename(filename))[0]}_background_meta.json"
                }
            }

            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            print(f"Saved camera metadata: {meta_path}")