import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from mvinverse.models.mvinverse import MVInverse


def load_model(ckpt_path_or_repo_id, device):
    if os.path.exists(ckpt_path_or_repo_id):
        print(f"Loading model from local checkpoint: {ckpt_path_or_repo_id}")
        model = MVInverse().to(device).eval()
        weight = torch.load(ckpt_path_or_repo_id, map_location=device, weights_only=False)
        weight = weight["model"] if "model" in weight else weight
        missing, unused = model.load_state_dict(weight, strict=False)
        if len(missing) != 0:
            print(f"Warning: Missing keys found: {missing}")
        if len(unused) != 0:
            print(f"Got unexpected keys: {unused}")
    else:
        print(f"Checkpoint not found locally. Attempting to download from Hugging Face Hub: {ckpt_path_or_repo_id}")
        model = MVInverse.from_pretrained(ckpt_path_or_repo_id)
        model = model.to(device).eval()
    return model


def load_single_image(image_path: Path) -> torch.Tensor:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        width, height = img.size

        # Match upstream MVInverse inference.py behavior.
        if max(width, height) > 1024:
            scaling_factor = 1024 / max(width, height)
            new_width, new_height = int(width * scaling_factor), int(height * scaling_factor)
        else:
            new_width, new_height = width, height

        # MVInverse requires dimensions divisible by 14.
        new_w = max(14, new_width // 14 * 14)
        new_h = max(14, new_height // 14 * 14)
        img = img.resize((new_w, new_h))

    return transforms.ToTensor()(img)


def save_one_result(res, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)

    albedo = res["albedo"][0, 0]
    metallic = res["metallic"][0, 0]
    roughness = res["roughness"][0, 0]
    normal = res["normal"][0, 0]
    shading = res["shading"][0, 0]

    Image.fromarray((albedo.cpu().numpy() * 255).astype(np.uint8)).save(save_dir / "000_albedo.png")
    Image.fromarray((metallic.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)).save(save_dir / "000_metallic.png")
    Image.fromarray((roughness.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)).save(save_dir / "000_roughness.png")
    Image.fromarray(((normal * 0.5 + 0.5).cpu().numpy() * 255).astype(np.uint8)).save(save_dir / "000_normal.png")
    Image.fromarray((shading.cpu().numpy() * 255).astype(np.uint8)).save(save_dir / "000_shading.png")


def main():
    parser = argparse.ArgumentParser(
        description="MVInverse single-image batch runner: load model once, process unrelated images independently."
    )
    parser.add_argument("--image_list", required=True, help="Text file containing one absolute image path per line.")
    parser.add_argument("--ckpt", default="maddog241/mvinverse")
    parser.add_argument("--save_path", default="outputs")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    with open(args.image_list, "r", encoding="utf-8") as f:
        image_paths = [Path(line.strip()) for line in f if line.strip()]

    if not image_paths:
        raise RuntimeError(f"No images listed in {args.image_list}")

    print("Loading model once...")
    device = torch.device(args.device)
    model = load_model(args.ckpt, device)

    save_root = Path(args.save_path)
    use_cuda_amp = device.type == "cuda"

    for idx, image_path in enumerate(image_paths, start=1):
        print("-" * 70)
        print(f"[{idx}/{len(image_paths)}] Single-image MVInverse: {image_path.name}")
        img = load_single_image(image_path).unsqueeze(0).to(device)   # [1, 3, H, W]
        print(f"Loaded 1 image, size is {tuple(img.shape[-2:])}")

        with torch.no_grad():
            if use_cuda_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    res = model(img[None])  # [B=1, N=1, C, H, W]
            else:
                res = model(img[None])

        out_dir = save_root / image_path.stem
        save_one_result(res, out_dir)
        print(f"Results saved to {out_dir}")

    print("Done. MVInverse model was loaded once for all unrelated images.")


if __name__ == "__main__":
    main()
