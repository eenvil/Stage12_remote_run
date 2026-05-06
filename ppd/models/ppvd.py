from PIL import Image
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import random
from omegaconf import DictConfig
from ppd.utils.diffusion.timesteps import Timesteps
from ppd.utils.diffusion.schedule import LinearSchedule
from ppd.utils.diffusion.sampler import EulerSampler
from ppd.utils.transform import video2tensor
from ppd.utils.align_vda import align_video_depth

from ppd.models.dit_video import DiT_Video
from safetensors.torch import load_file

# infer settings, do not change
INFER_LEN = 16
KEYFRAMES = [0, 8, 15]
OVERLAP = 3
STRIDE = 13

class PixelPerfectVideoDepth(nn.Module):
    def __init__(
        self,
        semantics_model='Pi3',
        semantics_pth='checkpoints/pi3.safetensors',
        sampling_steps=4,
        ):
        super().__init__()
        self.sampling_steps = sampling_steps
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.device = DEVICE

        if semantics_model == 'Pi3':
            from ppd.models.pi3.models.pi3 import Pi3
            self.sem_encoder = Pi3()
            self.sem_encoder.load_state_dict(load_file(semantics_pth))

        self.sem_encoder = self.sem_encoder.to(self.device).eval()
        self.sem_encoder.requires_grad_(False)

        self.configure_diffusion()
        self.dit_video = DiT_Video()

    def configure_diffusion(self):
        self.schedule = LinearSchedule(T=1000)
        self.sampling_timesteps = Timesteps(
            T=self.schedule.T,
            steps=self.sampling_steps,
            device=self.device,
            )
        self.sampler = EulerSampler(
            schedule=self.schedule,
            timesteps=self.sampling_timesteps,
            prediction_type='velocity'
            )
    
    @torch.no_grad()
    def infer_video(self, images, use_fp16: bool = True):
        images = video2tensor(images)
        images = [img.to(self.device) for img in images]
        p_imgs = [F.interpolate(img, size=(512, 512), mode='bilinear', align_corners=False) for img in images]
        LEN = len(p_imgs)
        R = (LEN - INFER_LEN) % STRIDE
        if R != 0:
            pad_len = STRIDE - R
            last_img = p_imgs[-1]
            p_imgs.extend([last_img.clone() for _ in range(pad_len)])
        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type=self.device.type, dtype=autocast_dtype):
            preds = self.forward_test(p_imgs)

        preds = [F.interpolate(pred, size=images[0].shape[-2:], mode='bilinear', align_corners=False) for pred in preds]
        preds = align_video_depth(preds, INFER_LEN, KEYFRAMES, OVERLAP)
        return preds[:LEN]
    
    @torch.no_grad()
    def forward_test(self, imgs):
        preds = []
        pre_img = None
        init_latent = torch.randn(size=[INFER_LEN, 1, imgs[0].shape[2], imgs[0].shape[3]]).to(self.device)
        for i in range(0, len(imgs)-INFER_LEN+1, STRIDE):
            cur_img = imgs[i:i+INFER_LEN]
            if pre_img is not None:
                cur_img[:OVERLAP] = [pre_img[k] for k in KEYFRAMES]
            pre_img = cur_img
            concat_img = torch.cat(cur_img, dim=0)
            semantics = self.semantics_prompt(concat_img)
            cond = concat_img - 0.5
            latent = init_latent
        
            for timestep in self.sampling_timesteps:
                input = torch.cat([latent, cond], dim=1)
                pred = self.dit_video(x=input, semantics=semantics, timestep=timestep)
                latent = self.sampler.step(pred=pred, x_t=latent, t=timestep)
            cur_pred = latent + 0.5
            preds.append(cur_pred)
        return preds

    @torch.no_grad()
    def semantics_prompt(self, images):
        with torch.no_grad():
            semantics = self.sem_encoder.forward_semantics(images)
        return semantics
