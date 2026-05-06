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
from ppd.utils.transform import image2tensor, resize_1024, resize_1024_crop, resize_keep_aspect

from ppd.models.depth_anything_v2.dpt import DepthAnythingV2
from ppd.models.dit import DiT

class PixelPerfectDepth(nn.Module):
    def __init__(
        self,
        semantics_model='MoGe2',
        semantics_pth='checkpoints/moge2.pt',
        sampling_steps=10,
        ):
        super().__init__()
        self.sampling_steps = sampling_steps
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.device = DEVICE

        if semantics_model == 'MoGe2':
            from ppd.moge.model.v2 import MoGeModel
            self.sem_encoder = MoGeModel.from_pretrained(semantics_pth)
        else:
            self.sem_encoder = DepthAnythingV2(
                encoder='vitl',
                features=256,
                out_channels=[256, 512, 1024, 1024]
            )
            self.sem_encoder.load_state_dict(torch.load(semantics_pth, map_location='cpu'), strict=False)
        self.sem_encoder = self.sem_encoder.to(self.device).eval()
        self.sem_encoder.requires_grad_(False)

        self.configure_diffusion()
        self.dit = DiT()

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
    def infer_image(self, image, use_fp16: bool = True):
        # Resize the image to match the training resolution area while keeping the original aspect ratio.
        resize_image = resize_keep_aspect(image)
        image = image2tensor(resize_image)
        image = image.to(self.device)
        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type=self.device.type, dtype=autocast_dtype):
            depth = self.forward_test(image)
        return depth, resize_image
    
    @torch.no_grad()
    def forward_test(self, image):

        semantics = self.semantics_prompt(image)
        cond = image - 0.5
        latent = torch.randn(size=[cond.shape[0], 1, cond.shape[2], cond.shape[3]]).to(self.device)
        
        for timestep in self.sampling_timesteps:
            input = torch.cat([latent, cond], dim=1)
            pred = self.dit(x=input, semantics=semantics, timestep=timestep)
            latent = self.sampler.step(pred=pred, x_t=latent, t=timestep)
        return latent + 0.5


    @torch.no_grad()
    def semantics_prompt(self, image):
        with torch.no_grad():
            semantics = self.sem_encoder.forward_semantics(image)
        return semantics
