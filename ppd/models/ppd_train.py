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
from ppd.utils.diffusion.logitnormal import LogitNormalTrainingTimesteps
from ppd.utils.transform import image2tensor, resize_1024, resize_1024_crop, resize_keep_aspect

from ppd.models.depth_anything_v2.dpt import DepthAnythingV2
from ppd.models.dit import DiT
from ppd.models.loss import multi_scale_grad_loss

def get_device() -> torch.device:
    """
    Get current rank device.
    """
    return torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))

class PixelPerfectDepth(nn.Module):
    def __init__(
        self, config: DictConfig):
        super().__init__()
        self.config = config
        self.configure_diffusion()

        if self.config.semantics_model == 'MoGe2':
            from ppd.moge.model.v2 import MoGeModel
            self.sem_encoder = MoGeModel.from_pretrained(self.config.semantics_pth)
        else:
            self.sem_encoder = DepthAnythingV2(
                encoder='vitl',
                features=256,
                out_channels=[256, 512, 1024, 1024]
            )
            self.sem_encoder.load_state_dict(torch.load(self.config.semantics_pth, map_location='cpu'), strict=False)
        self.sem_encoder = self.sem_encoder.to(get_device()).eval()
        self.sem_encoder.requires_grad_(False)

        self.dit = DiT()

    def configure_diffusion(self):
        self.schedule = LinearSchedule(T=1000)
        self.sampling_timesteps = Timesteps(
            T=self.schedule.T,
            steps=self.config.diffusion.timesteps.sampling.steps,
            device=get_device(),
            )
        self.sampler = EulerSampler(
            schedule=self.schedule,
            timesteps=self.sampling_timesteps,
            prediction_type='velocity'
            )
        self.training_timesteps = LogitNormalTrainingTimesteps(
            T=self.schedule.T,
            loc=self.config.diffusion.timesteps.training.loc,
            scale=self.config.diffusion.timesteps.training.scale,
            )
    
    @torch.no_grad()
    def forward_test(self, batch: dict):
        ori_h, ori_w = batch['image'].shape[-2:]
        current_area = ori_w * ori_h
        target_area = 512 * 512
        if not self.config.pretrain:
            target_area = 1024 * 768
        scale = scale = (target_area / current_area) ** 0.5
        new_h = max(16, int(round(ori_h * scale / 16)) * 16)
        new_w = max(16, int(round(ori_w * scale / 16)) * 16)
        image = F.interpolate(batch['image'], size=(new_h, new_w), mode='bilinear', align_corners=False)

        cond = self.get_cond(image)
        semantics = self.semantics_prompt(image)
        latent = torch.randn(size=[cond.shape[0], 1, cond.shape[2], cond.shape[3]]).to(get_device())
        
        for timestep in self.sampling_timesteps:
            x = torch.cat([latent, cond], dim=1)
            pred = self.dit(x=x, semantics=semantics, timestep=timestep)
            latent = self.sampler.step(pred=pred, x_t=latent, t=timestep)
        depth = latent + 0.5
        depth = F.interpolate(depth, size=batch['image'].shape[-2:], mode='nearest')

        return {'depth': depth, 'image': batch['image']}

    @torch.no_grad()
    def semantics_prompt(self, image):
        with torch.no_grad():
            semantics = self.sem_encoder.forward_semantics(image)
        return semantics

    @torch.no_grad()
    def get_cond(self, img):
        return img-0.5

    @torch.no_grad()
    def get_gt(self, batch: dict):
        depth = batch['depth']
        mask = batch['mask'].bool()
        B = depth.shape[0]
        min_val = []
        max_val = []
        clip_mask = mask & (depth<80.)
        depth = torch.log(depth+1.)
        for i in range(B):
            i_depth = depth[i]
            i_mask = clip_mask[i]
            i_min_val, i_max_val = torch.quantile(i_depth[i_mask], 0.02, dim=-1), torch.quantile(i_depth[i_mask], 0.98, dim=-1)
            min_val.append(i_min_val)
            max_val.append(i_max_val)
        min_val = torch.stack(min_val)
        max_val = torch.stack(max_val)
        invalid_mask = (max_val - min_val) < 1e-6
        if invalid_mask.any(): max_val[invalid_mask] = min_val[invalid_mask] + 1e-6
        min_val, max_val = min_val[:, None, None, None], max_val[:, None, None, None]
        depth = (depth - min_val) / (max_val - min_val)
        depth = torch.clamp(depth, -0.5, 1.0)
        return depth-0.5, mask

    def forward_train(self, batch: dict):
        batch_size = batch['image'].shape[0]
        cond = self.get_cond(batch['image'])
        latent, mask = self.get_gt(batch)
        semantics = self.semantics_prompt(batch['image'])
        noises = torch.randn_like(latent)
        timesteps = self.training_timesteps.sample([batch_size], device=get_device())
        latent_noised = self.schedule.forward(latent, noises, timesteps)
        x = torch.cat([latent_noised, cond], dim=1)
        pred = self.dit(x=x, semantics=semantics, timestep=timesteps)

        assert pred.shape == latent.shape == noises.shape
        latent_pred, noises_pred = self.schedule.convert_from_pred(
            pred=pred,
            pred_type='velocity',
            x_t=latent_noised,
            t=timesteps,
            )
        loss_input = self.schedule.convert_to_pred(
            x_0=latent_pred,
            x_T=noises_pred,
            t=timesteps,
            pred_type='velocity',
            )
        loss_target = self.schedule.convert_to_pred(
            x_0=latent,
            x_T=noises,
            t=timesteps,
            pred_type='velocity',
            )
        loss = F.mse_loss(
            input=loss_input,
            target=loss_target,
            reduction='none',
            )
        loss = loss * mask.float()
        loss = loss.sum() / (mask.float().sum() + 1e-6)

        ####### finetune stage
        if not self.config.pretrain:
            grad_loss = multi_scale_grad_loss(
                latent_pred.squeeze(1), latent.squeeze(1), mask.float().squeeze(1)
                )
            loss = loss + 0.2 * grad_loss
        ####### finetune stage

        return {'loss': loss, 'depth': latent_pred+0.5, 'image': batch['image']}

        




