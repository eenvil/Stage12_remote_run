import os
from os.path import join
import numpy as np
import imageio
import torch
import cv2
import pytorch_lightning as pl
from hydra.utils import instantiate
from typing import Any, Dict, List

from ppd.utils.align_depth_func import recover_metric_depth_ransac
from ppd.utils.parallel_utils import async_call
from ppd.utils.logger import Log
from ppd.utils.vis_utils import visualize_depth


class DepthEstimationModel(pl.LightningModule):
    def __init__(
        self,
        pipeline,  # The pipeline is the model itself
        optimizer,  # The optimizer is the optimizer used to train the model
        lr_table,  # The lr_table is the learning rate table
        output_dir: str,
        ignored_weights_prefix=["pipeline.sem_encoder"],
        save_vis_depth=False,  # Whether to save the visualized depth
        save_vis_depth_and_concat_img=False,
        save_vis_depth_and_concat_gt=True,
        **kwargs,
    ):
        super().__init__()

        self.pipeline = instantiate(pipeline, _recursive_=False)
        self.optimizer = instantiate(optimizer)
        self.lr_table = instantiate(lr_table)
        self.ignored_weights_prefix = ignored_weights_prefix

        self._save_vis_depth = save_vis_depth
        self._save_vis_depth_and_concat_img = save_vis_depth_and_concat_img
        self._save_vis_depth_and_concat_gt = save_vis_depth_and_concat_gt
        self.align_depth_func = recover_metric_depth_ransac
        self.output_dir = output_dir

        Log.info('Results will be saved to: {}'.format(self.output_dir))

    def training_step(self, batch, batch_idx):
        output = self.pipeline.forward_train(batch)
        if not isinstance(self.trainer.train_dataloader, List):
            B = self.trainer.train_dataloader.batch_size
        else:
            B = np.sum(
                [dataloader.batch_size for dataloader in self.trainer.train_dataloader])
        loss = output['loss']
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            raise ValueError(f"Loss is NaN or Inf: {loss}")
        self.log('train/loss', loss, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, batch_size=B, sync_dist=True)

        lr = self.optimizers().param_groups[0]['lr']
        self.log('train/lr', lr, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Save visualization every 100 steps
        if self.global_step % 100 == 0:
            if 'depth' in output and 'image' in output:
                depth_np = output['depth'][0][0].float().detach().cpu().numpy()
                rgb_np = output['image'][0].detach().cpu().numpy().transpose((1, 2, 0))
                depth_vis = visualize_depth(depth_np)
                depth_vis = (depth_vis * 255.).astype(np.uint8)
                rgb_vis = (rgb_np * 255.).astype(np.uint8)
                vis_img = np.concatenate([rgb_vis, depth_vis], axis=1)
                self.logger.experiment.add_image('train/depth_vis', 
                                              vis_img.transpose((2,0,1)),
                                              self.global_step)
        if 'depth' in output: del output['depth']
        return output

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        output = self.pipeline.forward_test(batch)
        if self._save_vis_depth:
            self.save_vis_depth(output['depth'], output['image'], batch['image_name'], 'vis_depth',
                                gt_depth=batch['depth'] if 'depth' in batch else None)
        return output

    def validation_step(self, batch, batch_idx, dataloader_idx=None) -> None:
        output = self.predict_step(batch, batch_idx, dataloader_idx)
        batch_size = batch['image'].shape[0]
        metrics_dict = self.compute_metrics(output, batch)
        for k, v in metrics_dict.items():
            self.log(f'val/{k}', np.mean(v), 
                     on_step=False, 
                     on_epoch=True,
                     prog_bar=True if 'l1' in k else False, 
                     logger=True, 
                     batch_size=batch_size, 
                     sync_dist=True)

    def compute_metrics(self, output, batch):
        B = batch['image'].shape[0]
        metrics_dict = {}
        for b in range(B):
            pred_depth = output['depth'][b][0].float().detach().cpu().numpy()
            gt_depth = batch['depth'][b][0].float().detach().cpu().numpy()
            msk = self.create_depth_mask(batch['dataset_name'], gt_depth)
            msk = msk & batch['mask'][b, 0].detach().cpu().numpy().astype(np.bool_)
            gt_depth[~msk] = 0.
            pred_depth = self.align_depth_func(
                pred_depth, gt_depth, msk, log=True)
                
            metrics_dict_item = self.compute_depth_metric(
                pred_depth, gt_depth, msk)
            metrics_dict = self.update_metrics_dict(
                metrics_dict, metrics_dict_item, 'relative')
        return metrics_dict

    def update_metrics_dict(self, metrics_dict, metrics_dict_item, prefix):
        for k, v in metrics_dict_item.items():
            if f'{prefix}_{k}' not in metrics_dict:
                metrics_dict[f'{prefix}_{k}'] = []
            metrics_dict[f'{prefix}_{k}'].append(v)
        return metrics_dict

    def create_depth_mask(self, dataset_name, gt_depth):
        return gt_depth > 1e-3

    def compute_depth_metric(self, pred_depth, gt_depth, msk):
        gt = gt_depth[msk]
        pred = pred_depth[msk]
        thresh = np.maximum((gt / (pred + 1e-5)), (pred / (gt + 1e-5)))
        d05 = (thresh < 1.25 ** 0.5).mean()
        d1 = (thresh < 1.25).mean()
        d2 = (thresh < 1.25 ** 2).mean()
        d3 = (thresh < 1.25 ** 3).mean()
        abs_rel = np.mean(np.abs(gt - pred) / (gt + 1e-5))

        return {
            'd0.5': d05,
            'd1': d1,
            'd2': d2,
            'd3': d3,
            'abs_rel': abs_rel,
        }

    @async_call
    def save_depth(self, depth, name, tag) -> None:
        if not isinstance(depth, torch.Tensor):
            depth = torch.tensor(depth).unsqueeze(0).unsqueeze(0)
        for b in range(len(depth)):
            depth_np = depth[b][0].float().detach().cpu().numpy()
            last_split_len = len(name[b].split('.')[-1])
            save_name = name[b][:-(last_split_len + 1)] + '.npz'
            img_path = join(self.output_dir, f'{tag}/{save_name}')
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            np.savez_compressed(img_path, data=np.round(depth_np, 3))

    @async_call
    def save_vis_depth(self, depth, rgb, name, tag, gt_depth=None) -> None:
        for b in range(len(depth)):
            depth_np = depth[b][0].float().detach().cpu().numpy()
            save_name = name[b]

            save_imgs = []
            save_img = visualize_depth(depth_np,
                            depth_np.min(),
                            depth_np.max()
                            )
            save_imgs.append(save_img)
            if self._save_vis_depth_and_concat_img:
                rgb_np = rgb[b].float().detach().cpu().numpy().transpose((1, 2, 0))


                rgb_np = cv2.resize(
                    rgb_np, (save_img.shape[1], save_img.shape[0]), interpolation=cv2.INTER_AREA)

                save_img = np.concatenate(
                    [rgb_np, save_img], axis=1)
                save_imgs.append(rgb_np)
            if gt_depth is not None and self._save_vis_depth_and_concat_gt:
                gt_depth_np = gt_depth[b][0].float().detach().cpu().numpy()
                gt_depth_vis = visualize_depth(gt_depth_np, 
                                    gt_depth_np.min(),
                                    gt_depth_np.max()
                                    )
                save_img = np.concatenate(
                    [save_img, gt_depth_vis], axis=1)
                save_imgs.append(gt_depth_vis)
            img_path = join(self.output_dir, f'{tag}/{save_name}')
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            imageio.imwrite(img_path.replace('.jpg', '.png'),
                            (save_img * 255.).astype(np.uint8))

    def configure_optimizers(self):
        group_table = {}
        params = []
        for k, v in self.pipeline.named_parameters():
            if v.requires_grad:
                group, lr = self.lr_table.get_lr(k)
                if lr == 0:
                    v.requires_grad = False
                if group not in group_table:
                    group_table[group] = len(group_table)
                    params.append({'params': [v], 'lr': lr, 'name': group})
                else:
                    params[group_table[group]]['params'].append(v)
        optimizer = self.optimizer(params=params)
        return optimizer
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        for ig_keys in self.ignored_weights_prefix:
            Log.debug(f"Remove key `{ig_keys}' from checkpoint.")
            for k in list(checkpoint["state_dict"].keys()):
                if k.startswith(ig_keys):
                    checkpoint["state_dict"].pop(k)
        super().on_save_checkpoint(checkpoint)

    def load_pretrained_model(self, ckpt_path):
        """Load pretrained checkpoint, and assign each weight to the corresponding part."""
        Log.info(f"Loading ckpt: {ckpt_path}")
        state_dict = torch.load(ckpt_path, "cpu")["state_dict"]
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        real_missing = []
        for k in missing:
            miss = True
            for ig_keys in self.ignored_weights_prefix:
                if k.startswith(ig_keys):
                    miss = False
            if miss:
                real_missing.append(k)
        if len(real_missing) > 0:
            Log.warn(f"Missing keys: {real_missing}")
        if len(unexpected) > 0:
            Log.error(f"Unexpected keys: {unexpected}")

    def load_pretrained_model_eval(self, ckpt_path):
        """Load pretrained checkpoint, and assign each weight to the corresponding part."""
        Log.info(f"Loading ckpt: {ckpt_path}")
        state_dict = torch.load(ckpt_path, "cpu")
        fixed_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("dit."):
                fixed_state_dict[f"pipeline.{k}"] = v
            else:
                fixed_state_dict[k] = v
        missing, unexpected = self.load_state_dict(fixed_state_dict, strict=False)
        real_missing = []
        for k in missing:
            miss = True
            for ig_keys in self.ignored_weights_prefix:
                if k.startswith(ig_keys):
                    miss = False
            if miss:
                real_missing.append(k)
        if len(real_missing) > 0:
            Log.warn(f"Missing keys: {real_missing}")
        if len(unexpected) > 0:
            Log.error(f"Unexpected keys: {unexpected}")

