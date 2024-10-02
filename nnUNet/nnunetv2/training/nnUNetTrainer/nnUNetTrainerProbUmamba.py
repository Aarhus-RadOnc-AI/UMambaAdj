
from typing import Tuple, Union, List
import numpy as np
import torch
from torch import autocast, nn
import pydoc

from time import time
from os.path import join


from torch import distributed as dist

from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.logging.nnunet_logger import nnUNetLogger, nnUNetLogger_hnts2024

from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler

#from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.nets.prob_resnet import ResidualEncoderProbUMamba

class nnUNetTrainerProbUmamba(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        # Gradient accumulation setup
        self.accumulation_steps = 8  # Number of batches to accumulate gradients over
        self.current_step = 0  #init current step for each epoch
        #logger for hnts 2024
        self.logger = nnUNetLogger_hnts2024()
        self.initial_lr = 1e-3
        self.num_epochs=1200
        
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        return get_network_from_plans(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision)
        
    # def configure_optimizers(self):
    #     optimizer = torch.optim.AdamW(self.network.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay)
    #     lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
    #     return optimizer, lr_scheduler
    
    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)


        self.optimizer.zero_grad(set_to_none=True)
        
        # Initialize grad_norm to None
        grad_norm = None
        
        # Use autocast only if the device is 'cuda'
        context = autocast if self.device.type == 'cuda' else dummy_context
        with context(self.device.type, enabled=True):
            output, mu, logvar = self.network(data)
            
            # Compute the segmentation loss
            segmentation_loss = self.loss(output, target)
            
            # Compute the KL divergence loss
            kl_loss = 0
            epsilon = 1e-6  # Small value to prevent numerical instability
            # The term  logvar can become very negative. 
            # When you apply the exponential function exp(logvar), 
            # a very negative logvar will result in a very small variance, 
            # which could lead to extremely high values when subtracted or divided. 
            # Adding a small epsilon ensures that the variance is never too close to zero.
            
            kl_loss += -0.5 * torch.sum(1 + logvar - mu.pow(2)  - (logvar + epsilon).exp())
            # KL Annealing        
            beta = min(0.2, 1 - torch.exp(torch.tensor(-((self.current_epoch) / self.num_epochs))).item())
        
            # Combine segmentation loss and KL divergence loss
            total_loss = (segmentation_loss + beta * kl_loss) /  self.accumulation_steps

        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()
            # Gradients are accumulated across multiple steps
            if (self.current_step + 1) % self.accumulation_steps == 0:
                self.grad_scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)  
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                self.optimizer.zero_grad(set_to_none=True)  # Reset gradients after the accumulation step
        else:
            total_loss.backward()
            # Gradients are accumulated across multiple steps
            if (self.current_epoch+ 1) % self.accumulation_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)   
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)  # Reset gradients after the accumulation step

        # Check grad_norm only if it was assigned (after an optimizer step)
        if grad_norm is not None and grad_norm > 100:
            self.print_to_log_file(f"->>> extreme Large gradient norm: {grad_norm}")

        # Increment current step
        self.current_step += 1

        return {
            'seg_loss': segmentation_loss.detach().cpu().numpy(),
            'kl_loss': kl_loss.detach().cpu().numpy(),
            'total_loss': total_loss.detach().cpu().numpy(),
            'beta': beta
    }
   
    def on_train_epoch_end(self, train_outputs: List[dict]):
        outputs = collate_outputs(train_outputs)

        if self.is_ddp:
            # For distributed data parallel (DDP) mode, gather losses and beta from all processes
            seg_losses_tr = [None for _ in range(dist.get_world_size())]
            kl_losses_tr = [None for _ in range(dist.get_world_size())]
            total_losses_tr = [None for _ in range(dist.get_world_size())]
            betas_tr = [None for _ in range(dist.get_world_size())]

            dist.all_gather_object(seg_losses_tr, outputs['seg_loss'])
            dist.all_gather_object(kl_losses_tr, outputs['kl_loss'])
            dist.all_gather_object(total_losses_tr, outputs['total_loss'])
            dist.all_gather_object(betas_tr, outputs['beta'])

            # Calculate the mean for each metric
            seg_loss_here = np.vstack(seg_losses_tr).mean()
            kl_loss_here = np.vstack(kl_losses_tr).mean()
            total_loss_here = np.vstack(total_losses_tr).mean()
            beta_here = np.vstack(betas_tr).mean()
        else:
            # For non-DDP mode, simply calculate the mean of losses and beta
            seg_loss_here = np.mean(outputs['seg_loss'])
            kl_loss_here = np.mean(outputs['kl_loss'])
            total_loss_here = np.mean(outputs['total_loss'])
            beta_here = np.mean(outputs['beta'])

        # Log the mean losses and beta for the epoch
        self.logger.log('train_seg_losses', seg_loss_here, self.current_epoch)
        self.logger.log('kl_losses', kl_loss_here, self.current_epoch)
        self.logger.log('total_losses', total_loss_here, self.current_epoch)
        self.logger.log('beta', beta_here, self.current_epoch)


    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Use autocast only if the device is 'cuda'
        context = autocast if self.device.type == 'cuda' else dummy_context
        with context(self.device.type, enabled=True):
            output, _, _ = self.network(data)  # Ignoring latent_vars in validation
            del data
            l = self.loss(output, target)

        # If deep supervision is enabled, select the highest output resolution
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        # Compute predicted segmentation for evaluation
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # No need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        # Handling ignored labels during evaluation
        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                target[target == self.label_manager.ignore_label] = 0  # Adjust target
            else:
                mask = 1 - target[:, -1:]
                target = target[:, :-1]  # Adjust target for regions
        else:
            mask = None

        # Calculate true positives, false positives, and false negatives
        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # Ignore background dice in conventional softmax training
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}


    def on_epoch_end(self):
        self.current_step = 0
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        # Log the KL loss, train loss, val loss, pseudo dice, total loss, and beta
        self.print_to_log_file('KL loss', np.round(self.logger.my_fantastic_logging['kl_losses'][-1], decimals=4))
        #self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        self.print_to_log_file('Segmentation loss', np.round(self.logger.my_fantastic_logging['train_seg_losses'][-1], decimals=4))
        self.print_to_log_file('Total loss', np.round(self.logger.my_fantastic_logging['total_losses'][-1], decimals=4))
        self.print_to_log_file('Beta', np.round(self.logger.my_fantastic_logging['beta'][-1], decimals=4))
        self.print_to_log_file('Pseudo dice', [np.round(i, decimals=4) for i in self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]])

        epoch_time = self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1]
        self.print_to_log_file(f"Epoch time: {np.round(epoch_time, decimals=2)} s")

        # Handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.new_num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        # Handle 'best' checkpointing based on EMA pseudo dice
        if self._best_ema is None or self.logger.my_fantastic_logging['ema_fg_dice'][-1] > self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
            self.print_to_log_file(f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        # Generate progress plot
        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        # Increment the epoch counter
        self.current_epoch += 1
        

def get_network_from_plans(arch_class_name, arch_kwargs, arch_kwargs_req_import, input_channels, output_channels,
                        allow_init=True, deep_supervision: Union[bool, None] = None):
    network_class = arch_class_name
    architecture_kwargs = dict(**arch_kwargs)
    for ri in arch_kwargs_req_import:
        if architecture_kwargs[ri] is not None:
            architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])
    #chose your network here: ResidualEncoderUNet, ResidualUNet, ResidualEncoderUMamba
    nw_class = ResidualEncoderProbUMamba

    if deep_supervision is not None and 'deep_supervision' not in arch_kwargs.keys():
        arch_kwargs['deep_supervision'] = deep_supervision
    print(architecture_kwargs)
    network = nw_class(
        input_channels=input_channels,
        num_classes=output_channels,
        **architecture_kwargs
    )

    if hasattr(network, 'initialize') and allow_init:
        network.apply(network.initialize)

    return network