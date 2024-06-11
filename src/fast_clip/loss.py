import math
import logging
from typing import Tuple, Optional

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss


class CoCaLoss(ClipLoss):
    def __init__(
            self,
            caption_loss_weight,
            clip_loss_weight,
            pad_id=0,  # pad_token for open_clip custom tokenizer
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )

        self.clip_loss_weight = clip_loss_weight
        self.caption_loss_weight = caption_loss_weight
        self.caption_loss = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, image_features, text_features, logits, labels, logit_scale, output_dict=False):
        
        clip_loss = torch.tensor(0)
        
        if self.clip_loss_weight:
            clip_loss = super().forward(image_features, text_features, logit_scale)
            clip_loss = self.clip_loss_weight * clip_loss

        caption_loss = self.caption_loss(
            logits.permute(0, 2, 1),
            labels,
        )
        caption_loss = caption_loss * self.caption_loss_weight

        if output_dict:
            return {"contrastive_loss": clip_loss, "caption_loss": caption_loss}

        return clip_loss, caption_loss


class DistillClipLoss(ClipLoss):

    def dist_loss(self, teacher_logits, student_logits):
        return -(teacher_logits.softmax(dim=1) * student_logits.log_softmax(dim=1)).sum(dim=1).mean(dim=0)

    def forward(
            self,
            image_features,
            text_features,
            logit_scale,
            dist_image_features,
            dist_text_features,
            dist_logit_scale,
            output_dict=False,
    ):
        logits_per_image, logits_per_text = \
            self.get_logits(image_features, text_features, logit_scale)

        dist_logits_per_image, dist_logits_per_text = \
            self.get_logits(dist_image_features, dist_text_features, dist_logit_scale)

        labels = self.get_ground_truth(image_features.device, logits_per_image.shape[0])

        contrastive_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        distill_loss = (
            self.dist_loss(dist_logits_per_image, logits_per_image) +
            self.dist_loss(dist_logits_per_text, logits_per_text)
        ) / 2

        if output_dict:
            return {"contrastive_loss": contrastive_loss, "distill_loss": distill_loss}

        return contrastive_loss, distill_loss


def neighbour_exchange(from_rank, to_rank, tensor, group=None):
    tensor_recv = torch.zeros_like(tensor)
    send_op = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor,
        to_rank,
        group=group,
    )
    recv_op = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_recv,
        from_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    return tensor_recv


def neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    tensor_from_left = torch.zeros_like(tensor_to_right)
    tensor_from_right = torch.zeros_like(tensor_to_left)
    send_op_left = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_left,
        left_rank,
        group=group,
    )
    send_op_right = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_right,
        right_rank,
        group=group,
    )
    recv_op_left = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_left,
        left_rank,
        group=group,
    )
    recv_op_right = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_right,
        right_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op_right, send_op_left, recv_op_right, recv_op_left])
    for req in reqs:
        req.wait()
    return tensor_from_right, tensor_from_left


class NeighbourExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, from_rank, to_rank, group, tensor):
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        return neighbour_exchange(from_rank, to_rank, tensor, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + (NeighbourExchange.apply(ctx.to_rank, ctx.from_rank, ctx.group, grad_output),)


def neighbour_exchange_with_grad(from_rank, to_rank, tensor, group=None):
    return NeighbourExchange.apply(from_rank, to_rank, group, tensor)


class NeighbourExchangeBidir(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left_rank, right_rank, group, tensor_to_left, tensor_to_right):
        ctx.group = group
        ctx.left_rank = left_rank
        ctx.right_rank = right_rank
        return neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=group)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None, None) + \
            NeighbourExchangeBidir.apply(ctx.right_rank, ctx.left_rank, ctx.group, *grad_outputs)


def neighbour_exchange_bidir_with_grad(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    return NeighbourExchangeBidir.apply(left_rank, right_rank, group, tensor_to_left, tensor_to_right)


class SigLipLoss(nn.Module):
    """ Sigmoid Loss for Language Image Pre-Training (SigLIP) - https://arxiv.org/abs/2303.15343

    @article{zhai2023sigmoid,
      title={Sigmoid loss for language image pre-training},
      author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
      journal={arXiv preprint arXiv:2303.15343},
      year={2023}
    }
    """
    def __init__(
            self,
            cache_labels=False,
            rank=0,
            world_size=1,
            bidir=True,
            use_horovod=False,
    ):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        assert not use_horovod  # FIXME need to look at hvd ops for ring transfers
        self.use_horovod = use_horovod
        self.bidir = bidir

        # cache state FIXME cache not currently used, worthwhile?
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, dtype, num_logits, negative_only=False) -> torch.Tensor:
        labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        if not negative_only:
            labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels
        return labels

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        logits = logit_scale * image_features @ text_features.T
        if logit_bias is not None:
            logits += logit_bias
        return logits

    def _loss(self, image_features, text_features, logit_scale, logit_bias=None, negative_only=False):
        logits = self.get_logits(image_features, text_features, logit_scale, logit_bias)
        labels = self.get_ground_truth(
            image_features.device,
            image_features.dtype,
            image_features.shape[0],
            negative_only=negative_only,
        )
        loss = -F.logsigmoid(labels * logits).sum() / image_features.shape[0]
        return loss

    def forward(self, image_features, text_features, logit_scale, logit_bias, output_dict=False):
        loss = self._loss(image_features, text_features, logit_scale, logit_bias)

        if self.world_size > 1:
            # exchange text features w/ neighbour world_size - 1 times
            right_rank = (self.rank + 1) % self.world_size
            left_rank = (self.rank - 1 + self.world_size) % self.world_size
            if self.bidir:
                text_features_to_right = text_features_to_left = text_features
                num_bidir, remainder = divmod(self.world_size - 1, 2)
                for i in range(num_bidir):
                    text_features_recv = neighbour_exchange_bidir_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_left,
                        text_features_to_right,
                    )

                    for f in text_features_recv:
                        loss += self._loss(
                            image_features,
                            f,
                            logit_scale,
                            logit_bias,
                            negative_only=True,
                        )
                    text_features_to_left, text_features_to_right = text_features_recv

                if remainder:
                    text_features_recv = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)

                    loss += self._loss(
                        image_features,
                        text_features_recv,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            else:
                text_features_to_right = text_features
                for i in range(self.world_size - 1):
                    text_features_from_left = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)

                    loss += self._loss(
                        image_features,
                        text_features_from_left,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
                    text_features_to_right = text_features_from_left

        return {"contrastive_loss": loss} if output_dict else loss


class FastCLIPLoss(nn.Module):
    def __init__(self,
                 data_size: int,
                 gamma: float,
                 gamma_schedule: str = "constant",
                 gamma_decay_epochs: int = -1,
                 rho: float = 8.0,
                 eps: float = 1e-14,
                 multiply_tau: bool = True,
                 cache_mask: bool = True,
                 device: torch.device = torch.device("cuda"),
                 ):
        """Create an instance of Global Contrastive Loss with global temperature parameter."""
        super(FastCLIPLoss, self).__init__()
        self.data_size = data_size
        self.gamma = 1.0
        self.gamma_orig = gamma
        self.gamma_schedule = gamma_schedule
        self.gamma_decay_epochs = gamma_decay_epochs
        if self.gamma_schedule != "none":
            assert self.gamma_decay_epochs > 0
        self.rho = rho
        self.eps = eps
        self.multiply_tau = multiply_tau
        self.cache_mask = cache_mask
        self.device = device

        self.u_im = torch.zeros(data_size, device=torch.device("cpu")).reshape(-1, 1)
        self.u_tt = torch.zeros(data_size, device=torch.device("cpu")).reshape(-1, 1)
        self.arange = {}
        self.mask = {}

        logging.info(f"data size: {data_size}, final gamma: {gamma}, gamma_schedule: {gamma_schedule}, "
                     f"gamma_decay_epochs: {self.gamma_decay_epochs}, rho: {rho}, eps: {self.eps}, "
                     f"multiply_tau: {self.multiply_tau}, cache_mask: {self.cache_mask}")

    def adjust_gamma(self, epoch: int):
        if epoch == 0:
            self.gamma = 1.0
        elif epoch >= self.gamma_decay_epochs:
            self.gamma = self.gamma_orig
        else:
            self.gamma = self.gamma_orig
            if self.gamma_schedule == "cosine":
                self.gamma = 0.5 * (1 + math.cos(math.pi * epoch / self.gamma_decay_epochs)) * \
                             (1 - self.gamma_orig) + self.gamma_orig
        logging.info(f"gamma: {self.gamma}")

    def adjust_hyperparams(self, epoch: int):
        self.adjust_gamma(epoch)

    def get_arange(self, length: int, offset: int):
        # here we assume arange is on self.device
        # the arange should be small in size, so we force caching it
        if offset not in self.arange.keys():
            self.arange[offset] = {}
        if length not in self.arange[offset].keys():
            self.arange[offset][length] = torch.arange(length, device=self.device) + offset
        return self.arange[offset][length]

    def get_mask(self, height: int, width: int, offset: int):
        """Return a height * width matrix, with diagonal [offset: offset + height, offset: offset + height]
            being 0 and the rest being 1
        """
        if not self.cache_mask or (height, width, offset) not in self.mask.keys():
            mask_inv = torch.nn.functional.one_hot(self.get_arange(height, offset), width).to(self.device)
            mask = 1 - mask_inv
            if self.cache_mask and (height, width, offset) not in self.mask.keys():
                self.mask[(height, width, offset)] = (mask, mask_inv)
        else:
            mask, mask_inv = self.mask[(height, width, offset)]
        return mask, mask_inv

    def pairwise_loss(self,
                      features1: Tuple[Tensor, Tensor],
                      features2: Tuple[Tensor, Tensor],
                      logit_scale_im: Tensor,
                      offset: int = 0,
                      sim: Optional[Tuple[Tensor, Tensor]] = None,
                      logit_scale_tt: Optional[Tensor] = None,
                      bounds: Optional[Tuple[Tensor, Tensor]] = None,
                      update_bounds: bool = True,
                      ):
        image_features1, text_features1 = features1[0], features1[1]
        image_features2, text_features2 = features2[0], features2[1]
        if logit_scale_tt is None:
            logit_scale_tt = logit_scale_im

        batch_size1 = image_features1.shape[0]  # b1
        batch_size2 = image_features2.shape[0]  # b2

        if sim is not None:
            sim_image, sim_text = sim[0], sim[1]
        else:
            sim_image = image_features1 @ text_features2.T  # shape [b1, b2]
            sim_text = text_features1 @ image_features2.T  # shape [b1, b2]
        diag_sim = torch.sum(torch.mul(image_features1, text_features1), dim=-1, keepdim=True)

        diff_image = (sim_image - diag_sim).mul(logit_scale_im)
        diff_text = (sim_text - diag_sim).mul(logit_scale_tt)
        bounds_image, bounds_text = None, None
        if bounds is not None:
            bounds_image, bounds_text = bounds[0], bounds[1]
            if update_bounds:
                bounds_image = torch.maximum(
                    bounds_image, torch.max(diff_image, dim=-1, keepdim=True).values.detach())
                bounds_text = torch.maximum(
                    bounds_text, torch.max(diff_text, dim=-1, keepdim=True).values.detach())
            diff_image = diff_image.sub(bounds_image)
            diff_text = diff_text.sub(bounds_text)
        exp_diff_image = torch.exp(diff_image)
        exp_diff_text = torch.exp(diff_text)

        if batch_size1 <= batch_size2:
            mask, mask_inv = self.get_mask(batch_size1, batch_size2, offset)
        else:
            mask, mask_inv = self.get_mask(batch_size2, batch_size1, offset)
            mask, mask_inv = mask.T, mask_inv.T
        exp_diff_image = torch.mul(exp_diff_image, mask)
        exp_diff_text = torch.mul(exp_diff_text, mask)

        if batch_size1 <= batch_size2:
            real_weights_sum = batch_size2 - 1
        else:
            real_weights_sum = batch_size2 * (batch_size1 - 1) / batch_size1
        loss_image = torch.sum(exp_diff_image, dim=-1, keepdim=True) / real_weights_sum
        loss_text = torch.sum(exp_diff_text, dim=-1, keepdim=True) / real_weights_sum

        return loss_image, loss_text, exp_diff_image, exp_diff_text, diff_image, diff_text, bounds_image, bounds_text

    def local(self,
              features: Tuple[Tensor, Tensor],
              indices: Tuple[Tensor, Tensor],
              remote_features: Tuple[Tensor, Tensor],
              logit_scale: Tensor,
              offset: int,
              ):
        image_ids, text_ids = indices[0], indices[1]
        u_im = self.u_im[image_ids].to(self.device)
        u_tt = self.u_tt[text_ids].to(self.device)

        loss1_im, loss1_tt, _, _, _, _, _, _ = self.pairwise_loss(
            features, remote_features, logit_scale, offset=offset)

        g_im = loss1_im.detach()
        g_tt = loss1_tt.detach()
        if self.gamma < 1.0:
            bad_im_idx = torch.nonzero(
                (u_im < 1e-35).logical_or(u_im.isinf()).logical_or(u_im.isnan()), as_tuple=True)[0]
            bad_tt_idx = torch.nonzero(
                (u_tt < 1e-35).logical_or(u_tt.isinf()).logical_or(u_tt.isnan()), as_tuple=True)[0]
        u_im = (1.0 - self.gamma) * u_im + self.gamma * g_im
        u_tt = (1.0 - self.gamma) * u_tt + self.gamma * g_tt
        if self.gamma < 1.0:
            if bad_im_idx.shape[0] > 0:
                u_im[bad_im_idx] = g_im[bad_im_idx].to(u_im.dtype)
            if bad_tt_idx.shape[0] > 0:
                u_tt[bad_tt_idx] = g_tt[bad_tt_idx].to(u_tt.dtype)
        self.u_im[image_ids] = u_im.to(torch.device("cpu"))
        self.u_tt[text_ids] = u_tt.to(torch.device("cpu"))

        image_ids = image_ids.unsqueeze(-1).to(device=self.device, dtype=image_ids.dtype, non_blocking=True)
        text_ids = text_ids.unsqueeze(-1).to(device=self.device, dtype=text_ids.dtype, non_blocking=True)

        return loss1_im, loss1_tt, u_im, u_tt, image_ids, text_ids

    def forward(self,
                features: Tuple[Tensor, Tensor],
                remote_features: Tuple[Tensor, Tensor],
                remote_u: Tuple[Tensor, Tensor],
                remote_ids: Tuple[Tensor, Tensor],
                loss1: Tuple[Tensor, Tensor],
                u: Tuple[Tensor, Tensor],
                logit_scale: Tensor,
                offset: int,
                output_dict: bool = False,
                **kwargs
                ):
        remote_u_im, remote_u_tt = remote_u[0], remote_u[1]
        remote_u_im_ids = remote_ids[0].squeeze(-1).to(torch.device("cpu"))
        remote_u_tt_ids = remote_ids[1].squeeze(-1).to(torch.device("cpu"))
        loss1_im, loss1_tt = loss1[0], loss1[1]
        u_im, u_tt = u[0], u[1]
        self.u_im[remote_u_im_ids] = remote_u_im.to(torch.device("cpu"))
        self.u_tt[remote_u_tt_ids] = remote_u_tt.to(torch.device("cpu"))

        loss2_im, loss2_tt, _, _, _, _, _, _ = self.pairwise_loss(remote_features, features, logit_scale, offset=offset)

        partial_grad1_im = loss1_im / (u_im + self.eps)
        partial_grad1_tt = loss1_tt / (u_tt + self.eps)
        partial_grad2_im = loss2_im / (remote_u_im + self.eps)
        partial_grad2_tt = loss2_tt / (remote_u_tt + self.eps)
        loss = (torch.mean(partial_grad1_im + partial_grad1_tt) + 
                torch.mean(partial_grad2_im + partial_grad2_tt)) / 2
        if self.multiply_tau:
            loss = loss / logit_scale.detach()
            loss = loss + self.rho / logit_scale
            loss = loss + torch.mean(torch.log(u_im) + torch.log(u_tt)) / 2 / logit_scale

        if output_dict:
            return {
                "contrastive_loss": loss,
            }
        else:
            return loss


class FastCLIPLossIndividual(FastCLIPLoss):
    def __init__(self,
                 data_size: int,
                 tau_init: float,
                 lr_tau: float,
                 beta1_tau: float = 0.9,
                 beta2_tau: float = 0.999,
                 eps_tau: float = 1e-8,
                 device: torch.device = torch.device("cuda"),
                 **kwargs
                 ):
        """Create an instance of Global Contrastive Loss with individual temperature parameters.
            This is a subclass of FastCLIPLoss, with additional parameters for individual temperature parameters.
        """
        super().__init__(data_size=data_size, device=device, **kwargs)
        self.tau_im = torch.ones(data_size, device=torch.device("cpu")).reshape(-1, 1) * tau_init
        self.tau_tt = torch.ones(data_size, device=torch.device("cpu")).reshape(-1, 1) * tau_init

        self.beta1_tau_orig = beta1_tau
        self.beta1_tau = 0.0
        self.beta2_tau_orig = beta2_tau
        self.beta2_tau = 0.0
        self.grad_clamp_tau = 5.0
        self.eps_tau = eps_tau
        self.epoch = 0

        self.m_grad_tau_im = torch.zeros(data_size, device=torch.device("cpu")).reshape(-1, 1)
        self.m_grad_tau_tt = torch.zeros(data_size, device=torch.device("cpu")).reshape(-1, 1)
        self.v_grad_tau_im = torch.zeros(data_size, device=torch.device("cpu")).reshape(-1, 1)
        self.v_grad_tau_tt = torch.zeros(data_size, device=torch.device("cpu")).reshape(-1, 1)

        self.tau_min, self.tau_max = 0.01, 1.0
        self.lr_tau_orig = lr_tau
        self.lr_tau = lr_tau

        self.bound_im = torch.zeros(data_size, device=torch.device("cpu")).reshape(-1, 1)
        self.bound_tt = torch.zeros(data_size, device=torch.device("cpu")).reshape(-1, 1)

        logging.info(f"beta1_tau: {self.beta1_tau_orig}, beta2_tau: {self.beta2_tau_orig}, eps_tau: {self.eps_tau}")

    def adjust_hyperparams(self, epoch: int):
        self.epoch = epoch
        self.adjust_gamma(epoch)
        # self.update_lr_tau(epoch)
        if epoch > 0:
            self.beta1_tau = self.beta1_tau_orig
            self.beta2_tau = self.beta2_tau_orig

    def local(self,
              features: Tuple[Tensor, Tensor],
              indices: Tuple[Tensor, Tensor],
              remote_features: Tuple[Tensor, Tensor],
              offset: int,
              ):
        image_ids, text_ids = indices[0], indices[1]
        bounds = (self.bound_im[image_ids].clone().to(self.device), self.bound_tt[text_ids].clone().to(self.device))
        u_im = self.u_im[image_ids].to(self.device)
        u_tt = self.u_tt[text_ids].to(self.device)
        tau_im = self.tau_im[image_ids].to(self.device)
        tau_tt = self.tau_tt[text_ids].to(self.device)
        m_grad_tau_im = self.m_grad_tau_im[image_ids].to(self.device)
        m_grad_tau_tt = self.m_grad_tau_tt[text_ids].to(self.device)
        v_grad_tau_im = self.v_grad_tau_im[image_ids].to(self.device)
        v_grad_tau_tt = self.v_grad_tau_tt[text_ids].to(self.device)

        loss1_im, loss1_tt, exp_diff_im, exp_diff_tt, diff_im, diff_tt, new_bound_im, new_bound_tt = self.pairwise_loss(
            features, remote_features, 1.0/tau_im, offset=offset, logit_scale_tt=1.0/tau_tt, bounds=bounds)
        assert new_bound_im is not None and new_bound_tt is not None

        g_im = loss1_im.detach()
        g_tt = loss1_tt.detach()
        if self.gamma < 1.0:
            bad_im_idx = torch.nonzero(
                (u_im < 1e-35).logical_or(u_im.isinf()).logical_or(u_im.isnan()), as_tuple=True)[0]
            bad_tt_idx = torch.nonzero(
                (u_tt < 1e-35).logical_or(u_tt.isinf()).logical_or(u_tt.isnan()), as_tuple=True)[0]
        u_im = (1.0 - self.gamma) * u_im * torch.exp(bounds[0] - new_bound_im) + self.gamma * g_im
        u_tt = (1.0 - self.gamma) * u_tt * torch.exp(bounds[1] - new_bound_tt) + self.gamma * g_tt
        if self.gamma < 1.0:
            if bad_im_idx.shape[0] > 0:
                u_im[bad_im_idx] = g_im[bad_im_idx].to(u_im.dtype)
            if bad_tt_idx.shape[0] > 0:
                u_tt[bad_tt_idx] = g_tt[bad_tt_idx].to(u_tt.dtype)

        batch_size = remote_features[0].shape[0] - 1
        # note that here diff is subtracted by new_bounds in pairwise_loss()
        # here we do not divide the gradient by dataset size,
        # since it is equivalent to dividing lr_tau by dataset size
        grad_tau_im = (-1 * exp_diff_im.mul(diff_im.add(new_bound_im)).sum(dim=-1, keepdim=True).div(u_im + self.eps).div(batch_size)
                       + torch.log(u_im + self.eps) + new_bound_im + self.rho).detach().clamp_(min=-self.grad_clamp_tau, max=self.grad_clamp_tau)
        grad_tau_tt = (-1 * exp_diff_tt.mul(diff_tt.add(new_bound_tt)).sum(dim=-1, keepdim=True).div(u_tt + self.eps).div(batch_size)
                       + torch.log(u_tt + self.eps) + new_bound_tt + self.rho).detach().clamp_(min=-self.grad_clamp_tau, max=self.grad_clamp_tau)

        m_grad_tau_im = self.beta1_tau * m_grad_tau_im + (1.0 - self.beta1_tau) * grad_tau_im
        m_grad_tau_tt = self.beta1_tau * m_grad_tau_tt + (1.0 - self.beta1_tau) * grad_tau_tt
        v_grad_tau_im = self.beta2_tau * v_grad_tau_im + (1.0 - self.beta2_tau) * grad_tau_im ** 2
        v_grad_tau_tt = self.beta2_tau * v_grad_tau_tt + (1.0 - self.beta2_tau) * grad_tau_tt ** 2
        m_hat_grad_tau_im = m_grad_tau_im / (1.0 - self.beta1_tau ** (self.epoch + 1))
        m_hat_grad_tau_tt = m_grad_tau_tt / (1.0 - self.beta1_tau ** (self.epoch + 1))
        v_hat_grad_tau_im = v_grad_tau_im / (1.0 - self.beta2_tau ** (self.epoch + 1))
        v_hat_grad_tau_tt = v_grad_tau_tt / (1.0 - self.beta2_tau ** (self.epoch + 1))

        lr_tau_im = self.get_lr_tau(tau_im)
        tau_im = (tau_im - lr_tau_im * m_hat_grad_tau_im / (v_hat_grad_tau_im + self.eps_tau)).clamp_(min=self.tau_min, max=self.tau_max)
        lr_tau_tt = self.get_lr_tau(tau_tt)
        tau_tt = (tau_tt - lr_tau_tt * m_hat_grad_tau_tt / (v_hat_grad_tau_tt + self.eps_tau)).clamp_(min=self.tau_min, max=self.tau_max)
        old_tau_im = tau_im.clone()
        old_tau_tt = tau_tt.clone()

        self.u_im[image_ids] = u_im.to(torch.device("cpu"))
        self.u_tt[text_ids] = u_tt.to(torch.device("cpu"))
        self.m_grad_tau_im[image_ids] = m_grad_tau_im.to(torch.device("cpu"))
        self.m_grad_tau_tt[text_ids] = m_grad_tau_tt.to(torch.device("cpu"))
        self.v_grad_tau_im[image_ids] = v_grad_tau_im.to(torch.device("cpu"))
        self.v_grad_tau_tt[text_ids] = v_grad_tau_tt.to(torch.device("cpu"))
        self.tau_im[image_ids] = tau_im.to(torch.device("cpu"))
        self.tau_tt[text_ids] = tau_tt.to(torch.device("cpu"))
        self.bound_im[image_ids] = new_bound_im.to(torch.device("cpu"))
        self.bound_tt[text_ids] = new_bound_tt.to(torch.device("cpu"))

        image_ids = image_ids.unsqueeze(-1).to(device=self.device, dtype=torch.int64, non_blocking=True)
        text_ids = text_ids.unsqueeze(-1).to(device=self.device, dtype=torch.int64, non_blocking=True)

        return loss1_im, loss1_tt, u_im, u_tt, old_tau_im, old_tau_tt, new_bound_im, new_bound_tt, image_ids, text_ids

    def forward(self,
                features: Tuple[Tensor, Tensor],
                remote_features: Tuple[Tensor, Tensor],
                remote_u: Tuple[Tensor, Tensor],
                remote_tau: Tuple[Tensor, Tensor],
                remote_bounds: Tuple[Tensor, Tensor],
                remote_ids: Tuple[Tensor, Tensor],
                loss1: Tuple[Tensor, Tensor],
                u: Tuple[Tensor, Tensor],
                offset: int,
                output_dict: bool = False,
                **kwargs
                ):
        remote_u_im, remote_u_tt = remote_u[0], remote_u[1]
        remote_u_im_ids = remote_ids[0].squeeze(-1).to(torch.device("cpu"))
        remote_u_tt_ids = remote_ids[1].squeeze(-1).to(torch.device("cpu"))
        loss1_im, loss1_tt = loss1[0], loss1[1]
        u_im, u_tt = u[0], u[1]
        self.u_im[remote_u_im_ids] = remote_u_im.to(torch.device("cpu"))
        self.u_tt[remote_u_tt_ids] = remote_u_tt.to(torch.device("cpu"))
        remote_tau_im, remote_tau_tt = remote_tau[0], remote_tau[1]

        loss2_im, loss2_tt, _, _, _, _, _, _ = self.pairwise_loss(
            remote_features, features, 1.0/remote_tau_im, offset=offset,
            logit_scale_tt=1.0/remote_tau_tt, bounds=remote_bounds, update_bounds=False)

        partial_grad1_im = loss1_im / (u_im + self.eps)
        partial_grad1_tt = loss1_tt / (u_tt + self.eps)
        partial_grad2_im = loss2_im / (remote_u_im + self.eps)
        partial_grad2_tt = loss2_tt / (remote_u_tt + self.eps)
        loss = (torch.mean(partial_grad1_im + partial_grad1_tt)
                + torch.mean(partial_grad2_im + partial_grad2_tt)) / 2

        if output_dict:
            return {
                "contrastive_loss": loss,
            }
        else:
            return loss
