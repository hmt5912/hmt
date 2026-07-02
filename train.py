import collections
import math
import statistics
from functools import reduce
import os


import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch import distributed
from torch.nn import functional as F
import numpy as np
from PIL import Image
from torch.autograd import Variable
import torchvision
# import matplotlib.pyplot as plt
from utils.utils import Label2Color, color_map
import pandas as pd
from tqdm import tqdm

from utils import get_regularizer
from utils.loss import (NCA, BCESigmoid, BCEWithLogitsLossWithIgnoreIndex,
                        criterion_self_entropy,
                        BCEWithLogitsLossWithIgnoreIndexSoftLabel,
                        ExcludedKnowledgeDistillationLoss, FocalLoss,
                        FocalLossNew, IcarlLoss, KnowledgeDistillationLoss,
                        UnbiasedCrossEntropy,
                        UnbiasedKnowledgeDistillationLoss, UnbiasedNCA,
                        MyselfKnowledgeDistillationLoss,
                        soft_crossentropy,loss_relation_consistency,CombinedLoss1,DiceLoss)
from utils.TaskMemory import compute_task_relevance
from collections import deque
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings(
    "ignore",
    message=r".*inplace_abn_sync is being called, but torch\.distributed is not initialized.*"
)

def _to_np_uint8(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    x = np.array(x)
    return x.astype(np.uint8)

def save_label_png(label_hw, path, palette=None, ignore=255):

    lab = _to_np_uint8(label_hw)
    img = Image.fromarray(lab, mode="P")
    if palette is not None:
        img.putpalette(palette)
    img.save(path)

def save_mask_png(mask_hw, path):
    if torch.is_tensor(mask_hw):
        mask_hw = mask_hw.detach().cpu().numpy()
    mask = (mask_hw > 0).astype(np.uint8) * 255
    Image.fromarray(mask, mode="L").save(path)


class Trainer:

    def __init__(self, model, model_old, device, rank, opts, trainer_state=None, classes=None, step=0,writer=None):

        self.model_old = model_old
        self.model = model
        self.device = device
        self.rank = rank
        self.step = step

        self.mem_size = opts.mem_size
        self.init_portion = opts.init_portion
        self.max_portion = opts.max_portion
        self.portion_step = opts.portion_step
        self.classes = classes
        self.soft_param = opts.soft_param
        self.regular_param = opts.regular_param
        self.batch_size = opts.batch_size
        self.inital_nb_classes = opts.inital_nb_classes
        self.method = opts.incremental_method
        self.scaler = GradScaler()
        self.writer = writer
        self.global_step = 0
        self.opts = opts

        if classes is not None:
            new_classes = classes[-1]
            tot_classes = reduce(lambda a, b: a + b, classes)
            self.old_classes = tot_classes - new_classes
            self.nb_classes = opts.num_classes
            self.nb_current_classes = tot_classes
            self.nb_new_classes = new_classes
        else:
            self.old_classes = 0
            self.nb_classes = None

        reduction = 'none'

        self.bce = opts.bce or opts.icarl
        if self.bce:
            self.criterion = BCEWithLogitsLossWithIgnoreIndex(reduction=reduction)
        elif opts.unce and self.old_classes != 0:
            self.criterion = UnbiasedCrossEntropy(
                old_cl=self.old_classes, ignore_index=255, reduction=reduction
            )
            self.lossresult = nn.L1Loss(size_average=True, reduction=True)
        elif opts.nca and self.old_classes != 0:
            self.criterion = UnbiasedNCA(
                old_cl=self.old_classes,
                ignore_index=255,
                reduction=reduction,
                scale=model.module.scalar,
                margin=opts.nca_margin
            )
        elif opts.nca:
            self.criterion = NCA(
                scale=model.module.scalar,
                margin=opts.nca_margin,
                ignore_index=255,
                reduction=reduction
            )
        elif opts.focal_loss:
            focal = FocalLoss(ignore_index=255, reduction=reduction, alpha=opts.alpha,
                                       gamma=opts.focal_loss_gamma)
            dice = DiceLoss(
                num_classes=self.nb_current_classes,
                ignore_index=255
            )

            self.criterion = CombinedLoss1(
                main_loss=focal,
                dice_loss=dice,
                dice_weight=0.5
            )
            self.lossresult = nn.L1Loss(size_average=True, reduction=True)
        elif opts.focal_loss_new:
            self.criterion = FocalLossNew(ignore_index=255, reduction=reduction, index=self.old_classes,
                                          alpha=opts.alpha, gamma=opts.focal_loss_gamma)
        else:
            ce = nn.CrossEntropyLoss(ignore_index=255, reduction=reduction)
            dice = DiceLoss(num_classes=self.nb_current_classes, ignore_index=255)

            self.criterion = CombinedLoss1(ce, dice, dice_weight=1.0)
            self.lossresult = nn.L1Loss(size_average=True, reduction=True)

        if opts.out_dis and step > 0:
            self.criterion_soft_label = BCEWithLogitsLossWithIgnoreIndexSoftLabel(reduction=reduction, ignore_index=255)

        self.lde = opts.loss_de
        self.lde_flag = self.lde > 0. and model_old is not None
        self.lde_loss = nn.MSELoss()

        self.lkd = opts.loss_kd
        self.local_rank = opts.local_rank
        self.lkd_mask = opts.kd_mask
        self.kd_mask_adaptative_factor = opts.kd_mask_adaptative_factor
        self.lkd_flag = self.lkd > 0. and model_old is not None
        self.kd_need_labels = False
        if opts.unkd:
            self.lkd_loss = UnbiasedKnowledgeDistillationLoss(reduction="none", alpha=opts.alpha)
        elif opts.myselfkd:
            self.lkd_loss = MyselfKnowledgeDistillationLoss(reduction="none", alpha=opts.alpha)
        elif opts.kd_bce_sig:
            self.lkd_loss = BCESigmoid(reduction="none", alpha=opts.alpha, shape=opts.kd_bce_sig_shape)
        elif opts.exkd_gt and self.old_classes > 0 and self.step > 0 and self.model_old is not None:
            self.lkd_loss = ExcludedKnowledgeDistillationLoss(
                reduction='none', index_new=self.old_classes, new_reduction="gt",
                initial_nb_classes=opts.inital_nb_classes,
                temperature_semiold=opts.temperature_semiold
            )
            self.kd_need_labels = True
        elif opts.exkd_sum and self.old_classes > 0 and self.step > 0 and self.model_old is not None:
            self.lkd_loss = ExcludedKnowledgeDistillationLoss(
                reduction='none', index_new=self.old_classes, new_reduction="sum",
                initial_nb_classes=opts.inital_nb_classes,
                temperature_semiold=opts.temperature_semiold
            )
            self.kd_need_labels = True
        else:
            self.lkd_loss = KnowledgeDistillationLoss(alpha=opts.alpha)

        self.icarl_combined = False
        self.icarl_only_dist = False
        if opts.icarl:
            self.icarl_combined = not opts.icarl_disjoint and model_old is not None
            self.icarl_only_dist = opts.icarl_disjoint and model_old is not None
            if self.icarl_combined:
                self.licarl = nn.BCEWithLogitsLoss(reduction='mean')
                self.icarl = opts.icarl_importance
            elif self.icarl_only_dist:
                self.licarl = IcarlLoss(reduction='mean', bkg=opts.icarl_bkg)
        self.icarl_dist_flag = self.icarl_only_dist or self.icarl_combined

        regularizer_state = trainer_state['regularizer'] if trainer_state is not None else None
        self.regularizer = get_regularizer(model, model_old, device, opts, regularizer_state)
        self.regularizer_flag = self.regularizer is not None
        self.reg_importance = opts.reg_importance

        self.ret_intermediate = self.lde or (opts.pod is not None)

        self.pseudo_labeling = opts.pseudo
        self.threshold = opts.threshold
        self.step_threshold = opts.step_threshold
        self.ce_on_pseudo = opts.ce_on_pseudo
        self.pseudo_nb_bins = opts.pseudo_nb_bins
        self.pseudo_soft = opts.pseudo_soft
        self.pseudo_soft_factor = opts.pseudo_soft_factor
        self.pseudo_ablation = opts.pseudo_ablation
        self.classif_adaptive_factor = opts.classif_adaptive_factor
        self.classif_adaptive_min_factor = opts.classif_adaptive_min_factor

        self.kd_new = opts.kd_new
        self.pod = opts.pod
        self.pod_options = opts.pod_options if opts.pod_options is not None else {}
        self.pod_factor = opts.pod_factor
        self.soft_factor = opts.soft_factor
        self.pod_prepro = opts.pod_prepro
        self.use_pod_schedule = not opts.no_pod_schedule
        self.pod_deeplab_mask = opts.pod_deeplab_mask
        self.pod_deeplab_mask_factor = opts.pod_deeplab_mask_factor
        self.pod_apply = opts.pod_apply
        self.pod_interpolate_last = opts.pod_interpolate_last
        self.deeplab_mask_downscale = opts.deeplab_mask_downscale
        self.spp_scales = opts.spp_scales
        self.pod_logits = opts.pod_logits
        self.pod_large_logits = opts.pod_large_logits

        self.align_weight = opts.align_weight
        self.align_weight_frequency = opts.align_weight_frequency

        self.dataset = opts.dataset

        self.entropy_min = opts.entropy_min

        self.kd_scheduling = opts.kd_scheduling

        self.sample_weights_new = opts.sample_weights_new

        self.temperature_apply = opts.temperature_apply
        self.temperature = opts.temperature

        self.ce_on_new = opts.ce_on_new
        self.out_dis = opts.out_dis
        self.pseudo_proto = opts.pseudo_proto
        self.feat_dim = opts.feat_dim
        self.no_mask = opts.no_mask
        self.overlap = opts.overlap
        self.proto_temperature = opts.proto_temperature
        self.llc_enable = getattr(opts, "llc_enable", True)
        self.llc_k = int(getattr(opts, "llc_k", 50))
        self.llc_tau = float(getattr(opts, "llc_tau", 0.5))
        self.llc_min_region_px = int(getattr(opts, "llc_min_region_px", 3))
        self.llc_bank_size = int(getattr(opts, "llc_bank_size", 5000))
        self.gmm_tol = float(getattr(opts, "GMMtol", 1e-4))
        self.gmm_reg = float(getattr(opts, "GMMreg_covar", 1e-6))
        self.gmm_iter = int(getattr(opts, "GMMmax_iter", 50))

        self.region_bank = RegionMemoryBank(
            max_regions=self.llc_bank_size,
            feat_dim=int(getattr(opts, "feat_dim", 256)),
            device=self.device
        )

    def before(self, cur_epoch, train_loader):
        self.target_portion = min(self.init_portion + self.portion_step * cur_epoch, self.max_portion)
        if self.step == 0 or self.model_old is None:
            return
        if self.pseudo_labeling is None:
            return
        if self.pseudo_labeling.split("_")[0] == "median" and self.step > 0:

            self.thresholds, _ = self.find_median(train_loader, self.device)
        elif self.pseudo_labeling.split("_")[0] == "entropy" and self.step > 0:

            self.thresholds, self.max_entropy = self.find_median(
                train_loader, self.target_portion, self.device, mode="entropy"
            )
        elif self.pseudo_labeling.split("_")[0] == "adapt" and self.step > 0:

            self.thresholds, self.max_entropy = self.find_median(
                train_loader, self.target_portion, self.device, mode="adapt"
            )

    def efficient_step_old_class_weight(self, output, label):
        pred = torch.sigmoid(output)
        labels_new = torch.where(label != 255, label, output.shape[1])
        target = F.one_hot(labels_new, output.shape[1] + 1).float().permute(0, 3, 1, 2)
        target = target[:, :output.shape[1], :, :]
        class_mask = F.one_hot(labels_new, output.shape[1] + 1).float().permute(0, 3, 1, 2)
        class_mask = class_mask[:, :output.shape[1], :, :]
        g = torch.abs(pred.detach() - target)
        if self.step > 0:
            z = torch.div(self.old_classes, self.nb_current_classes)
            z = g.clone().fill_(z)
            g = torch.pow(g, z)
        g = (g * class_mask).sum(1)

        if self.old_classes != 0:

            ids = torch.where(label >= self.inital_nb_classes, label, label.clone().fill_(-1))
            index1 = torch.eq(ids, -1).float()
            index2 = torch.ne(ids, -1).float()
            if index1.sum() != 0:
                w1 = torch.div(g * index1, (g * index1).sum() / index1.sum())
            else:
                w1 = g.clone().fill_(0.)
            if index2.sum() != 0:
                classes_sum = []
                for c in range(len(self.classes)):
                    classes_sum.append(self.classes[c])
                    if c > 0:
                        classes_sum[c] = classes_sum[c - 1] + self.classes[c]
                w2 = g.clone().fill_(0.)
                if self.step == 1:
                    w2 = index2
                else:
                    for j in range(0, len(self.classes) - 2):
                        a = g.clone().fill_(0.)
                        for i in range(classes_sum[0], classes_sum[j + 1]):
                            b = torch.eq(ids, i).float()
                            a = a + b
                        if a.sum() != 0:
                            m = torch.div(g * a, (g * a).sum() / a.sum())
                        else:
                            m = g.clone().fill_(0.)
                        w2 = w2 + m

                    a = g.clone().fill_(0.)
                    for i in range(classes_sum[-2], classes_sum[-1]):
                        b = torch.eq(ids, i).float()
                        a = a + b
                    if a.sum() != 0:
                        m = a
                    else:
                        m = g.clone().fill_(0.)
                    w2 = w2 + m
            else:
                w2 = g.clone().fill_(0.)
            w = w1 + w2

        else:
            w = g.clone().fill_(1.)

        return w

    def efficient_dis_old_class_weight(self, output, label):
        pred = torch.sigmoid(output)
        labels_new = torch.where(label != 255, label, output.shape[1])
        target = F.one_hot(labels_new, output.shape[1] + 1).float().permute(0, 3, 1, 2)
        target = target[:, :output.shape[1], :, :]
        class_mask = F.one_hot(labels_new, output.shape[1] + 1).float().permute(0, 3, 1, 2)
        class_mask = class_mask[:, :output.shape[1], :, :]
        g = torch.abs(pred.detach() - target)
        if self.step > 0:
            z = torch.div(self.old_classes, self.nb_current_classes)
            z = g.clone().fill_(z)
            g = torch.pow(g, z)
        g = (g * class_mask).sum(1)

        if self.old_classes != 0:

            ids = torch.where(label >= self.inital_nb_classes, label, label.clone().fill_(-1))
            index1 = torch.eq(ids, -1).float()
            index2 = torch.ne(ids, -1).float()

            if index1.sum() != 0:
                w1 = torch.div(g * index1, (g * index1).sum() / index1.sum())
            else:
                w1 = g.clone().fill_(0.)
            if index2.sum() != 0:
                classes_sum = []
                for c in range(len(self.classes)):
                    classes_sum.append(self.classes[c])
                    if c > 0:
                        classes_sum[c] = classes_sum[c - 1] + self.classes[c]
                w2 = g.clone().fill_(0.)
                if self.step == 1:
                    w2 = index2
                else:
                    for j in range(0, len(self.classes) - 2):
                        a = g.clone().fill_(0.)
                        for i in range(classes_sum[0], classes_sum[j + 1]):
                            b = torch.eq(ids, i).float()
                            a = a + b
                        if a.sum() != 0:
                            m = torch.div(g * a, (g * a).sum() / a.sum())
                        else:
                            m = g.clone().fill_(0.)
                        w2 = w2 + m

                    a = g.clone().fill_(0.)
                    for i in range(classes_sum[-2], classes_sum[-1]):
                        b = torch.eq(ids, i).float()
                        a = a + b
                    if a.sum() != 0:
                        m = a
                    else:
                        m = g.clone().fill_(0.)
                    w2 = w2 + m
            else:
                w2 = g.clone().fill_(0.)
            w = w1 + w2

        else:
            w = g.clone().fill_(1.)

        weight_aver = torch.zeros(self.nb_current_classes, dtype=torch.float32).to(
            self.device
        )
        weight_num = torch.zeros(self.nb_current_classes, dtype=torch.float32).to(
            self.device
        )
        for i in range(0, self.nb_current_classes):
            mask_weight = label == i
            weight_aver[i] = w[mask_weight].sum()
            weight_num[i] = mask_weight.sum()
        for i in range(0, self.nb_current_classes):
            if weight_num[i] == 0:
                weight_aver[i] = 0
            else:
                weight_aver[i] = weight_aver[i] / weight_num[i]
        return weight_aver

    def reg_pesudo_label(self, output, label, num_classes):
        output = torch.softmax(output, dim=1)
        loss = -(output * torch.log(output)).mean(dim=1)
        return loss

    def train(self, cur_epoch, optim, train_loader, scheduler=None, print_int=10):

        def _extract_oldhead_grad_signature(model_wrapped, step: int) -> torch.Tensor:
            m = model_wrapped.module if hasattr(model_wrapped, "module") else model_wrapped

            cls_mod = getattr(m, "cls", None)
            if cls_mod is None:
                return torch.zeros(1)

            if isinstance(cls_mod, (nn.ModuleList, list, tuple)):
                used_heads = cls_mod if step == 0 else cls_mod[:-1]
                params = []
                for head in used_heads:
                    params.extend(list(head.parameters()))
            else:
                params = list(cls_mod.parameters())

            grads = []
            for p in params:
                if p.grad is None:
                    grads.append(torch.zeros(p.numel(), dtype=torch.float32))
                else:
                    grads.append(p.grad.detach().reshape(-1).float().cpu())

            if len(grads) == 0:
                return torch.zeros(1)

            g = torch.cat(grads, 0)
            return F.normalize(g, dim=0)

        if self.rank == 0:
            print(f"Pseudo labeling is: {self.pseudo_labeling}")
            print("Epoch %d, lr = %f" % (cur_epoch + 1, optim.param_groups[0]['lr']))

        device = self.device
        model = self.model
        criterion = self.criterion
        lossresult = self.lossresult

        if hasattr(model, 'module'):
            model.module.in_eval = False
        else:
            model.in_eval = False
        if self.model_old is not None:
            self.model_old.in_eval = False

        epoch_loss = 0.0
        reg_loss = 0.0
        interval_loss = 0.0
        lkd = torch.tensor(0.)
        lde = torch.tensor(0.)
        l_icarl = torch.tensor(0.)
        l_reg = torch.tensor(0.)
        pod_loss = torch.tensor(0.)
        loss_entmin = torch.tensor(0.)
        loss_soft_label = torch.tensor(0.)
        Regularizer_soft = torch.tensor(0.)
        loss_dis = torch.tensor(0.)

        sample_weights = None

        train_loader.sampler.set_epoch(cur_epoch)
        G = []
        model.train()
        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            disable=(self.rank != 0),
            dynamic_ncols=True,
            desc=f"Epoch {cur_epoch + 1}"
        )
        for cur_step, (images, labels, raw_labels) in pbar:

            outputs_old = None
            features_old = None

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)


            original_labels = labels.clone()

            old_cls_aver = torch.zeros(self.nb_current_classes, dtype=torch.float32).to(
                self.device
            )
            old_cls_num = torch.zeros(self.nb_current_classes, dtype=torch.float32).to(
                self.device
            )
            new_cls_num = torch.zeros(self.nb_current_classes, dtype=torch.float32).to(
                self.device
            )
            new_cls_aver = torch.zeros(self.nb_current_classes, dtype=torch.float32).to(
                self.device
            )

            if (
                    self.lde_flag or self.lkd_flag or self.icarl_dist_flag or self.pod is not None or
                    self.pseudo_labeling is not None
            ) and self.model_old is not None:
                with torch.no_grad():
                    outputs_old, features_old = self.model_old(
                        images, ret_intermediate=self.ret_intermediate
                    )


            classif_adaptive_factor = 1.0
            if self.step > 0 and self.model_old is not None:
                mask_background = labels < self.old_classes

                if self.pseudo_labeling == "naive":
                    labels[mask_background] = outputs_old.argmax(dim=1)[mask_background]
                elif self.pseudo_labeling is not None and self.pseudo_labeling.startswith(
                        "threshold_"
                ):
                    threshold = float(self.pseudo_labeling.split("_")[1])
                    probs = torch.softmax(outputs_old, dim=1)
                    pseudo_labels = probs.argmax(dim=1)
                    pseudo_labels[probs.max(dim=1)[0] < threshold] = 255
                    labels[mask_background] = pseudo_labels[mask_background]
                elif self.pseudo_labeling == "confidence":
                    probs_old = torch.softmax(outputs_old, dim=1)
                    labels[mask_background] = probs_old.argmax(dim=1)[mask_background]
                    sample_weights = torch.ones_like(labels).to(device, dtype=torch.float32)
                    sample_weights[mask_background] = probs_old.max(dim=1)[0][mask_background]
                elif self.pseudo_labeling == "median":
                    probs = torch.softmax(outputs_old, dim=1)
                    max_probs, pseudo_labels = probs.max(dim=1)
                    mask_valid_pseudo = max_probs > self.thresholds[pseudo_labels]
                    pseudo_labels[max_probs < self.thresholds[pseudo_labels]] = 255
                    labels[mask_background] = pseudo_labels[mask_background]
                    if self.classif_adaptive_factor:
                        num = (mask_valid_pseudo & mask_background).float().sum(dim=(1, 2))
                        den = mask_background.float().sum(dim=(1, 2))
                        classif_adaptive_factor = num / (den + 1e-6)
                        classif_adaptive_factor = classif_adaptive_factor[:, None, None]

                        if self.classif_adaptive_min_factor:
                            classif_adaptive_factor = classif_adaptive_factor.clamp(
                                min=self.classif_adaptive_min_factor)
                elif self.pseudo_labeling == "entropy":
                    probs = torch.softmax(outputs_old, dim=1)
                    max_probs, pseudo_labels = probs.max(dim=1)

                    ent_norm = entropy(probs) / self.max_entropy

                    gamma_adapt = self.thresholds
                    if (self.rank == 0) and (cur_step == 0):
                        tqdm.write(f"[DBG] thresholds: {gamma_adapt.detach().cpu()}")
                        tqdm.write(f"[DBG] thresholds len = {gamma_adapt.numel()}")

                    R_cont = getattr(self.opts, "class_relevance_cont", None)

                    if (
                            self.step > 0
                            and R_cont is not None
                            and torch.is_tensor(R_cont)
                            and R_cont.numel() == self.thresholds.numel()
                    ):
                        alpha = float(getattr(self.opts, "rel_alpha", 0.3))
                        gamma_min = float(getattr(self.opts, "rel_gamma_min", 0.0))
                        gamma_max = float(getattr(self.opts, "rel_gamma_max", 1.0))

                        R_cont = R_cont.to(self.thresholds.device, dtype=self.thresholds.dtype).clone()

                        R_cont[0] = 0.0
                        old_end = int(self.old_classes)
                        if (self.rank == 0) and (cur_step == 0):
                            tqdm.write(f"[DBG] old_classes = {self.old_classes}")
                            tqdm.write(f"[DBG] old_end = {old_end}")

                        gamma_adapt = self.thresholds.clone()
                        gamma_adapt[:old_end] = torch.clamp(
                            self.thresholds[:old_end] + alpha * R_cont[:old_end],
                            min=gamma_min,
                            max=gamma_max
                        )
                        if (self.rank == 0) and (cur_step == 0):
                            tqdm.write(f"[DBG] 文本调整后thresholds: {gamma_adapt.detach().cpu()}")
                            tqdm.write(f"[DBG] 文本调整后thresholds len = {gamma_adapt.numel()}")

                    # per-pixel threshold lookup by pseudo label
                    thr_map = gamma_adapt[pseudo_labels]
                    # thr_map 本质上就是一张“阈值图”：形状和分割 mask 一样（[B, H, W]），
                    # 每个像素位置存的是“这个像素对应伪标签类别的阈值”。
                    # [B,H,W]例如某像素伪标签=7，就用 gamma_adapt[7] 当它的阈值。

                    mask_valid_pseudo = ent_norm < thr_map  # 像素的归一化熵 ent_norm 小于该像素阈值 thr_map
                    mask_unconfident_pseudo = ~mask_valid_pseudo



                    if self.pseudo_soft is None:
                        labels[~mask_valid_pseudo & mask_background] = 255
                        if self.pseudo_ablation is None:
                            labels[mask_valid_pseudo & mask_background] = pseudo_labels[mask_valid_pseudo &
                                                                                        mask_background]

                            labels_pseudo = labels.clone().fill_(0)
                            labels_pseudo[mask_valid_pseudo & mask_background] = labels[
                                mask_valid_pseudo & mask_background]
                            labels_unconfident = labels.clone().fill_(0)
                            labels_unconfident[mask_unconfident_pseudo & mask_background] = labels[
                                mask_unconfident_pseudo & mask_background]
                        elif self.pseudo_ablation == "corrected_errors":
                            pass
                        elif self.pseudo_ablation == "removed_errors":
                            pseudo_error_mask = labels != pseudo_labels
                            kept_pseudo_labels = mask_valid_pseudo & mask_background & ~pseudo_error_mask
                            removed_pseudo_labels = mask_valid_pseudo & mask_background & pseudo_error_mask

                            labels[kept_pseudo_labels] = pseudo_labels[kept_pseudo_labels]
                            labels[removed_pseudo_labels] = 255
                        else:
                            raise ValueError(f"Unknown type of pseudo_ablation={self.pseudo_ablation}")
                    elif self.pseudo_soft == "soft_uncertain":
                        labels[mask_valid_pseudo & mask_background] = pseudo_labels[mask_valid_pseudo &
                                                                                    mask_background]

                    for i in range(0, self.nb_current_classes):
                        mask_old = labels == i
                        old_cls_aver[i] = max_probs[mask_old].sum()  # 概率值
                        old_cls_num[i] = mask_old.sum()  # 像素数量
                    for i in range(0, self.nb_current_classes):
                        if old_cls_num[i] == 0:
                            old_cls_aver[i] = 0
                        else:
                            old_cls_aver[i] = old_cls_aver[i] / old_cls_num[i]

                    if self.classif_adaptive_factor:
                        num = (mask_valid_pseudo & mask_background).float().sum(dim=(1, 2))
                        den = mask_background.float().sum(dim=(1, 2))
                        classif_adaptive_factor = num / (den + 1e-6)
                        classif_adaptive_factor = classif_adaptive_factor[:, None, None]

                        if self.classif_adaptive_min_factor:
                            classif_adaptive_factor = classif_adaptive_factor.clamp(
                                min=self.classif_adaptive_min_factor)

                    if (self.rank == 0) and (cur_step < 3):
                        vis_dir = f"./debug_pseudo_2.7rib5-0/step{self.step}_ep{cur_epoch + 1}_b{cur_step}"
                        os.makedirs(vis_dir, exist_ok=True)

                        B = images.shape[0]
                        k = min(4, B)

                        rand_idx = torch.randperm(B, device=images.device)[:k].tolist()

                        pal = None
                        try:
                            cm = color_map('amos')
                            pal = cm.flatten().tolist()
                        except Exception:
                            pal = None

                        for b in rand_idx:
                            pl = pseudo_labels[b]
                            vm = mask_valid_pseudo[b]
                            final_lab = labels[b]
                            gt_lab = raw_labels[b]

                            save_label_png(pl, os.path.join(vis_dir, f"pseudo_label_s{cur_step}_idx{b}.png"),
                                           palette=pal)
                            save_mask_png(vm, os.path.join(vis_dir, f"valid_mask_s{cur_step}_idx{b}.png"))
                            save_label_png(final_lab,
                                           os.path.join(vis_dir, f"train_label_after_replace_s{cur_step}_idx{b}.png"),
                                           palette=pal)
                            save_label_png(gt_lab, os.path.join(vis_dir, f"gt_original_s{cur_step}_idx{b}.png"),
                                           palette=pal)

                            ent_img = (ent_norm[b].detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
                            Image.fromarray(ent_img, mode="L").save(
                                os.path.join(vis_dir, f"ent_norm_s{cur_step}_idx{b}.png"))

                            im = images[b].detach().cpu()  # (C,H,W)
                            im = (im - im.min()) / (im.max() - im.min() + 1e-8)

                            if im.shape[0] == 1:
                                im_np = (im[0].numpy() * 255).astype(np.uint8)  # (H,W)
                                Image.fromarray(im_np, mode="L").save(
                                    os.path.join(vis_dir, f"image_s{cur_step}_idx{b}.png"))
                            else:
                                im_np = (im.permute(1, 2, 0).numpy() * 255).astype(np.uint8)  # (H,W,C)
                                Image.fromarray(im_np).save(os.path.join(vis_dir, f"image_s{cur_step}_idx{b}.png"))

            optim.zero_grad()
            with autocast():
                outputs, features = model(images, ret_intermediate=self.ret_intermediate)  # output：(24,2,512,512)

                probs_new = torch.softmax(outputs, dim=1)
                max_probs_new, pseudo_labels_new = probs_new.max(dim=1)

                for i in range(0, self.nb_current_classes):
                    mask_new = pseudo_labels_new == i
                    new_cls_aver[i] = max_probs_new[mask_new].sum()
                    new_cls_num[i] = mask_new.sum()
                for i in range(0, self.nb_current_classes):
                    if new_cls_num[i] == 0:
                        new_cls_aver[i] = 0
                    else:
                        new_cls_aver[i] = new_cls_aver[i] / new_cls_num[i]

                if self.pseudo_soft is not None:
                    loss = soft_crossentropy(
                        outputs,
                        labels,
                        outputs_old,
                        mask_valid_pseudo,
                        mask_background,
                        self.pseudo_soft,
                        pseudo_soft_factor=self.pseudo_soft_factor
                    )
                elif not self.icarl_only_dist:  # True
                    if self.ce_on_pseudo and self.step > 0 and self.model_old is not None:
                        assert self.pseudo_labeling is not None
                        assert self.pseudo_labeling == "entropy"

                        loss_not_pseudo = criterion(
                            outputs,
                            original_labels,
                            mask=mask_background & mask_valid_pseudo
                        )

                        _labels = original_labels.clone()
                        _labels[~(mask_background & mask_valid_pseudo)] = 255
                        _labels[mask_background & mask_valid_pseudo] = pseudo_labels[mask_background &
                                                                                     mask_valid_pseudo]
                        loss_pseudo = F.cross_entropy(
                            outputs, _labels, ignore_index=255, reduction="none"
                        )
                        loss = loss_pseudo + loss_not_pseudo
                    elif self.ce_on_new:
                        _labels = labels.clone()
                        _labels[_labels == 0] = 255
                        loss = criterion(outputs, _labels)
                    else:
                        if self.out_dis and self.step > 0:
                            w = self.efficient_step_old_class_weight(outputs, labels)
                            w1 = self.efficient_dis_old_class_weight(outputs, labels)

                            loss_dis = lossresult(w1 * new_cls_aver, w1 * old_cls_aver) * 0.5

                            loss = criterion(outputs, labels)
                            loss = w * loss
                        else:
                            loss = criterion(outputs, labels)

                else:
                    loss = self.licarl(outputs, labels, torch.sigmoid(outputs_old))


                tau_eff = None
                if (getattr(self, "llc_enable", True)
                        and self.step > 0
                        and self.model_old is not None
                        and self.pseudo_labeling == "entropy"):
                    pre_logits = features["pre_logits"]  # head的输出[B,256,16=H/16,16=W/16]
                    B_, C_, h32, w32 = pre_logits.shape

                    pseudo_32 = F.interpolate(
                        pseudo_labels.unsqueeze(1).float(), size=(h32, w32), mode="nearest"
                    ).squeeze(1).long()  # [B,16,16]


                    valid_32 = F.interpolate(
                        (mask_valid_pseudo & mask_background).unsqueeze(1).float(),
                        size=(h32, w32),
                        mode="nearest"
                    ).squeeze(1).bool()  # [B,32,32]

                    reg_feat_cpu, reg_lab_cpu, reg_refs = extract_regions_from_prelogits(
                        pre_logits_bchw=pre_logits,  # [B,256,32,32]
                        pseudo_hw=pseudo_32,  # [B,32,32]
                        valid_hw=valid_32,  # [B,32,32]
                        old_classes=int(self.old_classes),
                        min_region_px=getattr(self, "llc_min_region_px", 8)
                    )

                    if reg_feat_cpu is not None and reg_feat_cpu.numel() > 0:
                        bank_feat, bank_lab = self.region_bank.get_tensors(device=self.device)  # GPU
                        q_feat = reg_feat_cpu.to(self.device, dtype=torch.float32)
                        q_lab = reg_lab_cpu.to(self.device, dtype=torch.long)

                        llc = knn_llc_cosine(
                            query_feat=q_feat, query_label=q_lab,
                            bank_feat=bank_feat, bank_label=bank_lab,
                            k=getattr(self, "llc_k", 50)
                        )  # [Nregion] on GPU

                        pr_cpu, gmm_info = gmm_2comp_pr(
                            llc=llc.detach().cpu(),
                            gmm_tol=getattr(self, "gmm_tol", 1e-3),
                            gmm_reg_covar=getattr(self, "gmm_reg", 1e-6),
                            gmm_max_iter=getattr(self, "gmm_iter", 10),
                        )

                        pr_mean = float(pr_cpu.mean().item())
                        tau_base = float(getattr(self.opts, "dro_tau", 0.5))
                        tau_alpha = float(getattr(self, "llc_tau_alpha", 0.5))

                        if pr_mean < 0.3:
                            tau_eff = 0.3
                        elif pr_mean > 0.6:
                            tau_eff = 0.6
                        else:
                            tau_eff = tau_base * ((1.0 - tau_alpha) + tau_alpha * pr_mean)

                        tau_min = float(getattr(self.opts, "dro_tau_min", 0.05))
                        tau_max = float(getattr(self.opts, "dro_tau_max", tau_base))
                        tau_eff = float(max(tau_min, min(tau_eff, tau_max)))


                        bank_pr_th = float(getattr(self, "llc_bank_pr_th", 0.7))
                        keep_idx = torch.where(pr_cpu >= bank_pr_th)[0]
                        if keep_idx.numel() > 0:
                            self.region_bank.add(reg_feat_cpu[keep_idx], reg_lab_cpu[keep_idx])

                        if (self.rank == 0) and (cur_step == 0):
                            try:
                                print(f"[LLC->tau] regions={int(pr_cpu.numel())} bank={len(self.region_bank)} "
                                      f"pr_mean={pr_mean:.3f} tau_base={tau_base:.3f} tau_eff={tau_eff:.3f} "
                                      f"gmm_means={gmm_info.get('means', None)}")
                            except Exception:
                                pass

                if getattr(self.opts, "dro_enable", False):
                    warm = int(getattr(self.opts, "dro_warmup_epochs", 2))
                    if cur_epoch >= warm:
                        tau_base = float(getattr(self.opts, "dro_tau", 0.5))
                        tau_use = float(tau_eff) if (tau_eff is not None) else tau_base

                        level = str(getattr(self.opts, "dro_level", "pixel")).lower()

                        loss_map = loss
                        if loss_map.dim() != 3:
                            if loss_map.dim() == 4:
                                loss_map = loss_map.mean(dim=1)
                            else:
                                raise ValueError(f"DRO expects loss dim 3 or 4, got {loss_map.shape}")

                        valid = (labels != 255)
                        dro_mask = valid

                        if level == "pixel":
                            if dro_mask.any():
                                vals = loss_map[dro_mask]
                                k = max(1, int(tau_use * vals.numel()))
                                loss = torch.topk(vals, k=k, largest=True).values.mean()
                            else:
                                loss = loss_map.mean()

                        elif level == "sample":
                            B__ = loss_map.shape[0]
                            per_sample = []
                            for b in range(B__):
                                vb = valid[b]
                                per_sample.append(loss_map[b][vb].mean() if vb.any() else loss_map[b].mean())
                            per_sample = torch.stack(per_sample, dim=0)  # [B]
                            k = max(1, int(tau_use * per_sample.numel()))
                            loss = torch.topk(per_sample, k=k, largest=True).values.mean()  # scalar

                        else:
                            raise ValueError(f"Unknown dro_level={level}, expected 'pixel' or 'sample'")


                if self.out_dis and self.step > 0 and outputs_old is not None:
                    old_classes = int(self.old_classes)
                    p_old = torch.softmax(outputs_old[:, :old_classes], dim=1)

                    prior_mat = getattr(self.opts, "text_prior_mat_cont", None)
                    if prior_mat is not None:
                        prior_mat = prior_mat[:old_classes, :old_classes].contiguous()  # [C_old, C_old]



                    if prior_mat is not None and self.rank == 0 and cur_step == 0:
                        pm = prior_mat

                        finite = torch.isfinite(pm)


                        rs = pm.sum(dim=1)
                        n_nan = torch.isnan(pm).sum().item()
                        n_inf = torch.isinf(pm).sum().item()

                    loss_soft_label = self.criterion_soft_label(
                        outputs,
                        p_old,
                        old_classes,
                        labels,
                        prior_mat=prior_mat,
                        lam=getattr(self.opts, "txt_fuse_lam", 0.1),
                        gate_beta=getattr(self.opts, "txt_gate_beta", 0.0),
                        lam_min=getattr(self.opts, "txt_lam_min", 0.0),
                        lam_max=getattr(self.opts, "txt_lam_max", 0.5),
                    )
                    Regularizer_soft = self.reg_pesudo_label(outputs, labels, self.old_classes)  #

                if self.sample_weights_new is not None:
                    sample_weights = torch.ones_like(original_labels).to(device, dtype=torch.float32)
                    sample_weights[original_labels >= 0] = self.sample_weights_new

                if sample_weights is not None and loss.dim() > 0:
                    loss = loss * sample_weights

                if torch.is_tensor(classif_adaptive_factor):
                    if loss.dim() > 0:
                        loss = classif_adaptive_factor * loss
                    else:
                        loss = loss * classif_adaptive_factor.mean()
                if loss.dim() > 0:
                    loss = loss.mean()  # scalar

                loss_soft_label = loss_soft_label.mean()
                Regularizer_soft = Regularizer_soft.mean()

                if self.icarl_combined and outputs_old is not None:
                    # tensor.narrow( dim, start, end) -> slice tensor from start to end in the specified dim
                    n_cl_old = outputs_old.shape[1]
                    # use n_cl_old to sum the contribution of each class, and not to average them (as done in our BCE).
                    l_icarl = self.icarl * n_cl_old * self.licarl(
                        outputs.narrow(1, 0, n_cl_old), torch.sigmoid(outputs_old)
                    )

                if self.lde_flag:
                    lde = self.lde * self.lde_loss(features['body'], features_old['body'])

                if self.lkd_flag:
                    if self.lkd_mask is not None and self.lkd_mask == "oldbackground":
                        kd_mask = labels < self.old_classes
                    elif self.lkd_mask is not None and self.lkd_mask == "new":
                        kd_mask = labels >= self.old_classes
                    else:
                        kd_mask = None

                    if self.temperature_apply is not None:
                        temp_mask = torch.ones_like(labels).to(outputs.device).to(outputs.dtype)

                        if self.temperature_apply == "all":
                            temp_mask = temp_mask / self.temperature
                        elif self.temperature_apply == "old":
                            mask_bg = labels < self.old_classes
                            temp_mask[mask_bg] = temp_mask[mask_bg] / self.temperature
                        elif self.temperature_apply == "new":
                            mask_fg = labels >= self.old_classes
                            temp_mask[mask_fg] = temp_mask[mask_fg] / self.temperature
                        temp_mask = temp_mask[:, None]
                    else:
                        temp_mask = 1.0

                    if self.kd_need_labels and outputs_old is not None:
                        lkd = self.lkd * self.lkd_loss(
                            outputs * temp_mask, outputs_old * temp_mask, labels, mask=kd_mask
                        )
                    else:
                        lkd = self.lkd * self.lkd_loss(
                            outputs * temp_mask, outputs_old * temp_mask, mask=kd_mask
                        )

                    if self.kd_new:
                        mask_bg = labels == 0
                        lkd = lkd[mask_bg]

                    if kd_mask is not None and self.kd_mask_adaptative_factor:
                        lkd = lkd.mean(dim=(1, 2)) * kd_mask.float().mean(dim=(1, 2))
                    lkd = torch.mean(lkd)

                if self.pod is not None and self.step > 0 and self.model_old is not None:
                    attentions_old = features_old["attentions"]
                    attentions_new = features["attentions"]

                    if self.pod_logits:
                        attentions_old.append(features_old["sem_logits_small"])
                        attentions_new.append(features["sem_logits_small"])
                    elif self.pod_large_logits:
                        attentions_old.append(outputs_old)
                        attentions_new.append(outputs)

                    pod_loss = features_distillation(
                        attentions_old,
                        attentions_new,
                        collapse_channels=self.pod,
                        labels=labels,
                        index_new_class=self.old_classes,
                        pod_apply=self.pod_apply,
                        pod_deeplab_mask=self.pod_deeplab_mask,
                        pod_deeplab_mask_factor=self.pod_deeplab_mask_factor,
                        interpolate_last=self.pod_interpolate_last,
                        pod_factor=self.pod_factor,
                        prepro=self.pod_prepro,
                        deeplabmask_upscale=not self.deeplab_mask_downscale,
                        spp_scales=self.spp_scales,
                        pod_options=self.pod_options,
                        outputs_old=outputs_old,
                        use_pod_schedule=self.use_pod_schedule,
                        nb_current_classes=self.nb_current_classes,
                        nb_new_classes=self.nb_new_classes
                    )

                if self.entropy_min > 0. and self.step > 0 and self.model_old is not None:
                    mask_new = labels > 0
                    entropies = entropy(torch.softmax(outputs, dim=1))
                    entropies[mask_new] = 0.
                    pixel_amount = (~mask_new).float().sum(dim=(1, 2))
                    loss_entmin = (entropies.sum(dim=(1, 2)) / pixel_amount).mean()

                if self.kd_scheduling:
                    lkd = lkd * math.sqrt(self.nb_current_classes / self.nb_new_classes)


                loss_tot = loss + pod_loss + loss_dis
                if (self.rank == 0) and (cur_step == 0):
                    def _fin(x):
                        return torch.isfinite(x).all().item()

                    def _stat(x):
                        x = x.detach()
                        return float(x), _fin(x)


            self.scaler.scale(loss_tot).backward()
            if (self.rank == 0) and (cur_step == 0):
                m = model.module if hasattr(model, "module") else model

                valid = (labels != 255)
                new_pix = (valid & (labels >= self.old_classes)).sum().item()
                valid_pix = valid.sum().item()
                print(f"[DBG] valid_pix={valid_pix}, new_pix={new_pix}, new_ratio={new_pix / (valid_pix + 1e-12):.6f}")
                print("[DBG] labels unique:", torch.unique(labels))

                printed = 0

                opt_ids = {id(p) for group in optim.param_groups for p in group["params"]}
                cls_cnt = 0
                miss = []
                for name, p in m.named_parameters():
                    if name.startswith("cls."):
                        cls_cnt += 1
                        if p.requires_grad and id(p) not in opt_ids:
                            miss.append(name)
                print(f"[DBG][OPT] cls_total={cls_cnt}, not_in_opt={len(miss)}; examples={miss[:10]}")
            if hasattr(self.opts, "pending_step") and self.opts.pending_step == self.step:
                if getattr(self.opts, "relevance_done_step", -1) != self.step:
                    g_sig = _extract_oldhead_grad_signature(model, step=self.step)  # CPU

                    R_orig = compute_task_relevance(
                        step=self.step,
                        task_classes_orig=self.opts.pending_task_classes_orig,  # 这个好像没定义
                        q_txt=self.opts.pending_q_txt,
                        g_sig=g_sig,
                        memory=self.opts.task_memory,
                        eta=getattr(self.opts, "rel_eta", 0.7),
                        tau=getattr(self.opts, "rel_tau", 0.5),
                        topk=getattr(self.opts, "rel_topk", 3),
                    )

                    inc_ds = train_loader.dataset
                    inv = inc_ds.inverted_order
                    num_cont = len(inc_ds.labels_cum)
                    R_cont = torch.zeros(num_cont, dtype=torch.float32)

                    if R_orig is not None:
                        for oc, w in R_orig.items():
                            if oc in inv:
                                R_cont[inv[oc]] = float(w)

                    self.opts.class_relevance_cont = R_cont
                    self.opts.relevance_done_step = self.step

                    if self.rank == 0:
                        print(f"[TaskRel] step={self.step} done. R_cont nonzero={int((R_cont > 0).sum().item())}")


            if self.regularizer_flag:
                if distributed.get_rank() == 0:
                    self.regularizer.update()
                l_reg = self.reg_importance * self.regularizer.penalty()
                if l_reg != 0.:
                    self.scaler.scale(l_reg).backward()

            self.scaler.step(optim)
            self.scaler.update()
            if scheduler is not None:
                scheduler.step()


            epoch_loss += loss.item()
            reg_loss += l_reg.item() if l_reg != 0. else 0.
            reg_loss += lkd.item() + lde.item() + l_icarl.item()
            interval_loss += loss.item() + lkd.item() + lde.item() + l_icarl.item() + pod_loss.item(
            ) + loss_entmin.item()
            interval_loss += l_reg.item() if l_reg != 0. else 0.

            if (cur_step + 1) % print_int == 0:

                interval_loss = interval_loss / print_int

                if self.rank == 0:
                    lr = optim.param_groups[0]["lr"]
                    pbar.set_postfix({
                        "lr": f"{lr:.2e}",
                        "loss": f"{interval_loss:.4f}",
                        "CE": f"{loss.item():.4f}",
                        "LKD": f"{lkd.item():.4f}",
                        "LDE": f"{lde.item():.4f}",
                        "POD": f"{pod_loss.item():.4f}",
                    })
                # visualization

                interval_loss = 0.0

        epoch_loss = torch.tensor(epoch_loss).to(self.device)
        reg_loss = torch.tensor(reg_loss).to(self.device)

        if torch.distributed.is_initialized():
            torch.distributed.reduce(epoch_loss, dst=0)
            torch.distributed.reduce(reg_loss, dst=0)
        else:
            print("[Warning] Distributed not initialized, skip reduce operation")


        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            epoch_loss = epoch_loss / distributed.get_world_size() / len(train_loader)
            reg_loss = reg_loss / distributed.get_world_size() / len(train_loader)
        else:
            epoch_loss = epoch_loss / len(train_loader)
            reg_loss = reg_loss / len(train_loader)

        if self.rank == 0:
            print(f"Epoch {cur_epoch + 1}, Class Loss={epoch_loss}, Reg Loss={reg_loss}")


        return (epoch_loss, reg_loss)

    def find_median(self, train_loader, target_portion, device, mode="probability"):

        if mode == "entropy":
            max_value = torch.log(torch.tensor(self.nb_current_classes).float().to(device))
            # log(C)可能的最大熵值
            nb_bins = 100
        else:
            max_value = 1.0
            nb_bins = 20
        if self.pseudo_nb_bins is not None:
            nb_bins = self.pseudo_nb_bins

        histograms = torch.zeros(self.nb_current_classes, nb_bins).long().to(self.device)

        for cur_step, (images, labels,_) in enumerate(train_loader):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs_old, features_old = self.model_old(images, ret_intermediate=True)

            mask_bg = labels == 0
            probas = torch.softmax(outputs_old, dim=1)

            max_probas, pseudo_labels = probas.max(dim=1)

            if mode == "entropy":
                values_to_bins = entropy(probas)[mask_bg].view(-1) / max_value  # 计算归一化熵值，0-1
            else:
                values_to_bins = max_probas[mask_bg].view(-1)

            x_coords = pseudo_labels[mask_bg].view(-1)  # 在真实背景区域中，旧模型预测的类别ID
            y_coords = torch.clamp((values_to_bins * nb_bins).long(), max=nb_bins - 1)

            histograms.index_put_(
                (x_coords, y_coords),
                torch.LongTensor([1]).expand_as(x_coords).to(histograms.device),
                accumulate=True
            )
        thresholds = torch.zeros(self.nb_current_classes, dtype=torch.float32).to(
            self.device
        )

        for c in range(self.nb_current_classes):
            total = histograms[c].sum()
            if total <= 0.:
                continue
            if self.out_dis:
                half = total * target_portion
            else:
                half = total / 2

            running_sum = 0.
            for lower_border in range(nb_bins):
                lower_border = lower_border / nb_bins
                bin_index = int(lower_border * nb_bins)
                if half >= running_sum and half <= (running_sum + histograms[c, bin_index]):
                    break
                running_sum += lower_border * nb_bins

            median = lower_border + ((half - running_sum) /
                                     histograms[c, bin_index].sum()) * (1 / nb_bins)

            thresholds[c] = median

        base_threshold = self.threshold
        if "_" in mode:
            mode, base_threshold = mode.split("_")
            base_threshold = float(base_threshold)
        if self.step_threshold is not None:
            self.threshold += self.step * self.step_threshold

        if mode == "entropy":
            for c in range(len(thresholds)):
                thresholds[c] = max(thresholds[c], base_threshold)
        else:
            for c in range(len(thresholds)):
                thresholds[c] = min(thresholds[c], base_threshold)
        return thresholds.to(device), max_value

    def validate(self, loader, metrics, ret_samples_ids=None, end_task=False):
        metrics.reset()
        model = self.model
        device = self.device
        criterion = self.criterion
        model.eval()

        if hasattr(model, 'module'):
            model.module.in_eval = True
        else:
            model.in_eval = True
        if self.model_old is not None:
            self.model_old.in_eval = True

        class_loss = 0.0
        reg_loss = 0.0
        lkd = torch.tensor(0.)
        lde = torch.tensor(0.)
        l_icarl = torch.tensor(0.)
        l_reg = torch.tensor(0.)

        if self.step > 0 and self.model_old is not None and self.align_weight_frequency == "epoch":
            model.module.align_weight(self.align_weight)
        elif self.step > 0 and self.model_old is not None and self.align_weight_frequency == "task" and end_task:
            model.module.align_weight(self.align_weight)

        ret_samples = []
        all_pred_classes = []
        all_label_classes = []
        with torch.no_grad():
            for i, (images, labels,_) in enumerate(loader):

                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)
                unique_labels = torch.unique(labels)
                num_valid_pixels = ((labels != 0) & (labels != 255)).sum().item()
                num0_valid_pixels = ((labels == 0) & (labels != 255)).sum().item()
                num255_valid_pixels = ((labels != 0) & (labels == 255)).sum().item()
                all_label_classes.append(unique_labels)

                if (
                        self.lde_flag or self.lkd_flag or self.icarl_dist_flag
                ) and self.model_old is not None:
                    with torch.no_grad():
                        outputs_old, features_old = self.model_old(images, ret_intermediate=True)

                outputs, features = model(images, ret_intermediate=True)

                if not self.icarl_only_dist:
                    loss = criterion(outputs, labels)

                else:
                    loss = self.licarl(outputs, labels, torch.sigmoid(outputs_old))

                loss = loss.mean()  # scalar

                if self.icarl_combined:
                    n_cl_old = outputs_old.shape[1]
                    l_icarl = self.icarl * n_cl_old * self.licarl(
                        outputs.narrow(1, 0, n_cl_old), torch.sigmoid(outputs_old)
                    )

                if self.lde_flag:
                    lde = self.lde_loss(features['body'], features_old['body'])

                if self.lkd_flag and not self.kd_need_labels:
                    lkd = self.lkd_loss(outputs, outputs_old).mean()

                if self.regularizer_flag:
                    l_reg = self.regularizer.penalty()

                class_loss += loss.item()
                reg_loss += l_reg.item() if l_reg != 0. else 0.
                reg_loss += lkd.item() + lde.item() + l_icarl.item()

                _, prediction = outputs.max(dim=1)

                labels = labels.cpu().numpy()
                prediction = prediction.cpu().numpy()
                pmin = int(prediction.min().item())
                pmax = int(prediction.max().item())
                if pmin < 0 or pmax >= outputs.shape[1]:
                    raise RuntimeError(f"[BAD PRED BEFORE METRICS] pred range {pmin}..{pmax}, C={outputs.shape[1]}")
                metrics.update(labels, prediction)

                if ret_samples_ids is not None and i in ret_samples_ids:
                    ret_samples.append((images[0].detach().cpu().numpy(), labels[0], prediction[0]))

            metrics.synch(device)
            score = metrics.get_results()

            class_loss = torch.tensor(class_loss).to(self.device)
            reg_loss = torch.tensor(reg_loss).to(self.device)

            is_distributed = torch.distributed.is_initialized()
            if is_distributed:
                torch.distributed.reduce(class_loss, dst=0)
                torch.distributed.reduce(reg_loss, dst=0)

                if distributed.get_rank() == 0:
                    class_loss = class_loss / distributed.get_world_size() / len(loader)
                    reg_loss = reg_loss / distributed.get_world_size() / len(loader)

                if self.rank == 0:
                    print(
                        f"Validation, Class Loss={class_loss}, Reg Loss={reg_loss} (without scaling)"
                    )
            else:
                class_loss = class_loss / len(loader)
                reg_loss = reg_loss / len(loader)
                print(
                    f"Validation, Class Loss={class_loss}, Reg Loss={reg_loss} (without scaling)"
                )

        return (class_loss, reg_loss), score, ret_samples

    def state_dict(self):
        state = {"regularizer": self.regularizer.state_dict() if self.regularizer_flag else None}

        return state

    def load_state_dict(self, state):
        if state["regularizer"] is not None and self.regularizer is not None:
            self.regularizer.load_state_dict(state["regularizer"])


def entropy(probabilities):

    factor = 1 / math.log(probabilities.shape[1] + 1e-8)
    return -factor * torch.mean(probabilities * torch.log(probabilities + 1e-8), dim=1)


def features_distillation(
        list_attentions_a,
        list_attentions_b,
        collapse_channels="spatial",
        normalize=True,
        labels=None,
        index_new_class=None,
        pod_apply="all",
        pod_deeplab_mask=False,
        pod_deeplab_mask_factor=None,
        interpolate_last=False,
        pod_factor=1.,
        prepro="pow",
        deeplabmask_upscale=True,
        spp_scales=[1, 2, 4],
        pod_options=None,
        outputs_old=None,
        use_pod_schedule=True,
        nb_current_classes=-1,
        nb_new_classes=-1
):

    device = list_attentions_a[0].device

    assert len(list_attentions_a) == len(list_attentions_b)

    if pod_deeplab_mask_factor is None:
        pod_deeplab_mask_factor = pod_factor

    normalize = False

    apply_mask = "background"
    upscale_mask_topk = 1
    mask_position = "last"
    use_adaptative_factor = False
    mix_new_old = None

    loss = torch.tensor(0.).to(list_attentions_a[0].device)
    for i, (a, b) in enumerate(zip(list_attentions_a, list_attentions_b)):
        adaptative_pod_factor = 1.0
        difference_function = "frobenius"
        pool = True
        use_adaptative_factor = False
        handle_extra_channels = "sum"
        normalize_per_scale = False

        if pod_options and pod_options.get("switch"):
            if i < len(list_attentions_a) - 1:
                if "before" in pod_options["switch"]:
                    collapse_channels = pod_options["switch"]["before"].get(
                        "type", collapse_channels
                    )
                    pod_factor = pod_options["switch"]["before"].get("factor", pod_factor)
                    normalize = pod_options["switch"]["before"].get("norm", False)
                    prepro = pod_options["switch"]["before"].get("prepro", prepro)
                    use_adaptative_factor = pod_options["switch"]["before"].get(
                        "use_adaptative_factor", use_adaptative_factor
                    )
            else:
                if "after" in pod_options["switch"]:
                    collapse_channels = pod_options["switch"]["after"].get(
                        "type", collapse_channels
                    )
                    pod_factor = pod_options["switch"]["after"].get("factor", pod_factor)
                    normalize = pod_options["switch"]["after"].get("norm", False)
                    prepro = pod_options["switch"]["after"].get("prepro", prepro)

                    apply_mask = pod_options["switch"]["after"].get("apply_mask", apply_mask)
                    upscale_mask_topk = pod_options["switch"]["after"].get(
                        "upscale_mask_topk", upscale_mask_topk
                    )
                    use_adaptative_factor = pod_options["switch"]["after"].get(
                        "use_adaptative_factor", use_adaptative_factor
                    )
                    mix_new_old = pod_options["switch"]["after"].get("mix_new_old", mix_new_old)

                    handle_extra_channels = pod_options["switch"]["after"].get(
                        "extra_channels", handle_extra_channels
                    )
                    spp_scales = pod_options["switch"]["after"].get(
                        "spp_scales", spp_scales
                    )
                    use_pod_schedule = pod_options["switch"]["after"].get(
                        "use_pod_schedule", use_pod_schedule
                    )

            mask_position = pod_options["switch"].get("mask_position", mask_position)
            normalize_per_scale = pod_options["switch"].get(
                "normalize_per_scale", normalize_per_scale
            )
            pool = pod_options.get("pool", pool)

        if a.shape[1] != b.shape[1]:
            assert i == len(list_attentions_a) - 1
            assert a.shape[0] == b.shape[0]
            assert a.shape[2] == b.shape[2]
            assert a.shape[3] == b.shape[3]

            assert handle_extra_channels in ("trim", "sum"), handle_extra_channels

            if handle_extra_channels == "sum":
                _b = torch.zeros_like(a).to(a.dtype).to(a.device)
                _b[:, 0] = b[:, 0] + b[:, index_new_class:].sum(dim=1)
                _b[:, 1:] = b[:, 1:index_new_class]
                b = _b
            elif handle_extra_channels == "trim":
                b = b[:, :index_new_class]

        assert a.shape == b.shape, (a.shape, b.shape)

        if not pod_deeplab_mask and use_adaptative_factor:
            adaptative_pod_factor = (labels == 0).float().mean()

        if prepro == "pow":
            a = torch.pow(a, 2)
            b = torch.pow(b, 2)
        elif prepro == "none":
            pass
        elif prepro == "abs":
            a = torch.abs(a, 2)
            b = torch.abs(b, 2)
        elif prepro == "relu":
            a = torch.clamp(a, min=0.)
            b = torch.clamp(b, min=0.)

        if collapse_channels == "spatial":
            a_h = a.sum(dim=3).view(a.shape[0], -1)
            b_h = b.sum(dim=3).view(b.shape[0], -1)
            a_w = a.sum(dim=2).view(a.shape[0], -1)
            b_w = b.sum(dim=2).view(b.shape[0], -1)
            a = torch.cat([a_h, a_w], dim=-1)
            b = torch.cat([b_h, b_w], dim=-1)
        elif collapse_channels == "global":
            a = _global_pod(a, spp_scales, normalize=False)
            b = _global_pod(b, spp_scales, normalize=False)
        elif collapse_channels == "local":
            if pod_deeplab_mask and (
                    (i == len(list_attentions_a) - 1 and mask_position == "last") or
                    mask_position == "all"
            ):
                if pod_deeplab_mask_factor == 0.:
                    continue

                pod_factor = pod_deeplab_mask_factor

                if apply_mask == "background":
                    mask = labels < index_new_class
                elif apply_mask == "old":
                    pseudo_labels = labels.clone()
                    mask_background = labels == 0
                    pseudo_labels[mask_background] = outputs_old.argmax(dim=1)[mask_background]

                    mask = (labels < index_new_class) & (0 < pseudo_labels)
                else:
                    raise NotImplementedError(f"Unknown apply_mask={apply_mask}.")

                if deeplabmask_upscale:
                    a = F.interpolate(
                        torch.topk(a, k=upscale_mask_topk, dim=1)[0],
                        size=labels.shape[-2:],
                        mode="bilinear",
                        align_corners=False
                    )
                    b = F.interpolate(
                        torch.topk(b, k=upscale_mask_topk, dim=1)[0],
                        size=labels.shape[-2:],
                        mode="bilinear",
                        align_corners=False
                    )
                else:
                    mask = F.interpolate(mask[:, None].float(), size=a.shape[-2:]).bool()[:, 0]

                if use_adaptative_factor:
                    adaptative_pod_factor = mask.float().mean(dim=(1, 2))

                a = _local_pod_masked(
                    a, mask, spp_scales, normalize=False, normalize_per_scale=normalize_per_scale
                )
                b = _local_pod_masked(
                    b, mask, spp_scales, normalize=False, normalize_per_scale=normalize_per_scale
                )
            else:
                a = _local_pod(
                    a, spp_scales, normalize=False, normalize_per_scale=normalize_per_scale
                )
                b = _local_pod(
                    b, spp_scales, normalize=False, normalize_per_scale=normalize_per_scale
                )
        else:
            raise ValueError("Unknown method to collapse: {}".format(collapse_channels))

        if i == len(list_attentions_a) - 1 and pod_options is not None:
            if "difference_function" in pod_options:
                difference_function = pod_options["difference_function"]
        elif pod_options is not None:
            if "difference_function_all" in pod_options:
                difference_function = pod_options["difference_function_all"]

        if normalize:
            a = F.normalize(a, dim=1, p=2)
            b = F.normalize(b, dim=1, p=2)

        if difference_function == "frobenius":
            if isinstance(a, list):
                layer_loss = torch.tensor(
                    [torch.frobenius_norm(aa - bb, dim=-1) for aa, bb in zip(a, b)]
                ).to(device)
            else:
                layer_loss = torch.frobenius_norm(a - b, dim=-1)
        elif difference_function == "frobenius_mix":
            layer_loss_old = torch.frobenius_norm(a[0] - b[0], dim=-1)
            layer_loss_new = torch.frobenius_norm(a[1] - b[1], dim=-1)

            layer_loss = mix_new_old * layer_loss_old + (1 - mix_new_old) * layer_loss_new
        elif difference_function == "l1":
            if isinstance(a, list):
                layer_loss = torch.tensor(
                    [torch.norm(aa - bb, p=1, dim=-1) for aa, bb in zip(a, b)]
                ).to(device)
            else:
                layer_loss = torch.norm(a - b, p=1, dim=-1)
        elif difference_function == "kl":
            d1, d2, d3 = a.shape
            a = (a.view(d1 * d2, d3) + 1e-8).log()
            b = b.view(d1 * d2, d3) + 1e-8

            layer_loss = F.kl_div(a, b, reduction="none").view(d1, d2, d3).sum(dim=(1, 2))
        elif difference_function == "bce":
            d1, d2, d3 = a.shape
            layer_loss = bce(a.view(d1 * d2, d3), b.view(d1 * d2, d3)).view(d1, d2,
                                                                            d3).mean(dim=(1, 2))
        else:
            raise NotImplementedError(f"Unknown difference_function={difference_function}")

        assert torch.isfinite(layer_loss).all(), layer_loss
        assert (layer_loss >= 0.).all(), layer_loss

        layer_loss = torch.mean(adaptative_pod_factor * layer_loss)
        if pod_factor <= 0.:
            continue

        layer_loss = pod_factor * layer_loss
        if use_pod_schedule:
            layer_loss = layer_loss * math.sqrt(nb_current_classes / nb_new_classes)
        loss += layer_loss

    return loss / len(list_attentions_a)


def bce(x, y):
    return -(y * torch.log(x + 1e-6) + (1 - y) * torch.log((1 - x) + 1e-6))


def _local_pod(x, spp_scales=[1, 2, 4], normalize=False, normalize_per_scale=False):
    b = x.shape[0]
    w = x.shape[-1]
    emb = []

    for scale_index, scale in enumerate(spp_scales):
        k = w // scale

        nb_regions = scale ** 2

        for i in range(scale):
            for j in range(scale):
                tensor = x[..., i * k:(i + 1) * k, j * k:(j + 1) * k]

                horizontal_pool = tensor.mean(dim=3).view(b, -1)
                vertical_pool = tensor.mean(dim=2).view(b, -1)

                if normalize_per_scale is True:
                    horizontal_pool = horizontal_pool / nb_regions
                    vertical_pool = vertical_pool / nb_regions
                elif normalize_per_scale == "spm":
                    if scale_index == 0:
                        factor = 2 ** (len(spp_scales) - 1)
                    else:
                        factor = 2 ** (len(spp_scales) - scale_index)
                    horizontal_pool = horizontal_pool / factor
                    vertical_pool = vertical_pool / factor

                if normalize:
                    horizontal_pool = F.normalize(horizontal_pool, dim=1, p=2)
                    vertical_pool = F.normalize(vertical_pool, dim=1, p=2)

                emb.append(horizontal_pool)
                emb.append(vertical_pool)

    return torch.cat(emb, dim=1)


def _local_pod_masked(
        x, mask, spp_scales=[1, 2, 4], normalize=False, normalize_per_scale=False
):
    b = x.shape[0]
    c = x.shape[1]
    w = x.shape[-1]
    emb = []

    mask = mask[:, None].repeat(1, c, 1, 1)
    x[mask] = 0.

    for scale_index, scale in enumerate(spp_scales):
        k = w // scale

        nb_regions = scale ** 2

        for i in range(scale):
            for j in range(scale):
                tensor = x[..., i * k:(i + 1) * k, j * k:(j + 1) * k]

                horizontal_pool = tensor.mean(dim=3).view(b, -1)
                vertical_pool = tensor.mean(dim=2).view(b, -1)

                if normalize_per_scale is True:
                    horizontal_pool = horizontal_pool / nb_regions
                    vertical_pool = vertical_pool / nb_regions
                elif normalize_per_scale == "spm":
                    if scale_index == 0:
                        factor = 2 ** (len(spp_scales) - 1)
                    else:
                        factor = 2 ** (len(spp_scales) - scale_index)
                if normalize:
                    horizontal_pool = F.normalize(horizontal_pool, dim=1, p=2)
                    vertical_pool = F.normalize(vertical_pool, dim=1, p=2)

                emb.append(horizontal_pool)
                emb.append(vertical_pool)

    return torch.cat(emb, dim=1)


def _global_pod(x, spp_scales=[2, 4, 8], normalize=False):
    b = x.shape[0]
    w = x.shape[-1]

    emb = []
    for scale in spp_scales:
        tensor = F.avg_pool2d(x, kernel_size=w // scale)
        horizontal_pool = tensor.sum(dim=2).view(b, -1)
        vertical_pool = tensor.sum(dim=3).view(b, -1)

        if normalize:
            horizontal_pool = F.normalize(horizontal_pool, dim=1, p=2)
            vertical_pool = F.normalize(vertical_pool, dim=1, p=2)

        tensor_pool = torch.cat([horizontal_pool, vertical_pool], dim=-1)

        emb.append(tensor_pool)

    return torch.cat(emb, dim=1)

def _connected_components_2d(bin_mask: np.ndarray):
    """
    bin_mask: (H,W) bool
    return: list of components, each is list of (y,x)
    4-neighborhood
    """
    H, W = bin_mask.shape
    visited = np.zeros_like(bin_mask, dtype=np.uint8)
    comps = []
    for y in range(H):
        for x in range(W):
            if bin_mask[y, x] and visited[y, x] == 0:
                q = [(y, x)]
                visited[y, x] = 1
                pts = []
                while q:
                    cy, cx = q.pop()
                    pts.append((cy, cx))
                    if cy > 0 and bin_mask[cy - 1, cx] and visited[cy - 1, cx] == 0:
                        visited[cy - 1, cx] = 1
                        q.append((cy - 1, cx))
                    if cy + 1 < H and bin_mask[cy + 1, cx] and visited[cy + 1, cx] == 0:
                        visited[cy + 1, cx] = 1
                        q.append((cy + 1, cx))
                    if cx > 0 and bin_mask[cy, cx - 1] and visited[cy, cx - 1] == 0:
                        visited[cy, cx - 1] = 1
                        q.append((cy, cx - 1))
                    if cx + 1 < W and bin_mask[cy, cx + 1] and visited[cy, cx + 1] == 0:
                        visited[cy, cx + 1] = 1
                        q.append((cy, cx + 1))
                comps.append(pts)
    return comps

class RegionMemoryBank:

    def __init__(self, max_regions=5000, feat_dim=256, device="cpu"):
        self.max_regions = int(max_regions)
        self.feat_dim = int(feat_dim)
        self.device = device
        self._feats = deque()
        self._labels = deque()

    def __len__(self):
        return len(self._labels)

    @torch.no_grad()
    def add(self, feats: torch.Tensor, labels: torch.Tensor):

        if feats.numel() == 0:
            return
        feats = feats.detach().to("cpu", dtype=torch.float32)
        labels = labels.detach().to("cpu", dtype=torch.long)

        N = feats.shape[0]
        for i in range(N):
            self._feats.append(feats[i].clone())
            self._labels.append(int(labels[i].item()))
        while len(self._labels) > self.max_regions:
            self._feats.popleft()
            self._labels.popleft()

    @torch.no_grad()
    def get_tensors(self, device=None):
        if device is None:
            device = self.device
        if len(self._labels) == 0:
            feats = torch.empty(0, self.feat_dim, dtype=torch.float32, device=device)
            labs = torch.empty(0, dtype=torch.long, device=device)
            return feats, labs
        feats = torch.stack(list(self._feats), dim=0).to(device, dtype=torch.float32)
        labs = torch.tensor(list(self._labels), dtype=torch.long, device=device)
        return feats, labs

@torch.no_grad()
def extract_regions_from_prelogits(pre_logits_bchw: torch.Tensor,
                                  pseudo_hw: torch.Tensor,
                                  valid_hw: torch.Tensor,
                                  old_classes: int,
                                  min_region_px: int = 0):

    B, C, h, w = pre_logits_bchw.shape
    region_feats = []
    region_labels = []
    region_refs = []
    pre = pre_logits_bchw.detach().cpu()
    pseudo = pseudo_hw.detach().cpu().numpy().astype(np.int32)
    valid = valid_hw.detach().cpu().numpy().astype(np.bool_)

    for b in range(B):
        for c in range(1, int(old_classes)):
            bin_mask = (pseudo[b] == c) & valid[b]
            if bin_mask.sum() < min_region_px:
                continue
            comps = _connected_components_2d(bin_mask)
            for pts in comps:
                if len(pts) < min_region_px:
                    continue
                yy = [p[0] for p in pts]
                xx = [p[1] for p in pts]
                feat_pix = pre[b, :, yy, xx]  # [C, n]
                feat = feat_pix.mean(dim=1)   # [C]
                region_feats.append(feat)
                region_labels.append(c)
                region_refs.append((b, pts))

    if len(region_feats) == 0:
        return (
            torch.empty(0, C, dtype=torch.float32),
            torch.empty(0, dtype=torch.long),
            []
        )

    region_feats = torch.stack(region_feats, dim=0)
    region_labels = torch.tensor(region_labels, dtype=torch.long)  # [N]
    return region_feats, region_labels, region_refs

@torch.no_grad()
def knn_llc_cosine(query_feat: torch.Tensor,
                   query_label: torch.Tensor,
                   bank_feat: torch.Tensor,
                   bank_label: torch.Tensor,
                   k: int = 50):

    if query_feat.numel() == 0:
        return torch.empty(0, dtype=torch.float32, device=query_feat.device)

    # cosine
    q = F.normalize(query_feat, dim=1)
    b = F.normalize(bank_feat, dim=1) if bank_feat.numel() > 0 else bank_feat

    if bank_feat.shape[0] == 0:
        return torch.full((query_feat.shape[0],), 0.5, dtype=torch.float32, device=query_feat.device)

    sim = q @ b.t()  # [N,M]
    k = min(int(k), sim.shape[1])
    _, idx = torch.topk(sim, k=k, dim=1, largest=True)
    nn_lab = bank_label[idx]  # [N,k]
    llc = (nn_lab == query_label.view(-1, 1)).float().mean(dim=1)
    return llc

@torch.no_grad()
def gmm_2comp_pr(
        llc: torch.Tensor,
        gmm_tol: float = 1e-4,
        gmm_reg_covar: float = 1e-6,
        gmm_max_iter: int = 50,
        min_samples_gmm: int = 10,
        fallback: str = "constant",
        fallback_const: float = 0.5,
        q: float = 0.3
):
    if llc is None or llc.numel() == 0:
        return llc, {"ok": False, "reason": "empty"}

    x = llc.detach().cpu().numpy().reshape(-1, 1).astype(np.float64)
    N = int(x.shape[0])
    if N < 2:
        pr = np.full((N,), float(fallback_const), dtype=np.float32)
        pr_t = torch.from_numpy(pr).to(dtype=llc.dtype, device=llc.device)
        info = {"ok": False, "used_gmm": False, "reason": f"too_few_samples(N={N})", "N": N}
        return pr_t, info

    if N < int(min_samples_gmm):
        xx = x.reshape(-1)

        if fallback == "minmax":
            denom = (xx.max() - xx.min()) + 1e-12
            pr = ((xx - xx.min()) / denom).astype(np.float32)
        elif fallback == "quantile":
            thr = float(np.quantile(xx, q))
            pr = (xx >= thr).astype(np.float32)
        else:
            pr = np.full((N,), float(fallback_const), dtype=np.float32)

        pr_t = torch.from_numpy(pr).to(dtype=llc.dtype, device=llc.device)
        info = {
            "ok": False,
            "used_gmm": False,
            "reason": f"small_sample(N={N})",
            "N": N,
            "fallback": fallback,
        }
        if fallback == "quantile":
            info["thr"] = thr
        return pr_t, info

    try:
        gmm = GaussianMixture(
            n_components=2,
            max_iter=int(gmm_max_iter),
            tol=float(gmm_tol),
            reg_covar=float(gmm_reg_covar),
            warm_start=False
        )
        gmm.fit(x)

        probs = gmm.predict_proba(x)
        means = gmm.means_.reshape(-1)
        clean_idx = int(np.argmax(means))
        pr = probs[:, clean_idx].astype(np.float32)

        pr_t = torch.from_numpy(pr).to(dtype=llc.dtype, device=llc.device)
        info = {
            "ok": True,
            "used_gmm": True,
            "N": N,
            "means": means.tolist(),
            "clean_idx": clean_idx,
            "weights": gmm.weights_.reshape(-1).tolist(),
            "covars": gmm.covariances_.reshape(-1).tolist(),
            "n_iter": int(getattr(gmm, "n_iter_", -1)),
            "converged": bool(getattr(gmm, "converged_", False)),
        }
        return pr_t, info

    except Exception as e:
        xx = x.reshape(-1)
        denom = (xx.max() - xx.min()) + 1e-12
        pr = ((xx - xx.min()) / denom).astype(np.float32)

        pr_t = torch.from_numpy(pr).to(dtype=llc.dtype, device=llc.device)
        info = {
            "ok": False,
            "used_gmm": False,
            "reason": f"gmm_exception: {type(e).__name__}: {str(e)}",
            "N": N,
            "fallback": "minmax",
        }
        return pr_t, info

@torch.no_grad()
def top_tau_mask_from_pr(pr: torch.Tensor, tau: float = 0.5):

    if pr.numel() == 0:
        return torch.empty(0, dtype=torch.bool)

    tau = float(tau)
    tau = max(0.0, min(1.0, tau))
    N = pr.numel()
    k = max(1, int(np.ceil(tau * N)))
    # 取 top-k
    _, idx = torch.topk(pr, k=k, largest=True)
    keep = torch.zeros(N, dtype=torch.bool)
    keep[idx] = True
    return keep

@torch.no_grad()
def build_pixel_keep_mask_from_regions(region_refs, keep_region: torch.Tensor,
                                       h: int, w: int):

    if len(region_refs) == 0:
        return None

    B = 0
    for (b, _) in region_refs:
        B = max(B, b + 1)

    keep_hw = torch.zeros((B, h, w), dtype=torch.bool)
    keep_region = keep_region.detach().cpu()

    for i, (b, pts) in enumerate(region_refs):
        if not bool(keep_region[i].item()):
            continue
        for (y, x) in pts:
            keep_hw[b, y, x] = True
    return keep_hw

