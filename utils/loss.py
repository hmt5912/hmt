import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def get_loss(loss_type):
    if loss_type == 'focal_loss':
        return FocalLoss(ignore_index=255, size_average=True)
    elif loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss(ignore_index=255, reduction='mean')


def soft_crossentropy(logits, labels, logits_old, mask_valid_pseudo,
                      mask_background, pseudo_soft, pseudo_soft_factor=1.0):
    if pseudo_soft not in ("soft_certain", "soft_uncertain"):
        raise ValueError(f"Invalid pseudo_soft={pseudo_soft}")
    nb_old_classes = logits_old.shape[1]
    bs, nb_new_classes, w, h = logits.shape

    loss_certain = F.cross_entropy(logits, labels, reduction="none", ignore_index=255)
    loss_uncertain = (torch.log_softmax(logits_old, dim=1) * torch.softmax(logits[:, :nb_old_classes], dim=1)).sum(dim=1)

    if pseudo_soft == "soft_certain":
        mask_certain = ~mask_background
        mask_uncertain = mask_valid_pseudo & mask_background
    elif pseudo_soft == "soft_uncertain":
        mask_certain = (mask_valid_pseudo & mask_background) | (~mask_background)
        mask_uncertain = ~mask_valid_pseudo & mask_background

    loss_certain = mask_certain.float() * loss_certain
    loss_uncertain = (~mask_certain).float() * loss_uncertain

    return loss_certain + pseudo_soft_factor * loss_uncertain

def criterion_self_entropy(outputs, pseudo_labels, mask_uncertain_pseudo, mask_valid_pseudo, mask_background, nb_old_classes, reduction):
    criterion = nn.CrossEntropyLoss(ignore_index=255, reduction=reduction)

    loss = criterion(outputs, pseudo_labels)
    outputs = torch.softmax(outputs, dim=1)
    mask_uncertain_pseudo = mask_uncertain_pseudo & mask_background
    mask_valid_pseudo = mask_valid_pseudo & mask_background
    #certain_value = mask_valid_pseudo * outputs
    loss_certain = -1/nb_old_classes * mask_valid_pseudo.float() * ((outputs * torch.log(outputs)).sum(dim=1))

    loss_uncertain = 1/nb_old_classes * mask_uncertain_pseudo.float() * ((outputs * torch.log(outputs)).sum(dim=1))
    loss = loss + loss_certain + loss_uncertain
    return loss

class FocalLoss(nn.Module):

    def __init__(self, alpha=1, gamma=2, reduction="mean", ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss

class DiceLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=255, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)

        valid_mask = (target != self.ignore_index)

        dice_loss = 0.0
        count = 0

        for c in range(self.num_classes):
            if c == self.ignore_index:
                continue

            pred_c = probs[:, c]
            gt_c = (target == c).float()

            pred_c = pred_c * valid_mask
            gt_c = gt_c * valid_mask

            if gt_c.sum() == 0:
                continue

            intersection = (pred_c * gt_c).sum()
            union = pred_c.sum() + gt_c.sum()

            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            dice_loss += (1 - dice)
            count += 1

        if count > 0:
            dice_loss /= count

        return dice_loss

class CombinedLoss1(nn.Module):
    def __init__(self, main_loss, dice_loss, dice_weight=1.0):
        super().__init__()
        self.main_loss = main_loss
        self.dice_loss = dice_loss
        self.dice_weight = dice_weight

    def forward(self, logits, target):
        loss_main = self.main_loss(logits, target)
        loss_dice = self.dice_loss(logits, target)
        return loss_main + self.dice_weight * loss_dice



class FocalLoss_1(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction="mean", ignore_index=255, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.class_weights = class_weights

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)

        pt = torch.exp(-ce_loss)

        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.class_weights is not None:
            focal_loss = focal_loss * self.class_weights[targets]

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss

class FocalLossNew(nn.Module):

    def __init__(self, alpha=0.1, gamma=2, reduction="mean", ignore_index=255, index=0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.index = index

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss

        mask_new = (targets >= 1).float()
        focal_loss = mask_new * focal_loss + (1. - mask_new) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class BCEWithLogitsLossWithIgnoreIndex(nn.Module):

    def __init__(self, reduction='mean', ignore_index=255):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        n_cl = torch.tensor(inputs.shape[1]).to(inputs.device)
        labels_new = torch.where(targets != self.ignore_index, targets, n_cl)
        targets = F.one_hot(labels_new, inputs.shape[1] + 1).float().permute(0, 3, 1, 2)
        targets = targets[:, :inputs.shape[1], :, :]
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        loss = loss.sum(dim=1)
        if self.reduction == 'mean':
            return torch.masked_select(loss, targets.sum(dim=1) != 0).mean()
        elif self.reduction == 'sum':
            return torch.masked_select(loss, targets.sum(dim=1) != 0).sum()
        else:
            return loss * targets.sum(dim=1)


class BCEWithLogitsLossWithIgnoreIndexSoftLabel(nn.Module):

    def __init__(self, ignore_index=255, eps=1e-8, reduction="mean"):
        super().__init__()
        self.ignore_index = ignore_index
        self.eps = eps
        self.reduction = reduction

    @staticmethod
    def _entropy(p, eps=1e-8):
        # p: [B,C,H,W] normalized over C
        return -(p * (p + eps).log()).sum(dim=1, keepdim=True)  # [B,1,H,W]

    def forward(
        self,
        student_logits,
        old_probs,
        old_classes: int,
        targets,
        prior_mat=None,
        lam: float = 0.1,
        gate_beta: float = 0.0,
        lam_min: float = 0.0,
        lam_max: float = 0.5,
    ):
        B, C_total, H, W = student_logits.shape
        C_old = int(old_classes)
        assert old_probs.shape[1] == C_old, f"old_probs channels {old_probs.shape[1]} != old_classes {C_old}"

        p_s = torch.softmax(student_logits, dim=1)

        valid = (targets != self.ignore_index)

        tgt = torch.where(valid, targets, torch.zeros_like(targets))
        tgt = tgt.clamp(min=0, max=C_total - 1).long()
        onehot = F.one_hot(tgt, num_classes=C_total).float().permute(0, 3, 1, 2)
        onehot = onehot * valid[:, None].float()

        p_old = old_probs

        valid_f = valid[:, None].float()  # [B,1,H,W]

        if prior_mat is not None:
            prior_mat = prior_mat.to(student_logits.device, dtype=student_logits.dtype)

            prior_old = prior_mat[:C_old, :C_old].contiguous()
            q = torch.einsum("bchw,cd->bdhw", p_old, prior_old)
            q = q * valid_f

            if gate_beta > 0.0:
                ent = self._entropy(p_old, eps=self.eps)  # [B,1,H,W]
                ent_norm = ent / (math.log(C_old + 1e-8))
                lam_map = (lam + gate_beta * ent_norm).clamp(lam_min, lam_max)
            else:
                lam_map = torch.full((B, 1, H, W), float(lam),
                                     device=student_logits.device, dtype=student_logits.dtype)

            lam_map = lam_map * valid_f

            p_fused = (1.0 - lam_map) * p_old + lam_map * q
            p_fused = p_fused / (p_fused.sum(dim=1, keepdim=True) + self.eps)
        else:
            p_fused = p_old * valid_f

        onehot[:, :C_old, :, :] = p_fused

        M = F.one_hot(tgt, num_classes=C_total).float().permute(0, 3, 1, 2)
        M = M * valid[:, None].float()

        N = H * W
        M_flat = M.view(B, C_total, N)
        T_flat = onehot.view(B, C_total, N)
        S_flat = p_s.view(B, C_total, N)

        Rt_num = torch.einsum("bkn,bcn->kc", M_flat, T_flat)
        Rs_num = torch.einsum("bkn,bcn->kc", M_flat, S_flat)
        denom = M_flat.sum(dim=(0, 2))

        present = denom > 0
        if present.sum() == 0:
            return torch.zeros((), device=student_logits.device, dtype=student_logits.dtype)

        Rt = Rt_num[present] / (denom[present][:, None] + self.eps)
        Rs = Rs_num[present] / (denom[present][:, None] + self.eps)

        Rt = Rt / (Rt.sum(dim=1, keepdim=True) + self.eps)
        Rs = Rs / (Rs.sum(dim=1, keepdim=True) + self.eps)

        kl = (Rt * ((Rt + self.eps).log() - (Rs + self.eps).log())).sum(dim=1)

        if self.reduction == "sum":
            return kl.sum()
        else:
            return kl.mean()


class BCEWithLogitsLossWithIgnoreIndexSoftLabel_fiss(nn.Module):


    def __init__(self, reduction='mean', ignore_index=255):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, old_outputs, old_classes, targets):
        n_cl = torch.tensor(inputs.shape[1]).to(inputs.device)
        labels_new = torch.where(targets != self.ignore_index, targets, n_cl)
        targets = F.one_hot(labels_new, inputs.shape[1] + 1).float().permute(0, 3, 1, 2)
        targets = targets[:, :inputs.shape[1], :, :]
        targets[:, :old_classes, :, :] = old_outputs
        outputs = torch.log_softmax(inputs, dim=1)
        labels = torch.softmax(targets, dim=1)
        loss = -(outputs * labels).mean(dim=1)
        return loss

class IcarlLoss(nn.Module):

    def __init__(self, reduction='mean', ignore_index=255, bkg=False):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.bkg = bkg

    def forward(self, inputs, targets, output_old):
        # inputs of size B x C x H x W
        n_cl = torch.tensor(inputs.shape[1]).to(inputs.device)
        labels_new = torch.where(targets != self.ignore_index, targets, n_cl)

        targets = F.one_hot(labels_new, inputs.shape[1] + 1).float().permute(0, 3, 1, 2)
        targets = targets[:, :inputs.shape[1], :, :]
        if self.bkg:
            targets[:, 1:output_old.shape[1], :, :] = output_old[:, 1:, :, :]
        else:
            targets[:, :output_old.shape[1], :, :] = output_old

        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        loss = loss.sum(dim=1)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class UnbiasedCrossEntropy(nn.Module):

    def __init__(self, old_cl=None, reduction='mean', ignore_index=255):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.old_cl = old_cl

    def forward(self, inputs, targets, mask=None):

        old_cl = self.old_cl
        outputs = torch.zeros_like(inputs)  # B, C (1+V+N), H, W
        den = torch.logsumexp(inputs, dim=1)  # B, H, W       den of softmax
        outputs[:, 0] = torch.logsumexp(inputs[:, 0:old_cl], dim=1) - den  # B, H, W       p(O)
        outputs[:, old_cl:] = inputs[:, old_cl:] - den.unsqueeze(dim=1)  # B, N, H, W    p(N_i)

        labels = targets.clone()  # B, H, W

        labels[targets < old_cl] = 0  # just to be sure that all labels old belongs to zero

        if mask is not None:
            labels[mask] = self.ignore_index
        loss = F.nll_loss(outputs, labels, ignore_index=self.ignore_index, reduction=self.reduction)

        return loss


def nca(
    similarities,
    targets,
    loss,
    class_weights=None,
    focal_gamma=None,
    scale=1,
    margin=0.,
    exclude_pos_denominator=True,
    hinge_proxynca=False,
    memory_flags=None,
):

    b = similarities.shape[0]
    c = similarities.shape[1]
    w = similarities.shape[-1]

    if margin > 0.:
        similarities = similarities.view(b, c, w * w)
        targets = targets.view(b * w * w)
        margins = torch.zeros_like(similarities)
        margins = margins.permute(0, 2, 1)
        margins[torch.arange(margins.shape[0]), targets, :] = margin
        margins = margins.permute(0, 2, 1)
        similarities = similarities - margin
        similarities = similarities.view(b, c, w, w)
        targets = targets.view(b, w, w)

    similarities = scale * similarities

    if exclude_pos_denominator:  # NCA-specific
        similarities = similarities - similarities.max(dim=1, keepdims=True)[0]  # Stability

        disable_pos = torch.zeros_like(similarities)
        disable_pos[torch.arange(len(similarities)),
                    targets] = similarities[torch.arange(len(similarities)), targets]

        numerator = similarities[torch.arange(similarities.shape[0]), targets]
        denominator = similarities - disable_pos

        losses = numerator - torch.log(torch.exp(denominator).sum(-1))
        if class_weights is not None:
            losses = class_weights[targets] * losses

        losses = -losses
        if hinge_proxynca:
            losses = torch.clamp(losses, min=0.)

        loss = torch.mean(losses)
        return loss

    return loss(similarities, targets)


class NCA(nn.Module):

    def __init__(self, scale=1., margin=0., ignore_index=255, reduction="mean"):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)
        self.scale = scale
        self.margin = margin

    def forward(self, inputs, targets):
        return nca(inputs, targets, self.ce, scale=self.scale, margin=self.margin)


class UnbiasedNCA(nn.Module):

    def __init__(self, scale=1., margin=0., old_cl=None, reduction='mean', ignore_index=255):
        super().__init__()
        self.unce = UnbiasedCrossEntropy(old_cl, reduction, ignore_index)
        self.scale = scale
        self.margin = margin

    def forward(self, inputs, targets):
        return nca(inputs, targets, self.unce, scale=self.scale, margin=self.margin)


class KnowledgeDistillationLoss(nn.Module):

    def __init__(self, reduction='mean', alpha=1., kd_cil_weights=False):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha
        self.kd_cil_weights = kd_cil_weights

    def forward(self, inputs, targets, mask=None):
        inputs = inputs.narrow(1, 0, targets.shape[1])

        outputs = torch.log_softmax(inputs, dim=1)
        labels = torch.softmax(targets * self.alpha, dim=1)

        loss = (outputs * labels).mean(dim=1)
        if self.kd_cil_weights:
            w = -(torch.softmax(targets, dim=1) * torch.log_softmax(targets, dim=1)).sum(dim=1) + 1.0
            loss = loss * w[:, None]

        if mask is not None:
            loss = loss * mask.float()

        if self.reduction == 'mean':
            outputs = -torch.mean(loss)
        elif self.reduction == 'sum':
            outputs = -torch.sum(loss)
        else:
            outputs = -loss

        return outputs

class ExcludedKnowledgeDistillationLoss(nn.Module):

    def __init__(self, reduction='mean', index_new=-1, new_reduction="gt",
                 initial_nb_classes=-1, temperature_semiold=1.0):
        super().__init__()
        self.reduction = reduction

        self.initial_nb_classes = initial_nb_classes
        self.temperature_semiold = temperature_semiold

        #assert index_new > 0, index_new
        self.index_new = index_new
        if new_reduction not in ("gt", "sum"):
            raise ValueError(f"Unknown new_reduction={new_reduction}")
        self.new_reduction = new_reduction

    def forward(self, inputs, targets, labels, mask=None):
        bs, ch_new, w, h = inputs.shape
        device = inputs.device
        labels_no_unknown = labels.clone()
        labels_no_unknown[labels_no_unknown == 255] = 0

        temperature_semiold = torch.ones(bs, self.index_new + 1, w , h).to(device)
        if self.index_new > self.initial_nb_classes:
            temperature_semiold[:, self.initial_nb_classes:self.index_new] = temperature_semiold[:, self.initial_nb_classes:self.index_new] / self.temperature_semiold

        new_inputs = torch.zeros(bs, self.index_new + 1, w, h).to(device)
        new_targets = torch.zeros(bs, self.index_new + 1, w, h).to(device)

        new_targets[:, 0] = 0.
        new_inputs[:, 0] = inputs[:, 0]
        new_targets[:, 1:self.index_new] = targets[:, 1:]
        new_inputs[:, 1:self.index_new] = inputs[:, 1:self.index_new]
        if self.new_reduction == "gt":
            nb_pixels = bs * w * h
            new_targets[:, self.index_new] = targets[:, 0]
            tmp = inputs.view(bs, ch_new, w * h).permute(0, 2, 1).reshape(nb_pixels, ch_new)[torch.arange(nb_pixels), labels_no_unknown.view(nb_pixels)]
            tmp = tmp.view(bs, w, h)
            new_inputs[:, self.index_new] = tmp
        elif self.new_reduction == "sum":
            new_inputs[:, self.index_new] = inputs[:, self.index_new:].sum(dim=1)

        loss_new = -(torch.log_softmax(temperature_semiold * new_inputs, dim=1) * torch.softmax(temperature_semiold * new_targets, dim=1)).sum(dim=1)

        old_inputs = torch.zeros(bs, self.index_new + 1, w, h).to(device)
        old_targets = torch.zeros(bs, self.index_new + 1, w, h).to(device)

        old_targets[:, 0] = targets[:, 0]
        old_inputs[:, 0] = inputs[:, 0]
        old_targets[:, 1:self.index_new] = targets[:, 1:self.index_new]
        old_inputs[:, 1:self.index_new] = inputs[:, 1:self.index_new]
        if self.new_reduction == "gt":
            old_targets[:, self.index_new] = 0.
            tmp = inputs.view(bs, ch_new, w * h).permute(0, 2, 1).reshape(nb_pixels, ch_new)[torch.arange(nb_pixels), labels_no_unknown.view(nb_pixels)]
            tmp = tmp.view(bs, w, h)
            old_inputs[:, self.index_new] = tmp
        elif self.new_reduction == "sum":
            old_inputs[:, self.index_new] = inputs[:, self.index_new:].sum(dim=1)

        loss_old = -(torch.log_softmax(temperature_semiold * old_inputs, dim=1) * torch.softmax(temperature_semiold * old_targets, dim=1)).sum(dim=1)

        mask_new = (labels >= self.index_new) & (labels < 255)
        mask_old = labels < self.index_new
        loss = (mask_new.float() * loss_new) + (mask_old.float() * loss_old)

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        return loss


class BCESigmoid(nn.Module):
    def __init__(self, reduction="mean", alpha=1.0, shape="trim"):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha
        self.shape = shape

    def forward(self, inputs, targets, mask=None):
        nb_old_classes = targets.shape[1]
        if self.shape == "trim":
            inputs = inputs[:, :nb_old_classes]
        elif self.shape == "sum":
            inputs[:, 0] = inputs[:, nb_old_classes:].sum(dim=1)
            inputs = inputs[:, :nb_old_classes]
        else:
            raise ValueError(f"Unknown parameter to handle shape = {self.shape}.")

        inputs = torch.sigmoid(self.alpha * inputs)
        targets = torch.sigmoid(self.alpha * targets)

        loss = F.binary_cross_entropy(inputs, targets, reduction=self.reduction)
        if mask is not None:
            loss = loss * mask.float()

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        return loss


class UnbiasedKnowledgeDistillationLoss(nn.Module):

    def __init__(self, reduction='mean', alpha=1.):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs, targets, mask=None):

        new_cl = inputs.shape[1] - targets.shape[1]

        targets = targets * self.alpha

        new_bkg_idx = torch.tensor([0] + [x for x in range(targets.shape[1], inputs.shape[1])]).to(
            inputs.device
        )

        den = torch.logsumexp(inputs, dim=1)  # B, H, W
        outputs_no_bgk = inputs[:, 1:-new_cl] - den.unsqueeze(dim=1)  # B, OLD_CL, H, W
        outputs_bkg = torch.logsumexp(
            torch.index_select(inputs, index=new_bkg_idx, dim=1), dim=1
        ) - den  # B, H, W

        labels = torch.softmax(targets, dim=1)
        loss = (labels[:, 0] * outputs_bkg +
                (labels[:, 1:] * outputs_no_bgk).sum(dim=1)) / targets.shape[1]

        if mask is not None:
            loss = loss * mask.float()

        if self.reduction == 'mean':
            outputs = -torch.mean(loss)
        elif self.reduction == 'sum':
            outputs = -torch.sum(loss)
        else:
            outputs = -loss

        return outputs


class MyselfKnowledgeDistillationLoss(nn.Module):

    def __init__(self, reduction='mean', alpha=1.):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs, targets, mask=None):

        new_cl = inputs.shape[1] - targets.shape[1]

        targets = targets * self.alpha

        new_bkg_idx = torch.tensor([0] + [x for x in range(targets.shape[1], inputs.shape[1])]).to(
            inputs.device
        )

        den = torch.logsumexp(inputs, dim=1)  # B, H, W
        outputs_no_bgk = inputs[:, 1:-new_cl] - den.unsqueeze(dim=1)  # B, OLD_CL, H, W

        labels = torch.softmax(targets, dim=1)
        loss = ((labels[:, 1:] * outputs_no_bgk).sum(dim=1)) / (targets.shape[1] -1)


        if mask is not None:
            loss = loss * mask.float()

        if self.reduction == 'mean':
            outputs = -torch.mean(loss)
        elif self.reduction == 'sum':
            outputs = -torch.sum(loss)
        else:
            outputs = -loss

        return outputs

#  hmt
class ImprovedFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', ignore_index=255):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

        if alpha is not None:
            self.register_buffer('alpha', torch.tensor(alpha))
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        device = inputs.device
        self.alpha = self.alpha.to(device)

        if self.ignore_index is not None:
            valid_mask = targets != self.ignore_index
            targets = targets.clone()
            targets[~valid_mask] = 0

        log_pt = F.log_softmax(inputs, dim=1)
        log_pt = log_pt.gather(1, targets.unsqueeze(1)).squeeze(1)  # [B, H, W]

        pt = torch.exp(log_pt)
        focal_weight = (1 - pt) ** self.gamma
        if self.alpha is not None:
            if self.alpha.dim() == 1:
                alpha_t = self.alpha.gather(0, targets.view(-1)).view_as(targets)
            else:
                alpha_t = self.alpha
            focal_weight = alpha_t * focal_weight

        loss = -focal_weight * log_pt

        if self.ignore_index is not None:
            loss = loss * valid_mask.float()

        if self.reduction == 'mean':
            if self.ignore_index is not None:
                return loss.sum() / (valid_mask.sum() + 1e-8)
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class CombinedLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.focal = ImprovedFocalLoss(
            alpha=[0.1] + [3.0] * (num_classes - 1),
            gamma=2.0
        )
        self.dice_weight = 0.5

    def dice_loss(self, inputs, targets):
        smooth = 1e-6
        pred = F.softmax(inputs, dim=1)

        dice = 0
        for cls in range(1, pred.shape[1]):
            pred_cls = pred[:, cls]
            target_cls = (targets == cls).float()

            intersection = (pred_cls * target_cls).sum()
            union = pred_cls.sum() + target_cls.sum()

            if union > 0:
                dice += (2. * intersection + smooth) / (union + smooth)

        return 1 - dice / (pred.shape[1] - 1)

    def forward(self, inputs, targets):
        focal = self.focal(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return focal + self.dice_weight * dice


class MedicalSegmentationLoss(nn.Module):

    def __init__(self, num_classes, alpha_weights=None, gamma=3.0,
                 dice_weight=0.4, ignore_index=255):
        super().__init__()
        self.num_classes = num_classes

        if alpha_weights is None:
            alpha_weights = [0.1] + [4.0] * (num_classes - 1)

        self.focal = ImprovedFocalLoss(
            alpha=alpha_weights,
            gamma=gamma,
            ignore_index=ignore_index,
            reduction='mean'
        )
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index

    def dice_loss(self, pred, target):
        smooth = 1e-6
        pred_softmax = F.softmax(pred, dim=1)

        valid_mask = (target != self.ignore_index) if self.ignore_index is not None else torch.ones_like(target,
                                                                                                         dtype=torch.bool)

        total_dice = 0
        count = 0

        for cls in range(1, self.num_classes):
            target_cls = (target == cls).float()

            if (target_cls * valid_mask.float()).sum() > 0:
                pred_cls = pred_softmax[:, cls]

                pred_cls = pred_cls * valid_mask.float()
                target_cls = target_cls * valid_mask.float()

                intersection = (pred_cls * target_cls).sum()
                union = pred_cls.sum() + target_cls.sum()

                if union > 0:
                    dice = (2. * intersection + smooth) / (union + smooth)
                    total_dice += dice
                    count += 1

        return 1 - (total_dice / count if count > 0 else 0)

    def forward(self, pred, target):
        focal_loss = self.focal(pred, target)
        dice_loss = self.dice_loss(pred, target)

        total_loss = focal_loss + self.dice_weight * dice_loss

        return total_loss



def _safe_normalize(p: torch.Tensor, dim: int = -1, eps: float = 1e-8):
    return p / (p.sum(dim=dim, keepdim=True) + eps)

@torch.no_grad()
def build_relation_label_per_pixel(
    gt_or_pseudo: torch.Tensor,
    probs_old: torch.Tensor,
    old_classes: int,
    ignore_index: int = 255,
    replace_background: bool = True,
):
    B, C, H, W = probs_old.shape
    labels = gt_or_pseudo.clone()
    valid = labels != ignore_index
    labels = torch.where(valid, labels, torch.zeros_like(labels))

    y = F.one_hot(labels.long(), num_classes=C).permute(0,3,1,2).float()
    start = 0
    end = old_classes if replace_background else max(old_classes, 1)
    y[:, start:end] = probs_old[:, start:end]

    y = y * valid[:, None].float()
    return y, valid

def loss_relation_consistency(
    logits_new: torch.Tensor,
    logits_old_aligned: torch.Tensor,
    pseudo_group: torch.Tensor,
    old_classes: int,
    ignore_index: int = 255,
    gamma_k= None,
    min_pixels_per_class: int = 8,
    temperature: float = 1.0,
    clinical_prior= None,
    prior_mode: str = "logit_add",
    prior_strength: float = 0.0,
):
    B, C, H, W = logits_new.shape
    device = logits_new.device

    if temperature != 1.0:
        p_new = torch.softmax(logits_new / temperature, dim=1)
        p_old = torch.softmax(logits_old_aligned / temperature, dim=1)
    else:
        p_new = torch.softmax(logits_new, dim=1)
        p_old = torch.softmax(logits_old_aligned, dim=1)

    if clinical_prior is not None and prior_strength > 0:
        if clinical_prior.dim() == 1:
            prior = clinical_prior.to(device)[None, :, None, None]
        else:
            prior = clinical_prior.to(device)

        if prior_mode == "logit_add":
            logits_prior = logits_old_aligned + prior_strength * prior
            p_old = torch.softmax(logits_prior / temperature, dim=1) if temperature != 1.0 else torch.softmax(logits_prior, dim=1)
        elif prior_mode == "prob_mix":
            p_prior = torch.softmax(prior, dim=1) if prior.shape[1] == C else prior
            p_old = (1 - prior_strength) * p_old + prior_strength * p_prior
            p_old = _safe_normalize(p_old, dim=1)
        else:
            raise ValueError(f"Unknown prior_mode={prior_mode}")

    with torch.no_grad():
        Y_rel, valid = build_relation_label_per_pixel(
            gt_or_pseudo=pseudo_group,
            probs_old=p_old,
            old_classes=old_classes,
            ignore_index=ignore_index,
            replace_background=True,
        )

    loss_k_list = []
    weight_k_list = []

    if gamma_k is not None:
        gamma_k = gamma_k.to(device).float()
        gamma_k = gamma_k / (gamma_k.mean() + 1e-8)
    else:
        gamma_k = torch.ones(C, device=device)

    for k in range(C):
        mask_k = (pseudo_group == k) & valid  # [B,H,W]
        num = mask_k.sum().item()
        if num < min_pixels_per_class:
            continue

        m = mask_k[:, None].float()  # [B,1,H,W]

        P_bar = (p_new * m).sum(dim=(0,2,3)) / (m.sum(dim=(0,2,3)) + 1e-8)
        Y_bar = (Y_rel * m).sum(dim=(0,2,3)) / (m.sum(dim=(0,2,3)) + 1e-8)   # [C]

        P_bar = _safe_normalize(P_bar, dim=0)
        Y_bar = _safe_normalize(Y_bar, dim=0)

        kl = (P_bar * (torch.log(P_bar + 1e-8) - torch.log(Y_bar + 1e-8))).sum()

        loss_k_list.append(kl)
        weight_k_list.append(gamma_k[k])

    if len(loss_k_list) == 0:
        return torch.zeros([], device=device, dtype=logits_new.dtype)

    loss_k = torch.stack(loss_k_list)
    w_k = torch.stack(weight_k_list)

    loss = (w_k * loss_k).sum() / (w_k.sum() + 1e-8)
    return loss
