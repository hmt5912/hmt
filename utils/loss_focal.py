import torch
import torch.nn as nn


# Helper function to enable loss function to be flexibly used for
# both 2D or 3D image segmentation - source: https://github.com/frankkramer-lab/MIScnn

def identify_axis(shape):
    # Three dimensional
    if len(shape) == 5:
        return [2, 3, 4]

    # Two dimensional
    elif len(shape) == 4:
        return [2, 3]

    # Exception - Unknown
    else:
        raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')


class SymmetricFocalLoss_orignal(nn.Module):
    """
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        Focal Tversky loss' focal parameter controls degree of down-weighting of easy examples, by default 2.0
    epsilon : float, optional
        clip values to prevent division by zero error
    """

    def __init__(self, delta=0.01, gamma=2., epsilon=1e-07, ignore_index=255):
        super(SymmetricFocalLoss, self).__init__()
        self.delta = delta # 较小会倾向于前景
        self.gamma = gamma  # 较大会使得模型更专注于难分类的样本
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        cross_entropy = -y_true * torch.log(y_pred)

        # Calculate losses separately for each class
        back_ce = torch.pow(1 - y_pred[:, 0, :, :], self.gamma) * cross_entropy[:, 0, :, :]
        back_ce = (1 - self.delta) * back_ce

        fore_ce = torch.pow(1 - y_pred[:, 1, :, :], self.gamma) * cross_entropy[:, 1, :, :]
        fore_ce = self.delta * fore_ce

        loss = torch.mean(torch.sum(torch.stack([back_ce, fore_ce], axis=-1), axis=-1))

        return loss


class AsymmetricFocalLoss(nn.Module):
    """For Imbalanced datasets
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.25
    gamma : float, optional
        Focal Tversky loss' focal parameter controls degree of down-weighting of easy examples, by default 2.0
    epsilon : float, optional
        clip values to prevent division by zero error
    """

    def __init__(self, delta=0.1, gamma=2., epsilon=1e-07, ignore_index=255):
        super(AsymmetricFocalLoss, self).__init__()
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        cross_entropy = -y_true * torch.log(y_pred)

        # Calculate losses separately for each class, only suppressing background class
        back_ce = torch.pow(1 - y_pred[:, 0, :, :], self.gamma) * cross_entropy[:, 0, :, :]
        back_ce = (1 - self.delta) * back_ce

        fore_ce = cross_entropy[:, 1, :, :]
        fore_ce = self.delta * fore_ce

        loss = torch.mean(torch.sum(torch.stack([back_ce, fore_ce], axis=-1), axis=-1))

        return loss


class SymmetricFocalTverskyLoss_orignal(nn.Module):
    """This is the implementation for binary segmentation.
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 0.75
    smooth : float, optional
        smooithing constant to prevent division by 0 errors, by default 0.000001
    epsilon : float, optional
        clip values to prevent division by zero error
    """

    def __init__(self, delta=0.1, gamma=0.75, epsilon=1e-07, ignore_index=255):
        super(SymmetricFocalTverskyLoss, self).__init__()
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        axis = identify_axis(y_true.size())

        # Calculate true positives (tp), false negatives (fn) and false positives (fp)
        tp = torch.sum(y_true * y_pred, axis=axis)
        fn = torch.sum(y_true * (1 - y_pred), axis=axis)
        fp = torch.sum((1 - y_true) * y_pred, axis=axis)
        dice_class = (tp + self.epsilon) / (tp + self.delta * fn + (1 - self.delta) * fp + self.epsilon)

        # Calculate losses separately for each class, enhancing both classes
        back_dice = (1 - dice_class[:, 0]) * torch.pow(1 - dice_class[:, 0], -self.gamma)
        fore_dice = (1 - dice_class[:, 1]) * torch.pow(1 - dice_class[:, 1], -self.gamma)

        # Average class scores
        loss = torch.mean(torch.stack([back_dice, fore_dice], axis=-1))
        return loss


class AsymmetricFocalTverskyLoss(nn.Module):
    """This is the implementation for binary segmentation.
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 0.75
    smooth : float, optional
        smooithing constant to prevent division by 0 errors, by default 0.000001
    epsilon : float, optional
        clip values to prevent division by zero error
    """

    def __init__(self, delta=0.1, gamma=0.75, epsilon=1e-07, ignore_index=255):
        super(AsymmetricFocalTverskyLoss, self).__init__()
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        # Clip values to prevent division by zero error
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        axis = identify_axis(y_true.size())

        # Calculate true positives (tp), false negatives (fn) and false positives (fp)
        tp = torch.sum(y_true * y_pred, axis=axis)
        fn = torch.sum(y_true * (1 - y_pred), axis=axis)
        fp = torch.sum((1 - y_true) * y_pred, axis=axis)
        dice_class = (tp + self.epsilon) / (tp + self.delta * fn + (1 - self.delta) * fp + self.epsilon)

        # Calculate losses separately for each class, only enhancing foreground class
        back_dice = (1 - dice_class[:, 0])
        fore_dice = (1 - dice_class[:, 1]) * torch.pow(1 - dice_class[:, 1], -self.gamma)

        # Average class scores
        loss = torch.mean(torch.stack([back_dice, fore_dice], axis=-1))
        return loss


import torch
import torch.nn as nn
import torch.nn.functional as F


class SymmetricFocalLoss(nn.Module):
    """
    多类别对称Focal Loss - 保持原始设计理念

    原始特性：
    1. 对每个类别单独计算损失
    2. 背景(类别0)使用(1-delta)*[(1-p)^gamma * CE]
    3. 前景类别使用delta*[(1-p)^gamma * CE]
    4. 忽略标签处理

    Parameters
    ----------
    delta : float, optional
        控制前景类别的权重，默认0.01（较小值倾向于前景）
    gamma : float, optional
        Focal参数，控制简单样本的降权程度，默认2.0
    reduction : str, optional
        损失归约方式，'mean'或'sum'，默认'mean'
    ignore_index : int, optional
        需要忽略的标签索引，默认255
    """

    def __init__(self, delta=0.01, gamma=2.0, reduction="mean", ignore_index=255):
        super(SymmetricFocalLoss, self).__init__()
        self.delta = delta
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        """
        Parameters
        ----------
        inputs : torch.Tensor
            模型预测的logits，形状为(B, C, H, W)
        targets : torch.Tensor
            真实标签，形状为(B, H, W)，值为[0, C-1]，ignore_index表示忽略的像素
        """
        # 获取类别数
        num_classes = inputs.shape[1]

        # 对logits应用softmax获取概率
        probs = F.softmax(inputs, dim=1)

        # 创建one-hot编码的目标
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        # 创建忽略掩码
        if self.ignore_index is not None:
            ignore_mask = (targets == self.ignore_index).unsqueeze(1).expand_as(targets_one_hot)
            valid_mask = ~ignore_mask
        else:
            valid_mask = torch.ones_like(targets_one_hot, dtype=torch.bool)

        # 计算每个像素每个类别的交叉熵损失
        # CE = -y_true * log(p)
        ce_per_class = -targets_one_hot * torch.log(torch.clamp(probs, min=1e-7))

        # 计算每个类别的focal权重：(1-p)^gamma
        focal_weight_per_class = torch.pow(1 - probs, self.gamma)

        # 应用focal权重
        focal_ce_per_class = focal_weight_per_class * ce_per_class

        # 应用对称权重：背景(0)使用(1-delta)，前景使用delta
        class_weights = torch.ones(num_classes, device=inputs.device)
        class_weights[0] = 1 - self.delta  # 背景权重
        for i in range(1, num_classes):
            class_weights[i] = self.delta  # 前景权重

        # 应用类别权重
        weighted_loss_per_class = focal_ce_per_class * class_weights.view(1, -1, 1, 1)

        # 将忽略像素的损失设为0
        if self.ignore_index is not None:
            weighted_loss_per_class = weighted_loss_per_class * valid_mask.float()

        # 按类别求和得到每个像素的总损失
        loss_per_pixel = torch.sum(weighted_loss_per_class, dim=1)

        # 根据reduction参数归约损失
        if self.reduction == "mean":
            if self.ignore_index is not None:
                # 只计算有效像素的平均
                valid_pixels = (targets != self.ignore_index)
                if valid_pixels.sum() > 0:
                    return torch.sum(loss_per_pixel) / valid_pixels.sum()
                else:
                    return torch.tensor(0.0, device=inputs.device)
            else:
                return torch.mean(loss_per_pixel)

        elif self.reduction == "sum":
            return torch.sum(loss_per_pixel)

        else:  # 'none'
            return loss_per_pixel


class SymmetricFocalTverskyLoss(nn.Module):
    """
    结合Tversky损失的对称Focal Loss，适用于类别不平衡
    参考Tversky损失：TP / (TP + α*FP + β*FN)
    """

    def __init__(self, alpha=0.3, beta=0.7, delta=0.01, gamma=2.0,
                 smooth=1e-7, reduction="mean", ignore_index=255):
        super(SymmetricFocalTverskyLoss, self).__init__()
        self.alpha = alpha  # FP权重
        self.beta = beta  # FN权重
        self.delta = delta
        self.gamma = gamma
        self.smooth = smooth
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # 获取类别数
        num_classes = inputs.shape[1]

        # 应用softmax获取概率
        probs = F.softmax(inputs, dim=1)

        # 创建one-hot编码的目标
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        # 创建忽略掩码
        if self.ignore_index is not None:
            ignore_mask = (targets == self.ignore_index).unsqueeze(1).expand_as(probs)
            probs = probs.clone()
            targets_one_hot = targets_one_hot.clone()
            probs[ignore_mask] = 0.0
            targets_one_hot[ignore_mask] = 0.0

        # 计算每个类别的TP, FP, FN
        tp = torch.sum(probs * targets_one_hot, dim=(2, 3))
        fp = torch.sum(probs * (1 - targets_one_hot), dim=(2, 3))
        fn = torch.sum((1 - probs) * targets_one_hot, dim=(2, 3))

        # 计算Tversky指数
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        # 应用Focal权重
        focal_weight = (1 - tversky) ** self.gamma

        # 应用类别对称权重：背景(0)用1-delta，前景用delta
        class_weights = torch.ones(num_classes, device=inputs.device)
        class_weights[0] = 1 - self.delta
        for i in range(1, num_classes):
            class_weights[i] = self.delta

        # 计算损失
        loss = focal_weight * (1 - tversky) * class_weights.unsqueeze(0)

        # 归约损失
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

class SymmetricUnifiedFocalLoss(nn.Module):
    """The Unified Focal loss is a new compound loss function that unifies Dice-based and cross entropy-based loss functions into a single framework.
    Parameters
    ----------
    weight : float, optional
        represents lambda parameter and controls weight given to symmetric Focal Tversky loss and symmetric Focal loss, by default 0.5
    delta : float, optional
        controls weight given to each class, by default 0.6
    gamma : float, optional
        focal parameter controls the degree of background suppression and foreground enhancement, by default 0.5
    epsilon : float, optional
        clip values to prevent division by zero error
    """

    def __init__(self, weight=0.5, delta=0.6, gamma=0.5, reduction="mean", ignore_index=255):
        super(SymmetricUnifiedFocalLoss, self).__init__()
        self.weight = weight
        self.delta = delta
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, y_pred, y_true):
        symmetric_ftl = SymmetricFocalTverskyLoss(delta=self.delta, gamma=self.gamma, reduction="mean", ignore_index=255)(y_pred, y_true)
        symmetric_fl = SymmetricFocalLoss(delta=self.delta, gamma=self.gamma, reduction="mean", ignore_index=255)(y_pred, y_true)
        if self.weight is not None:
            return (self.weight * symmetric_ftl) + ((1 - self.weight) * symmetric_fl)
        else:
            return symmetric_ftl + symmetric_fl


class AsymmetricUnifiedFocalLoss(nn.Module):
    """The Unified Focal loss is a new compound loss function that unifies Dice-based and cross entropy-based loss functions into a single framework.
    Parameters
    ----------
    weight : float, optional
        represents lambda parameter and controls weight given to asymmetric Focal Tversky loss and asymmetric Focal loss, by default 0.5
    delta : float, optional
        controls weight given to each class, by default 0.6
    gamma : float, optional
        focal parameter controls the degree of background suppression and foreground enhancement, by default 0.5
    epsilon : float, optional
        clip values to prevent division by zero error
    """

    def __init__(self, weight=0.5, delta=0.6, gamma=0.2, ignore_index=255):
        super(AsymmetricUnifiedFocalLoss, self).__init__()
        self.weight = weight
        self.delta = delta
        self.gamma = gamma

    def forward(self, y_pred, y_true):
        # Obtain Asymmetric Focal Tversky loss
        asymmetric_ftl = AsymmetricFocalTverskyLoss(delta=self.delta, gamma=self.gamma)(y_pred, y_true)

        # Obtain Asymmetric Focal loss
        asymmetric_fl = AsymmetricFocalLoss(delta=self.delta, gamma=self.gamma)(y_pred, y_true)

        # Return weighted sum of Asymmetrical Focal loss and Asymmetric Focal Tversky loss
        if self.weight is not None:
            return (self.weight * asymmetric_ftl) + ((1 - self.weight) * asymmetric_fl)
        else:
            return asymmetric_ftl + asymmetric_fl