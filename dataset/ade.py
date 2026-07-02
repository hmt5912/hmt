import os
import random

import numpy as np
import torch.utils.data as data
import torchvision as tv
from PIL import Image
from torch import distributed

from .utils import Subset, filter_images, group_images
# 允许数据集在每个任务中增量地加载新标签并忽略背景/未来类别
classes = [
    "void", "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed ", "windowpane",
    "grass", "cabinet", "sidewalk", "person", "earth", "door", "table", "mountain", "plant",
    "curtain", "chair", "car", "water", "painting", "sofa", "shelf", "house", "sea", "mirror",
    "rug", "field", "armchair", "seat", "fence", "desk", "rock", "wardrobe", "lamp", "bathtub",
    "railing", "cushion", "base", "box", "column", "signboard", "chest of drawers", "counter",
    "sand", "sink", "skyscraper", "fireplace", "refrigerator", "grandstand", "path", "stairs",
    "runway", "case", "pool table", "pillow", "screen door", "stairway", "river", "bridge",
    "bookcase", "blind", "coffee table", "toilet", "flower", "book", "hill", "bench", "countertop",
    "stove", "palm", "kitchen island", "computer", "swivel chair", "boat", "bar", "arcade machine",
    "hovel", "bus", "towel", "light", "truck", "tower", "chandelier", "awning", "streetlight",
    "booth", "television receiver", "airplane", "dirt track", "apparel", "pole", "land",
    "bannister", "escalator", "ottoman", "bottle", "buffet", "poster", "stage", "van", "ship",
    "fountain", "conveyer belt", "canopy", "washer", "plaything", "swimming pool", "stool",
    "barrel", "basket", "waterfall", "tent", "bag", "minibike", "cradle", "oven", "ball", "food",
    "step", "tank", "trade name", "microwave", "pot", "animal", "bicycle", "lake", "dishwasher",
    "screen", "blanket", "sculpture", "hood", "sconce", "vase", "traffic light", "tray", "ashcan",
    "fan", "pier", "crt screen", "plate", "monitor", "bulletin board", "shower", "radiator",
    "glass", "clock", "flag"
]


class AdeSegmentation(data.Dataset):

    def __init__(self, root, train=True, transform=None):

        root = os.path.expanduser(root)
        base_dir = "ADEChallengeData2016"
        ade_root = os.path.join(root, base_dir)
        if train:
            split = 'training'
        else:
            split = 'validation'
        annotation_folder = os.path.join(ade_root, 'annotations', split) # 数据和标签路径都是分成training和validation
        image_folder = os.path.join(ade_root, 'images', split)

        self.images = []
        fnames = sorted(os.listdir(image_folder))
        # os.listdir(image_folder)：列出 image_folder 文件夹中的所有文件名（不包括路径），返回一个包含文件名的列表，
        # sorted对文件名列表进行排序，以确保文件的顺序是稳定的
        self.images = [
            (os.path.join(image_folder, x), os.path.join(annotation_folder, x[:-3] + "png"))
            for x in fnames
        ]
# 创建一个列表 self.images，每个元素是一个包含两个路径的元组 (image_path, annotation_path)，
        # 通过 x[:-3] 去除文件名的最后三个字符（即 .jpg 或 .jpeg），然后将其替换为 .png
         # self.images 列表的一个元素可能是('/path/to/images/image_1.jpg', '/path/to/annotations/image_1.png')
        self.transform = transform
        # self.images 列表，列表中的每个元素是图像路径与标签路径的元组

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index][0]).convert('RGB')
        target = Image.open(self.images[index][1])

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.images)
# 基础的 AdeSegmentation 类,下面的是扩展的适合增量学习的,能够根据不同的任务（task）加载不同的类别，并进行增量训练。

class AdeSegmentationIncremental(data.Dataset):

    def __init__(
        self,
        root,
        train=True,
        transform=None,
        labels=None,
        labels_old=None,
        idxs_path=None,
        masking=True,
        overlap=True,
        data_masking="current",
        ignore_test_bg=False,
        class_ratio=0.5,
        sample_ratio2 = 0.8,
        sample_ratio1 = 0.6,
        **kwargs
    ):
# idxs_path：存储数据集索引路径，帮助选择要使用的图像
# masking：是否进行掩蔽操作，忽略某些类别。
# overlap：是否允许标签重叠。
# sample_ratio1 和 sample_ratio2：控制选择图像时的采样比例
        full_data = AdeSegmentation(root, train) # 获取数据和标签

        self.labels = []
        self.labels_old = []

        if labels is not None:
            # store the labels
            labels_old = labels_old if labels_old is not None else []

            self.__strip_zero(labels)
            self.__strip_zero(labels_old)

            assert not any(
                l in labels_old for l in labels
            ), "labels and labels_old must be disjoint sets" # 并确保它们是互不重叠的

            self.labels = labels
            self.labels_old = labels_old

            self.order = [0] + labels_old + labels # 多加一个背景

            if train:
                if len(labels) == 1:
        
                    # if idxs_path is not None and os.path.exists(idxs_path):
                    #     idxs = np.load(idxs_path).tolist()
                    # else:
                    idxs = filter_images(full_data, labels, labels_old, overlap=overlap) # 所有符合条件的图像的索引
                    idxs = random.sample(idxs, int(len(idxs)*sample_ratio1)) 
                        # if idxs_path is not None and distributed.get_rank() == 0:
                        #     np.save(idxs_path, np.array(idxs, dtype=int))
                else:
                    select_labels_num = int(len(labels) * class_ratio) # 论文中的先对类别取样，每个客户端有全部的任务，随机取样
                    tmp_labels = random.sample(labels, select_labels_num) 

                    # if idxs_path is not None and os.path.exists(idxs_path):
                    #     idxs = np.load(idxs_path).tolist()
                    # else:
                    idxs = filter_images(full_data, tmp_labels, labels_old, overlap=overlap)
                    idxs = random.sample(idxs, int(len(idxs)*(sample_ratio2)))  # 获取当前任务所有要用到的索引，对图像本身采样，也就是说不是一个类被的全部数据都被用起来
                        # if idxs_path is not None and distributed.get_rank() == 0:
                        #     np.save(idxs_path, np.array(idxs, dtype=int))

            else:
                # if idxs_path is not None and os.path.exists(idxs_path):
                #     idxs = np.load(idxs_path).tolist()
                # else:
                idxs = filter_images(full_data, labels, labels_old, overlap=overlap)
                    # if idxs_path is not None and distributed.get_rank() == 0:
                    #     np.save(idxs_path, np.array(idxs, dtype=int))
# 通过过滤函数（filter_images）选择满足标签条件的图像，并根据任务需求应用标签掩蔽。
            #if train:
            #    masking_value = 0
            #else:
            #    masking_value = 255

            #self.inverted_order = {label: self.order.index(label) for label in self.order}
            #self.inverted_order[0] = masking_value

            self.inverted_order = {label: self.order.index(label) for label in self.order}
            # self.inverted_order 是self.order中的label的索引
            if ignore_test_bg:
                masking_value = 255 # 255是背景区域，255表示不需要关注的像素
                self.inverted_order[0] = masking_value
            else:
                masking_value = 0  # 保留为背景类，不忽略，因此在训练时应该为false，Future classes will be considered as background.
            self.inverted_order[255] = 255
# self.inverted_order：为每个标签分配一个新的索引，处理增量任务中的标签转化。掩蔽操作：未来类别将作为背景类别进行处理，避免其影响模型训练
            reorder_transform = tv.transforms.Lambda(
                lambda t: t.apply_(
                    lambda x: self.inverted_order[x] if x in self.inverted_order else masking_value
                ) # 全部标签查找值
            )

            if masking: # 训练时为true，如果 masking=True，则只有当前任务的标签会被保留
                target_transform = tv.transforms.Lambda(
                    lambda t: t.
                    apply_(lambda x: self.inverted_order[x] if x in self.labels else masking_value)
                )
            else:
                target_transform = reorder_transform # 否则是全部标签

            # make the subset of the dataset,target_transform：对目标标签（分割标签）进行变换，使得在增量学习中旧类别和新类别能正确映射
            self.dataset = Subset(full_data, idxs, transform, target_transform)# 用这些操作处理一下得到sample, target
        else:
            self.dataset = full_data

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """

        return self.dataset[index]

    @staticmethod
    def __strip_zero(labels):
        while 0 in labels:
            labels.remove(0)

    def __len__(self):
        return len(self.dataset)


class RibFracSegmentationIncremental(data.Dataset):
    def __init__(
            self,
            root,
            train=True,
            transform=None,
            labels=None,
            labels_old=None,
            idxs_path=None,
            masking=True,
            overlap=True,
            data_masking="current",
            ignore_test_bg=False,
            class_ratio=0.5,
            sample_ratio2=0.8,
            sample_ratio1=0.6,
            **kwargs
    ):
        # 假设数据集的根路径和文件夹结构类似于其他语义分割任务
        root = os.path.expanduser(root)
        ribfrac_root = os.path.join(root, "RibFrac")
        split = 'train' if train else 'test'
        image_folder = os.path.join(ribfrac_root, 'images', split)
        annotation_folder = os.path.join(ribfrac_root, 'annotations', split)

        # 获取图像和标注路径
        fnames = sorted(os.listdir(image_folder))
        self.images = [
            (os.path.join(image_folder, fname), os.path.join(annotation_folder, fname[:-3] + "png"))
            for fname in fnames
        ]

        self.transform = transform
        self.labels = labels
        self.labels_old = labels_old

        # 标签和标签的顺序，基于任务的增量设置
        if labels is not None:
            labels_old = labels_old if labels_old is not None else []
            self.__strip_zero(labels)
            self.__strip_zero(labels_old) # 移除0即背景
            assert not any(l in labels_old for l in labels), "labels and labels_old must be disjoint sets"

            self.labels = labels
            self.labels_old = labels_old
            self.order = [0] + labels_old + labels
        else:
            self.order = []

        # 数据过滤
        self.idxs = self.filter_images(self.images, labels, labels_old, overlap)
        self.dataset = Subset(self.images, self.idxs, transform)

        # 创建标签映射
        self.inverted_order = {label: self.order.index(label) for label in self.order}
        self.inverted_order[255] = 255  # 255用于忽略的类别

        # 应用标签转换
        self.target_transform = self.create_target_transform()

    def __getitem__(self, index):
        """返回图像和标注"""
        img_path, target_path = self.dataset[index]
        img = Image.open(img_path).convert('RGB')
        target = Image.open(target_path)

        if self.transform is not None:
            img, target = self.transform(img, target)

        # 对标签进行转换
        target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.dataset)

    def filter_images(self, images, labels, labels_old, overlap):
        """
        选择符合标签条件的图像，支持重叠标签的过滤。
        """
        # 在此函数中，根据标签、标签顺序和重叠规则过滤图像
        idxs = []
        for idx, (img_path, target_path) in enumerate(images):
            target = Image.open(target_path)
            # 根据标签和重叠规则选择数据
            # 这里你可以根据任务需求调整选择哪些图像
            # 比如：检查图像中的标签是否匹配
            idxs.append(idx)  # 举例，选择所有符合条件的图像
        return idxs

    def create_target_transform(self):
        """创建目标标签的转换操作"""
        return tv.transforms.Lambda(
            lambda target: target.apply_(
                lambda x: self.inverted_order.get(x, 255)  # 未来类别作为背景（255）
            )
        )

    @staticmethod
    def __strip_zero(labels):
        """去除零标签"""
        while 0 in labels:
            labels.remove(0)
