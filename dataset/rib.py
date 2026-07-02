# dataset_ribfrac.py
import os
import random
import numpy as np
import torch
from torch.utils import data
from PIL import Image
import scipy.ndimage
import torchvision.transforms as transforms
from .utils import Subset, filter_images, group_images


classes={
            0: "background",
            1: "displaced",
            2: "non_displaced",
            3: "buckle",
            4: "segmental"
}

class_dis={
    0: "A normal rib with intact cortical continuity, smooth rib contour, and no visible fracture line or deformity on CT bone window",
    1: "A displaced rib fracture characterized by a clear cortical break with obvious fragment displacement, angulation, or overlap on CT bone window",
    2:"A non-displaced rib fracture showing a subtle cortical discontinuity or fine fracture line without visible displacement on CT bone window",
    3:"A buckle rib fracture presenting as focal cortical buckling or compression without a complete fracture line on CT bone window",
    4:"A segmental rib fracture with two or more fracture sites along the same rib, forming a free bone segment on CT bone window",
}

class MultiRootSliceDataset(data.Dataset):
    """
    txt 每行：
      1) "amos/amos_0192_z008"  -> prefix=amos, name=amos_0192_z008
      2) "lits/xxx_z012"        -> prefix=lits, name=xxx_z012
      3) "amos_0192_z008"       -> 无 prefix：会在所有 roots 里找
    roots_map:
      {"amos": "/path/to/amos_root", "lits": "/path/to/lits_root"}
    每个 root 结构必须是：
      root/{train,val}/images/*.npy
      root/{train,val}/labels/*.png
    """
    def __init__(self, roots_map, transform=None, file_list=None, search_sets=("train", "val")):
        self.roots_map = roots_map
        self.transform = transform

        if file_list is None:
            raise ValueError("file_list must be provided (from txt).")
        self.files = [x.strip() for x in file_list if x.strip()]

        self.search_sets = search_sets

        # 全局索引： (prefix, name) -> (img_path, lbl_path)
        self.key2paths = {}

        for prefix, root in self.roots_map.items():
            root = os.path.expanduser(root)
            for s in search_sets:
                img_dir = os.path.join(root, s, "images")
                lbl_dir = os.path.join(root, s, "labels")
                if not (os.path.isdir(img_dir) and os.path.isdir(lbl_dir)):
                    continue
                for fn in os.listdir(img_dir):
                    if not fn.endswith(".png"):
                        continue
                    name = os.path.splitext(fn)[0]
                    img_path = os.path.join(img_dir, fn)
                    lbl_path = os.path.join(lbl_dir, name + ".png")
                    if os.path.exists(lbl_path):
                        self.key2paths[(prefix, name)] = (img_path, lbl_path)

        # 预检查：txt 里找不到的样本
        missing = []
        for item in self.files:
            pref, name = self._parse_item(item)
            if pref is not None:
                if (pref, name) not in self.key2paths:
                    missing.append(item)
            else:
                # 无 prefix：只要任意 root 里存在即可
                ok = any((p, name) in self.key2paths for p in self.roots_map.keys())
                if not ok:
                    missing.append(item)

        if missing:
            print(f"[WARN] {len(missing)}/{len(self.files)} items not found. Examples: {missing[:10]}")

    @staticmethod
    def _parse_item(item: str):
        """返回 (prefix or None, name)"""
        item = item.strip()
        if "/" in item:
            pref, name = item.split("/", 1)
            return pref, name
        return None, item

    def _resolve_paths(self, item: str):
        pref, name = self._parse_item(item)

        if pref is not None:
            if (pref, name) not in self.key2paths:
                raise FileNotFoundError(f"Not found for '{item}' -> key ({pref},{name})")
            return self.key2paths[(pref, name)], pref, name

        # 无 prefix：在所有 roots 里找第一个匹配
        for p in self.roots_map.keys():
            key = (p, name)
            if key in self.key2paths:
                return self.key2paths[key], p, name

        raise FileNotFoundError(f"Not found for '{item}' in any roots. Parsed name='{name}'")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        item = self.files[idx]
        (img_path, lbl_path), pref, name = self._resolve_paths(item)

        img = np.array(Image.open(img_path).convert("L"), dtype=np.float32)
        target = Image.open(lbl_path)

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target


def load_indices_from_txt(dataset, txt_path):
    """
    dataset: amosFracSegmentation (full_amos)
    txt_path: e.g. splits_fcl/step_1/client_0.txt
    根据txt获取图像的索引
    """
    name_to_idx = {}
    for i, (img_path, _) in enumerate(dataset.images):
        name = os.path.splitext(os.path.basename(img_path))[0]
        name_to_idx[name] = i

    with open(txt_path, "r") as f:
        names = [line.strip() for line in f]

    idxs = [name_to_idx[n] for n in names if n in name_to_idx]
    return idxs


# class amosFracSegmentation(data.Dataset):
#     def __init__(self, root, image_set="train", transform=None, file_list=None):
#         self.root = os.path.expanduser(root)
#         self.image_set = image_set  # "train" or "val"
#         self.transform = transform
#
#         if file_list is None:
#             raise ValueError("AMOS2D: file_list must be provided (from txt).")
#         # flie_list只是图像名字
#
#         self.files = file_list
#
#     def __len__(self):
#         return len(self.files)
#
#     def __getitem__(self, idx):
#         name = self.files[idx]
#         img_path = os.path.join(self.root, self.image_set, "images", name + ".npy")
#         lbl_path = os.path.join(self.root, self.image_set, "labels", name + ".png")
#
#         img = np.load(img_path).astype(np.float32)  # [H,W] 把磁盘上的 .npy 文件读进来，并转成 float32 类型的 NumPy 数组
#         target = Image.open(lbl_path)               # PIL
#
#         if self.transform is not None:
#             img, target = self.transform(img, target)
#
#         return img, target

class RibFracSegmentation(data.Dataset):
    def __init__(self, root, image_set="train", transform=None, file_list=None,
                 search_sets=("train", "val")):
        self.root = os.path.expanduser(root)
        self.image_set = image_set  # 保留，但如果用 search_sets 就不强依赖它
        self.transform = transform

        if file_list is None:
            raise ValueError("rib2D: file_list must be provided (from txt).")
        self.files = file_list

        # 建立全局索引：name -> (img_path, lbl_path)
        self.name2paths = {}
        for s in search_sets:
            img_dir = os.path.join(self.root, s, "images")
            lbl_dir = os.path.join(self.root, s, "labels")
            if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
                continue

            # 以 images 里的 .npy 为准
            for fn in os.listdir(img_dir):
                if not fn.endswith(".png"):
                    continue
                name = os.path.splitext(fn)[0]
                img_path = os.path.join(img_dir, fn)
                lbl_path = os.path.join(lbl_dir, name + ".png")
                if os.path.exists(lbl_path):
                    # train/val 同名时：后面的会覆盖前面的
                    self.name2paths[name] = (img_path, lbl_path)

        # 可选：提前检查 txt 里有哪些 name 找不到
        missing = [n for n in self.files if n not in self.name2paths]
        if len(missing) > 0:
            print(f"[WARN] {len(missing)}/{len(self.files)} names not found in {search_sets}. "
                  f"Examples: {missing[:10]}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        if name not in self.name2paths:
            raise FileNotFoundError(f"name '{name}' not found in global index (train/val).")

        img_path, lbl_path = self.name2paths[name]
        #img = np.load(img_path).astype(np.float32)
        img = np.array(Image.open(img_path).convert("L"), dtype=np.float32)
        target = Image.open(lbl_path)

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target



class RibFracSegmentationIncremental(data.Dataset):
    def __init__(
            self,
            root,
            train=True,
            transform=None,
            labels=None,
            labels_old=None,
            step=0,
            task_name="3-1",
            client_id=0,
            split_root="splits",
            ignore_future=True,
            masking=True,
            overlap=True,
            data_masking="current",
            class_ratio=1.0,
            sample_ratio=0.8,
            **kwargs
    ):
        self.root = root
        self.train = train
        self.transform = transform
        self.p_fg = 0.5  # nnU-Net 默认

        image_set = "train" if train else "val"

        # 1) 读 txt：train 用 client_x；val 用 val.txt（全局）
        if train:
            split_file = os.path.join(
                root, split_root, "rib", task_name, f"step{step}", f"client_{client_id}.txt"
            )
        else:
            split_file = os.path.join(
                root, split_root, "rib", task_name, f"step{step}", "val.txt"
            )

        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file, "r") as f:
            file_list = [line.strip() for line in f if line.strip()]  # strip() 去掉换行符和空格。

        # base_ds = amosFracSegmentation(root, image_set=image_set, transform=None, file_list=file_list)
        roots_map = {
            "rib": "/mnt/newdisk/hmt/kt2/ribfrac_slice2",
        }


        base_ds = MultiRootSliceDataset(
            roots_map=roots_map,
            transform=None,
            file_list=file_list,
            search_sets=("train", "val")  # 让它去各自 root 的 train/val 里找
        )
        # 改动
        self.has_foreground = []

        for i in range(len(base_ds)):
            _, lbl = base_ds[i]  # 注意：这里 transform=None
            lbl = np.array(lbl)
            self.has_foreground.append((lbl > 0).any())
        print(f"[rib][{image_set}] step={step} "
              f"{'client=' + str(client_id) if train else 'global-val'} "
              f"samples={len(base_ds)}")

        # 2) 任务标签：当前/过去/累计
        labels = list(labels) if labels is not None else []
        labels_old = list(labels_old) if labels_old is not None else []
        # 不要把背景当成“前景类”处理（但映射里仍保留0）
        while 0 in labels: labels.remove(0)
        while 0 in labels_old: labels_old.remove(0)

        self.labels = [0] + labels
        self.labels_old = [0] + labels_old

        # 累计类别（去重保序）
        self.labels_cum = []  # labels_cum里包括0，labels和labels_old
        for x in (self.labels_old + self.labels):
            if x not in self.labels_cum:
                self.labels_cum.append(x)

        # 连续映射：原始类id -> [0..K-1]
        self.inverted_order = {lab: i for i, lab in enumerate(self.labels_cum)}
        self.inverted_order[255] = 255

        allowed_set = set(self.labels_cum) | {0, 255}
        # self.labels_cum 中的所有元素与固定的 0 和 255 合并，创建一个新的集合，set() 将其转换为集合（去重、无序）

        def target_transform(label):
            # label: numpy [H,W] or torch [H,W]
            if torch.is_tensor(label):
                t = label.clone()
                is_tensor = True
            else:
                t = np.array(label, dtype=np.int64)
                is_tensor = False

            out = np.full_like(t, 255)  # default ignore
            #raw_out = np.copy(t)

            # 当前 step 的类 → 前景（连续映射）
            for lab in labels:
                out[t == lab] = self.inverted_order[lab]

            #
            if masking and data_masking == "current":
                # 旧类 → background
                for lab in labels_old:
                    out[t == lab] = 0
            else:
                # 累积学习：旧类也作为前景
                for lab in labels_old:
                    out[t == lab] = self.inverted_order[lab]

            # 背景
            out[t == 0] = 0

            # ignore
            out[t == 255] = 255

            if is_tensor:
                return torch.from_numpy(out).long().to(label.device)#, torch.from_numpy(raw_out).long().to(label.device)
            else:
                return out#,raw_out

        indices = list(range(len(base_ds)))
        self.dataset = Subset(base_ds, indices, transform=transform, target_transform=target_transform)

    def __len__(self):
        return len(self.dataset)

    # 改动
    def __getitem__(self, idx):
        if self.train and random.random() < self.p_fg:
            # 强制采样前景 slice
            fg_indices = [i for i, v in enumerate(self.has_foreground) if v]
            if len(fg_indices) > 0:
                idx = random.choice(fg_indices)


        return self.dataset[idx]

class PreprocessorTransform:
    def __init__(self, hu_clip=(-1000, 1000), zscore=True, augment=True,
                 out_size=256, n_patches=3, max_tries=30, min_fg_pixels=10,
                 fg_value_threshold=0):
        self.hu_clip = hu_clip
        self.zscore = zscore
        self.augment = augment
        self.out_size = out_size
        self.n_patches = n_patches
        self.max_tries = max_tries
        self.min_fg_pixels = min_fg_pixels
        self.fg_value_threshold = fg_value_threshold  # label > threshold 视为前景（默认 >0）

    def _resize_min_size(self, img, label, min_size):
        H, W = label.shape
        scale = max(min_size / H, min_size / W, 1.0)
        if scale > 1.0:
            img = scipy.ndimage.zoom(img, (scale, scale), order=1, prefilter=False)
            label = scipy.ndimage.zoom(label, (scale, scale), order=0, prefilter=False)
        return img, label

    def _clamp_window(self, y0, x0, size, H, W):
        y0 = int(max(0, min(y0, H - size)))
        x0 = int(max(0, min(x0, W - size)))
        return y0, x0
    # 把裁剪窗口左上角坐标限制在合法范围内，保证 [y0:y0+size, x0:x0+size] 不越界

    def _sample_one_fg_patch(self, img, label, size):
        """采样 1 个强制含前景的 patch（失败则退化为前景中心裁剪）"""
        H, W = label.shape
        fg = (label > self.fg_value_threshold) & (label != 255)
        if not fg.any():
            # 无前景：退化为随机裁（也可以改成中心裁）
            y0 = random.randint(0, H - size)
            x0 = random.randint(0, W - size)
            return img[y0:y0+size, x0:x0+size], label[y0:y0+size, x0:x0+size]

        ys, xs = np.where(fg)
        coords = list(zip(ys, xs))

        for _ in range(self.max_tries):
            cy, cx = coords[random.randrange(len(coords))]
            y0 = cy - random.randint(0, size - 1)
            x0 = cx - random.randint(0, size - 1)
            y0, x0 = self._clamp_window(y0, x0, size, H, W)

            crop_lbl = label[y0:y0+size, x0:x0+size]
            if ((crop_lbl > self.fg_value_threshold) & (crop_lbl != 255)).sum() >= self.min_fg_pixels:
                crop_img = img[y0:y0+size, x0:x0+size]
                return crop_img, crop_lbl

        # fallback：前景中心裁
        cy = int(np.mean(ys))
        cx = int(np.mean(xs))
        y0 = cy - size // 2
        x0 = cx - size // 2
        y0, x0 = self._clamp_window(y0, x0, size, H, W)
        return img[y0:y0+size, x0:x0+size], label[y0:y0+size, x0:x0+size]

    def __call__(self, img: np.ndarray, label: Image.Image):
        label = np.array(label).astype(np.int64)

        # HU clip
        if self.hu_clip is not None:
            low, high = -125, 275
            img = np.clip(img, *self.hu_clip)
            img = (img - low) / (high - low)

            # z-score（你原逻辑）
        if self.zscore:
            mask = img != 0
            if mask.sum() > 0:
                mean = img[mask].mean()
                std = img[mask].std()
                img = (img - mean) / (std + 1e-8)

        # augment（对整张图做一次，再切patch）
        if self.augment:
            if random.random() > 0.5:
                img = np.flip(img, axis=1).copy()
                label = np.flip(label, axis=1).copy()
            if random.random() > 0.5:
                k = random.randint(0, 3)
                img = np.rot90(img, k).copy()
                label = np.rot90(label, k).copy()
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                img = scipy.ndimage.rotate(img, angle, reshape=False, order=1)
                label = scipy.ndimage.rotate(label, angle, reshape=False, order=0)
                img = np.ascontiguousarray(img).copy()
                label = np.ascontiguousarray(label).copy()

        # 保证尺寸够切 128
        img, label = self._resize_min_size(img, label, self.out_size)

        # 采 N 个强制前景 patch
        img_patches = []
        lbl_patches = []
        for _ in range(self.n_patches):
            p_img, p_lbl = self._sample_one_fg_patch(img, label, self.out_size)
            img_patches.append(p_img)
            lbl_patches.append(p_lbl)

        # -> torch
        # img: [N, 3, H, W], lbl: [N, H, W]
        img_patches = np.stack(img_patches, axis=0)  # [N,H,W]
        img_patches = np.stack([img_patches] * 3, axis=1)  # [N,3,H,W]
        lbl_patches = np.stack(lbl_patches, axis=0)  # [N,H,W]

        # 强制连续 + 可写（防 memmap/负stride/不可resize storage）
        img_patches = np.ascontiguousarray(img_patches).copy()
        lbl_patches = np.ascontiguousarray(lbl_patches).copy()

        img_t = torch.from_numpy(img_patches).float()
        lbl_t = torch.from_numpy(lbl_patches).long()
        return img_t, lbl_t

def collate_flatten_patches(batch):
    """
        batch: list of (img_t, lbl_t)
          img_t: [N,3,128,128]
          lbl_t: [N,128,128]
        return:
          imgs: [B*N,3,128,128]
          lbls: [B*N,128,128]
        """
    num_returns = len(batch[0])

    if num_returns == 3:
        # 如果有3个返回值 (img, label, raw_label)
        imgs = torch.cat([x[0] for x in batch], dim=0)
        lbls = torch.cat([x[1] for x in batch], dim=0)
        raw_lbls = torch.cat([x[2] for x in batch], dim=0)
        return imgs, lbls, raw_lbls
    elif num_returns == 2:
        # 如果只有2个返回值 (img, label)，用label作为raw_label
        imgs = torch.cat([x[0] for x in batch], dim=0)
        lbls = torch.cat([x[1] for x in batch], dim=0)
        return imgs, lbls  # 返回lbls两次


class PreprocessorTransformtest:
    def __init__(self, hu_clip=(-1000, 1000), zscore=True, augment=False):
        self.hu_clip = hu_clip
        self.zscore = zscore
        self.augment = augment  # eval 默认 False

    def __call__(self, img: np.ndarray, label):
        # label 兼容 PIL/numpy/torch
        if torch.is_tensor(label):
            label = label.detach().cpu().numpy().astype(np.int64)
        else:
            label = np.array(label, dtype=np.int64)

        if self.hu_clip is not None:
            low, high = -200, 275
            img = np.clip(img, *self.hu_clip)
            img = (img - low) / (high - low)

        if self.zscore:
            mask = img != 0
            if mask.sum() > 0:
                mean = img[mask].mean()
                std = img[mask].std()
                img = (img - mean) / (std + 1e-8)

        # 不裁剪、不resize：保持原始大小（你说的 512）
        H, W = img.shape
        if (H, W) != (512, 512):
            zoom_y = 512 / H
            zoom_x = 512 / W
            img = scipy.ndimage.zoom(img, (zoom_y, zoom_x), order=1, prefilter=False)
            label = scipy.ndimage.zoom(label, (zoom_y, zoom_x), order=0, prefilter=False)
        img = np.stack([img] * 3, axis=0)  # [3,H,W]
        # 强制连续 + 可写
        img = np.ascontiguousarray(img).copy()
        label = np.ascontiguousarray(label).copy()

        img = torch.from_numpy(img).float()
        label = torch.from_numpy(label).long()
        return img, label

def check_label_pixel_values(dataset, num_samples=5):
    """检查原始标签文件的像素值"""
    for i in range(min(num_samples, len(dataset))):
        img_path, label_path = dataset.images[i]  # 直接访问底层数据
        label = Image.open(label_path)
        label_np = np.array(label)

        print(f"样本 {i}: {os.path.basename(img_path)}")
        print(f"  标签唯一值: {np.unique(label_np)}")
        print(f"  标签形状: {label_np.shape}")
        print(f"  像素值范围: [{label_np.min()}, {label_np.max()}]")
        # 简单统计
        for cls in np.unique(label_np):
            print(f"    类别 {cls}: {np.sum(label_np == cls)} 像素")

# 在你的AmosFracSegmentationIncremental类初始化后（或在类内添加检查）

def scan_dataset_label_stats(ds, num_classes=16, ignore=255, max_samples=None, stride=1):
    """
    统计 ds 中 GT mask 的真实出现情况：
    - 每个类累计像素数
    - 每个类出现过的样本数
    """
    pix_cnt = np.zeros(num_classes, dtype=np.int64)
    img_cnt = np.zeros(num_classes, dtype=np.int64)

    N = len(ds)
    if max_samples is None:
        idxs = range(0, N, stride)
    else:
        idxs = range(0, min(N, max_samples), stride)

    for i in idxs:
        x, y,_ = ds[i]               # y 可能是 PIL/np/torch
        if hasattr(y, "cpu"):      # torch tensor
            y = y.cpu().numpy()
        else:
            y = np.array(y)

        # 统计 unique
        u, c = np.unique(y, return_counts=True)

        for uu, cc in zip(u, c):
            if uu == ignore:
                continue
            if 0 <= uu < num_classes:
                pix_cnt[uu] += int(cc)
                img_cnt[uu] += 1

    present = np.where(pix_cnt > 0)[0].tolist()
    return pix_cnt, img_cnt, present


def debug_incremental_mappings(dataset):
    print("=== 增量数据集调试信息 ===")
    print(f"当前任务标签 (labels): {dataset.labels}")
    print(f"旧任务标签 (labels_old): {dataset.labels_old}")
    #print(f"完整顺序 (order): {dataset.order}")
    #print(f"倒序映射 (inverted_order): {dataset.inverted_order}")
    print(f"数据集大小: {len(dataset)}")

    # 检查第一个样本的映射结果
    if len(dataset) > 0:
        img, target,_ = dataset[0]
        if torch.is_tensor(target):
            target_np = target.numpy()
        else:
            target_np = np.array(target)
        print(f"第一个样本的目标标签唯一值: {np.unique(target_np)}")
        print(f"第一个样本的目标标签形状: {target_np.shape}")
        # 统计映射后的类别分布
        for val in np.unique(target_np):
            count = np.sum(target_np == val)
            # 尝试反向查找这个值对应原始什么类别
            orig_cls = [k for k, v in dataset.inverted_order.items() if v == val]
            print(f"  映射值 {val} (可能对应原始类别 {orig_cls}): {count} 像素")

        # 全局统计
        #num_classes = len(dataset.labels) if hasattr(dataset, "labels") else 16
        # pix_cnt, img_cnt, present = scan_dataset_label_stats(dataset, num_classes=17, max_samples=len(dataset))
        #
        # print(f"[GT-STATS] present classes in samples: {present}")
        # for k in present:
        #     print(f"  class {k}: pix={pix_cnt[k]}, imgs={img_cnt[k]}")

# 调用示例
# inc_dataset = RibFracSegmentationIncremental(...) # 用你的参数初始化
# debug_incremental_mappings(inc_dataset)

# rib_root = '/mnt/newdisk/hmt/kt2/ribfrac_try'
# full_set = RibFracSegmentation(rib_root, 'train')
# check_label_pixel_values(full_set)

# def visualize_rib_samples(dataset, num_samples=4, save_dir="./debug_vis_amos"):
#     os.makedirs(save_dir, exist_ok=True)
#
#     # ImageNet 归一化参数
#     mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
#     std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
#
#     # 类别颜色
#     palette = {
#         0:  (0, 0, 0),          # background -> 黑
#         1:  (255, 0, 0),        # displaced -> 红
#         2:  (0, 255, 0),        # non_displaced -> 绿
#         3:  (0, 0, 255),        # buckle -> 蓝
#         4:  (255, 255, 0),      # segmental -> 黄
#         255: (255, 255, 255),   # ignore -> 白
#     }
#
#     # --- 随机选 num_samples 个索引 ---
#     total_len = len(dataset)
#     if total_len == 0:
#         print("[debug] Dataset is empty! 无法可视化")
#         return
#
#     num_samples = min(num_samples, total_len)
#     indices = random.sample(range(total_len), num_samples)
#     print(f"[debug] 随机选取样本索引: {indices}")
#
#     for idx in indices:
#         img, target = dataset[idx]
#
#         # ---- 反归一化图像 ----
#         if torch.is_tensor(img):
#             # 情况1: PyTorch Tensor [C, H, W] 或 [B, C, H, W]
#             img_denorm = img * std + mean
#             img_denorm = img_denorm.clamp(0, 1)
#             img_np = (img_denorm.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
#
#         elif isinstance(img, Image.Image):
#             # 情况2: PIL Image 对象
#             img_np = np.array(img.convert("RGB"))
#
#         elif isinstance(img, np.ndarray):
#             # 情况3: NumPy 数组 (你的数据实际类型)
#             img_np = img.astype(np.float32)  # 确保为浮点数处理
#
#             # 检查并处理数组维度和通道数
#             if img_np.ndim == 2:
#                 # 单通道 [H, W] -> 复制为三通道 [H, W, 3]
#                 img_np = np.stack([img_np, img_np, img_np], axis=-1)
#             elif img_np.ndim == 3 and img_np.shape[-1] == 1:
#                 # 单通道 [H, W, 1] -> 扩展为三通道 [H, W, 3]
#                 img_np = np.concatenate([img_np, img_np, img_np], axis=-1)
#             elif img_np.ndim == 3 and img_np.shape[-1] == 3:
#                 # 已经是三通道 [H, W, 3]，无需处理
#                 pass
#             else:
#                 raise ValueError(f"不支持的NumPy数组形状: {img_np.shape}")
#
#             # 假设你的NumPy数组可能是归一化的浮点数，将其转换到[0, 255]
#             if img_np.max() <= 1.0:
#                 img_np = (img_np * 255).astype(np.uint8)
#             else:
#                 img_np = img_np.astype(np.uint8)
#         else:
#             raise TypeError(f"不支持的图像类型: {type(img)}。支持类型: torch.Tensor, PIL.Image, numpy.ndarray")
#         # ---- 标签 ----
#         target_np = target.numpy() if torch.is_tensor(target) else np.array(target)
#         h, w = target_np.shape
#         color_label = np.zeros((h, w, 3), dtype=np.uint8)
#
#         for cls_id, color in palette.items():
#             mask = (target_np == cls_id)
#             color_label[mask] = color
#
#         # ---- 拼接图像 ----
#         concat = np.concatenate([img_np, color_label], axis=1)
#         concat_img = Image.fromarray(concat)
#
#         save_path = os.path.join(save_dir, f"sample_{idx}.png")
#         concat_img.save(save_path)
#         print(f"[debug] 保存样本 {idx} 到: {save_path}")

def visualize_rib_samples(dataset, num_samples=4, save_dir="./debug_vis_amos"):
    os.makedirs(save_dir, exist_ok=True)

    # -------- 15类调色板：0背景 + 1~15前景 + 255 ignore --------
    palette = {
        0: (0, 0, 0),  # background
        1: (255, 0, 0),
        2: (0, 255, 0),
        3: (0, 0, 255),
        4: (255, 255, 0),
        5: (255, 0, 255),
        6: (0, 255, 255),
        7: (255, 128, 0),
        8: (128, 0, 255),
        9: (0, 128, 255),
        10: (128, 255, 0),
        11: (255, 0, 128),
        12: (0, 255, 128),
        13: (128, 128, 0),
        14: (0, 128, 128),
        15: (128, 0, 128),
        16: (200, 0, 100),
        255: (255, 255, 255),  # ignore
    }

    # --- 随机选 num_samples 个索引 ---
    total_len = len(dataset)
    if total_len == 0:
        print("[debug] Dataset is empty! 无法可视化")
        return

    num_samples = min(num_samples, total_len)
    indices = random.sample(range(total_len), num_samples)
    print(f"[debug] 随机选取样本索引: {indices}")

    def _tensor_to_uint8_rgb_list(img_t: torch.Tensor):
        """
        img_t: [3,H,W] or [N,3,H,W]，数值可能是z-score后的任意范围
        返回: list of uint8 RGB images, each [H,W,3]
        """
        x = img_t.detach().cpu()

        # [3,H,W] -> [1,3,H,W]
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.dim() != 4:
            raise ValueError(f"不支持的Tensor形状: {tuple(x.shape)}，期望 [3,H,W] 或 [N,3,H,W]")

        # 你的CT通常是复制成3通道的，这里取第0通道做灰度显示即可
        g = x[:, 0]  # [N,H,W]

        outs = []
        for i in range(g.size(0)):
            gi = g[i].numpy().astype(np.float32)

            # 稳健拉伸：用1%~99%分位裁剪，避免极端值毁掉显示
            lo, hi = np.percentile(gi, 1), np.percentile(gi, 99)
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = float(np.min(gi)), float(np.max(gi) + 1e-6)

            gi = np.clip(gi, lo, hi)
            gi = (gi - lo) / (hi - lo + 1e-8)  # -> [0,1]
            gi_u8 = (gi * 255.0).astype(np.uint8)
            rgb = np.stack([gi_u8, gi_u8, gi_u8], axis=-1)  # [H,W,3]
            outs.append(rgb)
        return outs

    def _img_to_uint8_rgb_list(img):
        """
        支持 torch.Tensor / PIL / numpy
        返回: list of [H,W,3] uint8
        """
        if torch.is_tensor(img):
            return _tensor_to_uint8_rgb_list(img)

        if isinstance(img, Image.Image):
            return [np.array(img.convert("RGB"))]

        if isinstance(img, np.ndarray):
            img_np = img.astype(np.float32)

            # 兼容 [H,W] / [H,W,1] / [H,W,3]
            if img_np.ndim == 2:
                img_np = np.stack([img_np, img_np, img_np], axis=-1)
            elif img_np.ndim == 3 and img_np.shape[-1] == 1:
                img_np = np.concatenate([img_np, img_np, img_np], axis=-1)
            elif img_np.ndim == 3 and img_np.shape[-1] == 3:
                pass
            else:
                raise ValueError(f"不支持的NumPy数组形状: {img_np.shape}")

            # 如果已经是[0,1]，映射到[0,255]
            if img_np.max() <= 1.0:
                img_np = (img_np * 255.0).astype(np.uint8)
            else:
                img_np = np.clip(img_np, 0, 255).astype(np.uint8)

            return [img_np]

        raise TypeError(f"不支持的图像类型: {type(img)}。支持: torch.Tensor, PIL.Image, numpy.ndarray")

    def _target_to_list(target):
        """
        支持 torch.Tensor / PIL / numpy
        返回: list of [H,W] int64
        """
        if torch.is_tensor(target):
            t = target.detach().cpu().numpy().astype(np.int64)
        else:
            t = np.array(target, dtype=np.int64)

        if t.ndim == 2:
            return [t]
        if t.ndim == 3:
            return [t[i] for i in range(t.shape[0])]
        raise ValueError(f"不支持的target形状: {t.shape}，期望 [H,W] 或 [N,H,W]")

    for idx in indices:
        img, target,_ = dataset[idx]

        img_list = _img_to_uint8_rgb_list(img)  # list of [H,W,3]
        tgt_list = _target_to_list(target)  # list of [H,W]

        n_img, n_tgt = len(img_list), len(tgt_list)
        if n_img != n_tgt:
            n = min(n_img, n_tgt)
            print(f"[warn] idx={idx} patch数不一致: img={n_img}, tgt={n_tgt}，将只保存前 {n} 个")
            img_list = img_list[:n]
            tgt_list = tgt_list[:n]

        for p, (img_np, target_np) in enumerate(zip(img_list, tgt_list)):
            h, w = target_np.shape
            color_label = np.zeros((h, w, 3), dtype=np.uint8)

            for cls_id, color in palette.items():
                mask = (target_np == cls_id)
                if mask.any():
                    color_label[mask] = color

            # 拼接：左图右mask
            concat = np.concatenate([img_np, color_label], axis=1)
            concat_img = Image.fromarray(concat)

            # 文件名：单图 sample_{idx}.png，多patch sample_{idx}_p{p}.png
            save_name = f"sample_{idx}.png" if len(img_list) == 1 else f"sample_{idx}_p{p}.png"
            save_path = os.path.join(save_dir, save_name)
            concat_img.save(save_path)

            uniq = np.unique(target_np)
            print(f"[debug] 保存样本 {idx} patch {p} 到: {save_path} | unique labels: {uniq}")
