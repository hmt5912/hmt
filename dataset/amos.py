
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
            1: "spleen",
            2:"right kidney",
            3:"left kidney",
             4:"gallbladder",
            5:"esophagus",
             6:"liver",
            7:"stomach",
    8:"aorta",
    9:"inferior vena cava",
    10:"pancreas",
    11:"right adrenal gland",
    12:" left adrenal gland",
    13:"duodenum",
    14:" bladder",
    15:"prostate/uterus",
    16:"liver tumor"
}
class_location = {
    1: "left upper quadrant of the abdomen, posterior and lateral to the stomach",
    2: "right retroperitoneum, posterior to the right lobe of the liver",
    3: "left retroperitoneum, posterior to the spleen and pancreatic tail",
    4: "within the gallbladder fossa on the inferior surface of the right hepatic lobe",
    5: "within the mediastinum, anterior to the thoracic spine, traversing the diaphragm to connect with the stomach",
    6: "right upper quadrant and upper abdomen, beneath the diaphragm",
    7: "left upper quadrant, beneath the left hemidiaphragm and liver",
    8: "left anterior to the spine, descending from the diaphragm to the bifurcation",
    9: "right anterior to the spine, posterior to the liver, draining into the right atrium",
    10: "upper abdomen in the retroperitoneum, posterior to the stomach, at the level of L1-L2 vertebrae",
    11: "superomedial aspect of the right kidney, posterior to the inferior vena cava",
    12: "superomedial aspect of the left kidney, posterior to the pancreatic tail",
    13: "upper abdomen, forming a C-loop around the head of the pancreas",
    14: "anterior part of the pelvis, posterior to the pubic symphysis",
    15: "in the pelvis, inferior to the bladder neck and anterior to the rectum/in the central pelvis, posterior to the bladder and anterior to the rectum",  # 子宫位置
    16:"within the liver, typically appearing as a region with lower or heterogeneous CT intensity compared to the surrounding liver parenchyma"
}
class_hu = {
    1: (40, 60),
    2: (30, 50),
    3: (30, 50),
    4: (0, 30),
    5: None,
    6: (40, 70),
    7: (-100, 100),
    8: (35, 50),
    9: (35, 50),
    10: (30, 50),
    11: None,
    12: None,
    13: (-100, 100),
    14: (0, 30),
    15: (40, 60),
    16: None
}

class MultiRootSliceDataset(data.Dataset):
    def __init__(self, roots_map, transform=None, file_list=None, search_sets=("train", "val")):
        self.roots_map = roots_map
        self.transform = transform

        if file_list is None:
            raise ValueError("file_list must be provided (from txt).")
        self.files = [x.strip() for x in file_list if x.strip()]

        self.search_sets = search_sets
        self.key2paths = {}

        for prefix, root in self.roots_map.items():
            root = os.path.expanduser(root)
            for s in search_sets:
                img_dir = os.path.join(root, s, "images")
                lbl_dir = os.path.join(root, s, "labels")
                if not (os.path.isdir(img_dir) and os.path.isdir(lbl_dir)):
                    continue
                for fn in os.listdir(img_dir):
                    if not fn.endswith(".npy"):
                        continue
                    name = os.path.splitext(fn)[0]
                    img_path = os.path.join(img_dir, fn)
                    lbl_path = os.path.join(lbl_dir, name + ".png")
                    if os.path.exists(lbl_path):
                        self.key2paths[(prefix, name)] = (img_path, lbl_path)

        missing = []
        for item in self.files:
            pref, name = self._parse_item(item)
            if pref is not None:
                if (pref, name) not in self.key2paths:
                    missing.append(item)
            else:
                ok = any((p, name) in self.key2paths for p in self.roots_map.keys())
                if not ok:
                    missing.append(item)

        if missing:
            print(f"[WARN] {len(missing)}/{len(self.files)} items not found. Examples: {missing[:10]}")

    @staticmethod
    def _parse_item(item: str):
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

        img = np.load(img_path).astype(np.float32)
        target = Image.open(lbl_path)

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target


def load_indices_from_txt(dataset, txt_path):
    name_to_idx = {}
    for i, (img_path, _) in enumerate(dataset.images):
        name = os.path.splitext(os.path.basename(img_path))[0]
        name_to_idx[name] = i

    with open(txt_path, "r") as f:
        names = [line.strip() for line in f]

    idxs = [name_to_idx[n] for n in names if n in name_to_idx]
    return idxs


class amosFracSegmentation(data.Dataset):
    def __init__(self, root, image_set="train", transform=None, file_list=None,
                 search_sets=("train", "val")):
        self.root = os.path.expanduser(root)
        self.image_set = image_set
        self.transform = transform

        if file_list is None:
            raise ValueError("AMOS2D: file_list must be provided (from txt).")
        self.files = file_list

        self.name2paths = {}
        for s in search_sets:
            img_dir = os.path.join(self.root, s, "images")
            lbl_dir = os.path.join(self.root, s, "labels")
            if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
                continue

            for fn in os.listdir(img_dir):
                if not fn.endswith(".npy"):
                    continue
                name = os.path.splitext(fn)[0]
                img_path = os.path.join(img_dir, fn)
                lbl_path = os.path.join(lbl_dir, name + ".png")
                if os.path.exists(lbl_path):
                    self.name2paths[name] = (img_path, lbl_path)

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
        img = np.load(img_path).astype(np.float32)
        target = Image.open(lbl_path)

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target



class AmosFracSegmentationIncremental(data.Dataset):
    def __init__(
            self,
            root,
            train=True,
            transform=None,
            labels=None,
            labels_old=None,
            step=0,
            task_name="13-1",
            client_id=0,
            split_root="splits_fcl",
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
        self.p_fg = 0.5

        image_set = "train" if train else "val"

        if train:
            split_file = os.path.join(
                root, split_root, "amos_lits_3", task_name, f"step{step}", f"client_{client_id}.txt"
            )
        else:
            split_file = os.path.join(
                root, split_root, "amos_lits_3", task_name, f"step{step}", "val_b_CT.txt"
            )

        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file, "r") as f:
            file_list = [line.strip() for line in f if line.strip()]


        roots_map = {
            "amos": "/mnt/newdisk/hmt/kt2/amos/amos_slices_12.16/image-all-ct",
            "lits": "/mnt/newdisk/LITS/slices",
            "lits_part3": "/mnt/newdisk/LITS/part3/slices",
            "lits_part6": "/mnt/newdisk/LITS/part6/slices",
        }
        # 两个根目录"amos": "/mnt/newdisk/hmt/kt2/amos/amos_slices_12.16/image-all-ct",
        #         "lits": "/mnt/newdisk/LITS/slices_try",
        #"amos": "/mnt/newdisk/hmt/kt2/amos/amos_slice_try",
        #    "lits": "/mnt/newdisk/LITS/slices_try",

        base_ds = MultiRootSliceDataset(
            roots_map=roots_map,
            transform=None,
            file_list=file_list,
            search_sets=("train", "val")
        )
        self.has_foreground = []

        for i in range(len(base_ds)):
            _, lbl = base_ds[i]
            lbl = np.array(lbl)
            self.has_foreground.append((lbl > 0).any())
        print(f"[AMOS][{image_set}] step={step} "
              f"{'client=' + str(client_id) if train else 'global-val'} "
              f"samples={len(base_ds)}")

        labels = list(labels) if labels is not None else []
        labels_old = list(labels_old) if labels_old is not None else []
        while 0 in labels: labels.remove(0)
        while 0 in labels_old: labels_old.remove(0)

        self.labels = [0] + labels
        self.labels_old = [0] + labels_old

        self.labels_cum = []
        for x in (self.labels_old + self.labels):
            if x not in self.labels_cum:
                self.labels_cum.append(x)

        self.inverted_order = {lab: i for i, lab in enumerate(self.labels_cum)}
        self.inverted_order[255] = 255

        allowed_set = set(self.labels_cum) | {0, 255}

        def target_transform(label):
            if torch.is_tensor(label):
                t = label.clone()
                is_tensor = True
            else:
                t = np.array(label, dtype=np.int64)
                is_tensor = False

            out = np.full_like(t, 255)
            for lab in labels:
                out[t == lab] = self.inverted_order[lab]

            #
            if masking and data_masking == "current":
                for lab in labels_old:
                    out[t == lab] = 0
            else:
                for lab in labels_old:
                    out[t == lab] = self.inverted_order[lab]

            out[t == 0] = 0

            out[t == 255] = 255

            if is_tensor:
                return torch.from_numpy(out).long().to(label.device)
            else:
                return out

        indices = list(range(len(base_ds)))
        self.dataset = Subset(base_ds, indices, transform=transform, target_transform=target_transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.train and random.random() < self.p_fg:
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
        self.fg_value_threshold = fg_value_threshold

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

    def _sample_one_fg_patch(self, img, label, size):
        H, W = label.shape
        fg = (label > self.fg_value_threshold) & (label != 255)
        if not fg.any():
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

        cy = int(np.mean(ys))
        cx = int(np.mean(xs))
        y0 = cy - size // 2
        x0 = cx - size // 2
        y0, x0 = self._clamp_window(y0, x0, size, H, W)
        return img[y0:y0+size, x0:x0+size], label[y0:y0+size, x0:x0+size]

    def __call__(self, img: np.ndarray, label: Image.Image):
        label = np.array(label).astype(np.int64)

        if self.hu_clip is not None:
            low, high = -125, 275
            img = np.clip(img, *self.hu_clip)
            img = (img - low) / (high - low)

        if self.zscore:
            mask = img != 0
            if mask.sum() > 0:
                mean = img[mask].mean()
                std = img[mask].std()
                img = (img - mean) / (std + 1e-8)

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

        img, label = self._resize_min_size(img, label, self.out_size)

        img_patches = []
        lbl_patches = []
        for _ in range(self.n_patches):
            p_img, p_lbl = self._sample_one_fg_patch(img, label, self.out_size)
            img_patches.append(p_img)
            lbl_patches.append(p_lbl)

        img_patches = np.stack(img_patches, axis=0)  # [N,H,W]
        img_patches = np.stack([img_patches] * 3, axis=1)  # [N,3,H,W]
        lbl_patches = np.stack(lbl_patches, axis=0)  # [N,H,W]

        img_patches = np.ascontiguousarray(img_patches).copy()
        lbl_patches = np.ascontiguousarray(lbl_patches).copy()

        img_t = torch.from_numpy(img_patches).float()
        lbl_t = torch.from_numpy(lbl_patches).long()
        return img_t, lbl_t

def collate_flatten_patches(batch):

    num_returns = len(batch[0])

    if num_returns == 3:
        imgs = torch.cat([x[0] for x in batch], dim=0)
        lbls = torch.cat([x[1] for x in batch], dim=0)
        raw_lbls = torch.cat([x[2] for x in batch], dim=0)
        return imgs, lbls, raw_lbls
    elif num_returns == 2:
        imgs = torch.cat([x[0] for x in batch], dim=0)
        lbls = torch.cat([x[1] for x in batch], dim=0)
        return imgs, lbls  # 返回lbls两次


class PreprocessorTransformtest:
    def __init__(self, hu_clip=(-1000, 1000), zscore=True, augment=False):
        self.hu_clip = hu_clip
        self.zscore = zscore
        self.augment = augment  # eval 默认 False

    def __call__(self, img: np.ndarray, label):
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

        H, W = img.shape
        if (H, W) != (512, 512):
            zoom_y = 512 / H
            zoom_x = 512 / W
            img = scipy.ndimage.zoom(img, (zoom_y, zoom_x), order=1, prefilter=False)
            label = scipy.ndimage.zoom(label, (zoom_y, zoom_x), order=0, prefilter=False)
        img = np.stack([img] * 3, axis=0)
        img = np.ascontiguousarray(img).copy()
        label = np.ascontiguousarray(label).copy()

        img = torch.from_numpy(img).float()
        label = torch.from_numpy(label).long()
        return img, label


