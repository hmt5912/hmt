import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, binary_erosion

matplotlib.use('Agg')


class _StreamMetrics(object):

    def __init__(self):
        """ Overridden by subclasses """
        pass

    def update(self, gt, pred):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def to_str(self, metrics):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def synch(self, device):
        """ Overridden by subclasses """
        raise NotImplementedError()


class StreamSegMetrics(_StreamMetrics):

    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        self.total_samples = 0
        self.hd95_list = [[] for _ in range(n_classes)]
        self.asd_list = [[] for _ in range(n_classes)]

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            lt_np = self._to_numpy(lt)
            lp_np = self._to_numpy(lp)

            self.confusion_matrix += self._fast_hist(lt_np.flatten(), lp_np.flatten())

            self._update_surface_metrics(lt_np, lp_np)

        self.total_samples += len(label_trues)

    def to_str(self, results):
        string = "\n"

        for k, v in results.items():
            if k in ["Class IoU", "Class Acc", "Class Dice", "Class HD95", "Class ASD", "Confusion Matrix"]:
                continue

            if isinstance(v, (int, np.integer)):
                string += "%s: %d\n" % (k, v)
            else:
                string += "%s: %f\n" % (k, float(v))

        string += 'Class IoU:\n'
        for k, v in results['Class IoU'].items():
            string += "\tclass %d: %s\n" % (k, str(v))

        string += 'Class Acc:\n'
        for k, v in results['Class Acc'].items():
            string += "\tclass %d: %s\n" % (k, str(v))

        if "Class Dice" in results:
            string += 'Class Dice:\n'
            for k, v in results['Class Dice'].items():
                string += "\tclass %d: %s\n" % (k, str(v))

        if "Class HD95" in results:
            string += 'Class HD95:\n'
            for k, v in results['Class HD95'].items():
                string += "\tclass %d: %s\n" % (k, str(v))

        if "Class ASD" in results:
            string += 'Class ASD:\n'
            for k, v in results['Class ASD'].items():
                string += "\tclass %d: %s\n" % (k, str(v))

        return string

    def _to_numpy(self, x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _surface_distances(self, result, reference):
        result = result.astype(bool)
        reference = reference.astype(bool)

        result_border = result ^ binary_erosion(result)
        reference_border = reference ^ binary_erosion(reference)

        dt = distance_transform_edt(~reference_border)
        distances = dt[result_border]

        return distances

    def _surface_metrics_binary(self, pred, gt):
        pred = pred.astype(bool)
        gt = gt.astype(bool)

        pred_sum = pred.sum()
        gt_sum = gt.sum()

        if pred_sum == 0 and gt_sum == 0:
            return None, None

        if gt_sum > 0 and pred_sum == 0:
            penalty = np.sqrt(np.sum(np.array(gt.shape) ** 2))
            return penalty, penalty

        if gt_sum == 0 and pred_sum > 0:
            return None, None

        d1 = self._surface_distances(pred, gt)
        d2 = self._surface_distances(gt, pred)

        all_distances = np.concatenate([d1, d2])

        if all_distances.size == 0:
            return 0.0, 0.0

        hd95 = np.percentile(all_distances, 95)
        asd = np.mean(all_distances)

        return hd95, asd

    def _update_surface_metrics(self, label_true, label_pred):
        for cls in range(self.n_classes):
            gt_cls = (label_true == cls)
            pred_cls = (label_pred == cls)

            hd95, asd = self._surface_metrics_binary(pred_cls, gt_cls)

            if hd95 is not None:
                self.hd95_list[cls].append(hd95)

            if asd is not None:
                self.asd_list[cls].append(asd)

    def _fast_hist(self, label_true, label_pred):
        n = self.n_classes
        lt = np.asarray(label_true).astype(np.int64).ravel()
        lp = np.asarray(label_pred).astype(np.int64).ravel()
        if (lp.max() >= n) or (lp.min() < 0):
            print(f"[WARN] pred out of range: min={lp.min()} max={lp.max()} n={n}")

        mask = (lt >= 0) & (lt < n) & (lp >= 0) & (lp < n)

        idx = n * lt[mask] + lp[mask]
        hist = np.bincount(idx, minlength=n * n).reshape(n, n)
        return hist

    def get_results(self):
        EPS = 1e-6
        hist = self.confusion_matrix

        gt_sum = hist.sum(axis=1)
        pred_sum = hist.sum(axis=0)
        mask = (gt_sum != 0)
        diag = np.diag(hist)

        acc = diag.sum() / hist.sum()
        acc_cls_c = diag / (gt_sum + EPS)
        acc_cls = np.mean(acc_cls_c[mask])
        iu = diag / (gt_sum + hist.sum(axis=0) - diag + EPS)
        mean_iu = np.mean(iu[mask])
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        dice = (2.0 * diag) / (gt_sum + pred_sum + EPS)
        mean_dice = np.mean(dice[mask])

        cls_hd95 = {}
        hd95_values = []

        for i in range(self.n_classes):
            if len(self.hd95_list[i]) > 0:
                cls_mean_hd95 = np.mean(self.hd95_list[i])
                cls_hd95[i] = cls_mean_hd95
                hd95_values.append(cls_mean_hd95)
            else:
                cls_hd95[i] = "X"

        if len(hd95_values) > 0:
            mean_hd95 = np.mean(hd95_values)
        else:
            mean_hd95 = np.nan

        cls_asd = {}
        asd_values = []

        for i in range(self.n_classes):
            if len(self.asd_list[i]) > 0:
                cls_mean_asd = np.mean(self.asd_list[i])
                cls_asd[i] = cls_mean_asd
                asd_values.append(cls_mean_asd)
            else:
                cls_asd[i] = "X"

        if len(asd_values) > 0:
            mean_asd = np.mean(asd_values)
        else:
            mean_asd = np.nan

        cls_iu = dict(zip(range(self.n_classes), [iu[i] if m else "X" for i, m in enumerate(mask)]))
        cls_acc = dict(
            zip(range(self.n_classes), [acc_cls_c[i] if m else "X" for i, m in enumerate(mask)])
        )
        cls_dice = dict(zip(range(self.n_classes), [dice[i] if m else "X" for i, m in enumerate(mask)]))

        return {
            "Total samples": self.total_samples,
            "Overall Acc": acc,  
            "Mean Acc": acc_cls,
            "FreqW Acc": fwavacc,
            "Mean Dice": mean_dice,
            "Mean IoU": mean_iu,
            "Class IoU": cls_iu,
            "Class Acc": cls_acc,
            "Class Dice": cls_dice,
            "Mean HD95": mean_hd95,
            "Class HD95": cls_hd95,
            "Mean ASD": mean_asd,
            "Class ASD": cls_asd,
            "Confusion Matrix": self.confusion_matrix_to_fig()
        }

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        self.total_samples = 0
        self.hd95_list = [[] for _ in range(self.n_classes)]
        self.asd_list = [[] for _ in range(self.n_classes)]

    def synch(self, device):
        # collect from multi-processes
        confusion_matrix = torch.tensor(self.confusion_matrix).to(device)
        samples = torch.tensor(self.total_samples).to(device)

        if torch.distributed.is_initialized():
            # 只在分布式环境下执行reduce
            torch.distributed.reduce(confusion_matrix, dst=0)
            torch.distributed.reduce(samples, dst=0)

            # 只在rank 0上处理结果
            if torch.distributed.get_rank() == 0:
                self.confusion_matrix = confusion_matrix.cpu().numpy()
                self.total_samples = samples.cpu().numpy()
                print("[Dist Rank 0] Reduced confusion matrix and samples")
        else:
            # 单进程模式：直接使用原始值
            self.confusion_matrix = confusion_matrix.cpu().numpy()
            self.total_samples = samples.cpu().numpy()
            print("[Single Process] Using local confusion matrix and samples")

    def confusion_matrix_to_fig(self):
        cm = self.confusion_matrix.astype('float') / (self.confusion_matrix.sum(axis=1) +
                                                      0.000001)[:, np.newaxis]
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        ax.set(title=f'Confusion Matrix', ylabel='True label', xlabel='Predicted label')

        fig.tight_layout()
        return fig


class AverageMeter(object):
    """Computes average values"""

    def __init__(self):
        self.book = dict()

    def reset_all(self):
        self.book.clear()

    def reset(self, id):
        item = self.book.get(id, None)
        if item is not None:
            item[0] = 0
            item[1] = 0

    def update(self, id, val):
        record = self.book.get(id, None)
        if record is None:
            self.book[id] = [val, 1]
        else:
            record[0] += val
            record[1] += 1

    def get_results(self, id):
        record = self.book.get(id, None)
        assert record is not None
        return record[0] / record[1]
