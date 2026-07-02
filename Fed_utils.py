from tkinter import N
import torch.nn as nn
import torch
import copy
from torchvision import transforms
import numpy as np
from torch.nn import functional as F
from PIL import Image
import torch.optim as optim
from myNetwork import *
from torch.utils.data import DataLoader
import random

from train import Trainer
from train_rcil import Trainer_rcil

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from FC import *
import task_self1
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings(
    "ignore",
    message=r".*inplace_abn_sync is being called, but torch\.distributed is not initialized.*"
)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def local_train(args, clients, index, model_g, current_step, ep_g,writer):
    clients[index].beforeTrain(args, current_step)
    if args.base_weights == False:
        if args.use_entropy_detection == True:
            clients[index].update_entropy_signal(model_g)
        local_model = clients[index].train(args, model_g, ep_g,writer)
    else:
        local_model = None

    return local_model


def FedAvg(models):
    w_avg = copy.deepcopy(models[0])
    for k in w_avg.keys():
        for i in range(1, len(models)):
            w_avg[k] += models[i][k]
        w_avg[k] = torch.div(w_avg[k], len(models))
    return w_avg


def model_global_eval(args, model_g, test_loader, current_step, val_metrics, device, rank,writer):
    tmp_model_g = copy.deepcopy(model_g)

    if distributed.is_available() and distributed.is_initialized():
        tmp_model_g = DDP(tmp_model_g.cuda(device),
                          device_ids=[device])  # self.device 应该是 local_rank 或 torch.device
    else:
        tmp_model_g = tmp_model_g.to(device)
    # tmp_model_g = DDP(tmp_model_g.cuda(device), device_ids=[device])

    if args.incremental_method != 'RCIL':
        trainer = Trainer(tmp_model_g, None, device=device, rank=rank, opts=args,classes=task_self1.get_per_task_classes(args.dataset, args.task, current_step), step=current_step,writer = writer)
    else:
        trainer = Trainer_rcil(tmp_model_g, None, device=device, rank=rank, opts=args, step=current_step)

    tmp_model_g.eval()

    _, val_score,_= trainer.validate(
        loader=test_loader, metrics=val_metrics, end_task=True
    )

    if rank == 0:
        print(val_metrics.to_str(val_score))
        # 打印出结果

    tmp_model_g = tmp_model_g.to('cpu')
    torch.cuda.empty_cache()

    del tmp_model_g
    del trainer

    return val_score

def _ensure_tensor(x, device=None):
    if torch.is_tensor(x):
        return x.to(device) if device is not None else x
    return torch.tensor(x, device=device)

@torch.no_grad()
def FedAvg_OADA(
    models,
    w_global_prev,
    adr_ema_state=None,
    rho=0.95,
    gamma=1.0,
    beta_min=1.0,
    beta_max=3.0,
    server_lr=1.0,
    eps=1e-12,
    only_keys_prefix=("cls.", "module.cls."),
    use_bias=True,
):
    if adr_ema_state is None:
        adr_ema_state = {}

    K = len(models)
    w_new = copy.deepcopy(models[0])

    for k in w_new.keys():
        for i in range(1, K):
            w_new[k] += models[i][k]
        w_new[k] = w_new[k] / K

    log_info = {
        "oada_keys": [],
        "adr_min": None,
        "adr_mean": None,
        "adr_max": None,
        "beta_min": None,
        "beta_mean": None,
        "beta_max": None,

        "adr_per_key": {},
        "beta_per_key": {},
    }

    all_adr_vals = []
    all_beta_vals = []

    def _is_target_key(key: str):
        if not key.startswith(only_keys_prefix):
            return False
        if key.endswith(".weight") or (use_bias and key.endswith(".bias")):
            return True
        return False

    for key in w_new.keys():
        if not _is_target_key(key):
            continue

        Wg_prev = w_global_prev[key]
        deltas = []
        for i in range(K):
            deltas.append(models[i][key] - Wg_prev)
        deltas = torch.stack(deltas, dim=0)
        if deltas.dim() == 5:
            d_flat = deltas[..., 0, 0]
            num = torch.norm(d_flat.sum(dim=0), dim=1)                 # [Cout]
            den = torch.norm(d_flat, dim=2).sum(dim=0) + eps           # [Cout]
            adr = num / den                                            # [Cout]
        elif deltas.dim() == 2:
            d_flat = deltas
            num = torch.abs(d_flat.sum(dim=0))                         # [Cout]
            den = torch.abs(d_flat).sum(dim=0) + eps                   # [Cout]
            adr = num / den
        else:
            continue

        dev = Wg_prev.device
        dtype = Wg_prev.dtype
        adr = adr.to(device=dev, dtype=dtype)

        adr_ema = adr_ema_state.get(key, None)
        if torch.is_tensor(adr_ema):
            adr_ema = adr_ema.to(device=dev, dtype=dtype)

        if (adr_ema is None) or (adr_ema.numel() != adr.numel()):
            adr_ema = adr.detach().clone()
        else:
            adr_ema = rho * adr_ema + (1 - rho) * adr

        adr_ema_state[key] = adr_ema.detach().clone()

        # beta
        beta = (1.0 / (adr_ema + eps)).pow(gamma)
        beta = beta.clamp(min=beta_min, max=beta_max)

        delta_g = deltas.mean(dim=0)

        if deltas.dim() == 5:
            beta_view = beta.view(-1, 1, 1, 1)  # [Cout,1,1,1]
            Wg_new = Wg_prev + server_lr * beta_view * delta_g
        else:
            Wg_new = Wg_prev + server_lr * beta * delta_g

        w_new[key] = Wg_new

        log_info["oada_keys"].append(key)
        all_adr_vals.append(adr.detach().cpu())
        all_beta_vals.append(beta.detach().cpu())

        if key.endswith(".weight"):
            log_info["adr_per_key"][key] = adr.detach().float().cpu().tolist()
            log_info["beta_per_key"][key] = beta.detach().float().cpu().tolist()

    if len(all_adr_vals) > 0:
        adr_cat = torch.cat(all_adr_vals)
        beta_cat = torch.cat(all_beta_vals)
        log_info["adr_min"] = float(adr_cat.min())
        log_info["adr_mean"] = float(adr_cat.mean())
        log_info["adr_max"] = float(adr_cat.max())
        log_info["beta_min"] = float(beta_cat.min())
        log_info["beta_mean"] = float(beta_cat.mean())
        log_info["beta_max"] = float(beta_cat.max())

    return w_new, adr_ema_state, log_info
