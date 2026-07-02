import os

# 在导入torch之前设置
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import os.path as osp
from tkinter import N
from turtle import pd
import colorsys
from FC import FC_model
import torch
torch.backends.cudnn.enabled = False
import random
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
import matplotlib.pyplot as plt
from utils.utils import Label2Color, color_map




from myNetwork import make_model
from myNetwork_rcil import make_model_rcil

from Fed_utils import *
from option import args_parser, modify_command_options
import datetime


from torch import distributed
from torch.utils import data
from torch.utils.data.distributed import DistributedSampler

import task_self1

from dataset.amos import AmosFracSegmentationIncremental,PreprocessorTransform,PreprocessorTransformtest
from metrics import StreamSegMetrics
from modules.clinical_bert import ClinicalTextEncoder,build_text_prior_mat_cont
from utils.TaskMemory import TaskMemory, compute_task_relevance

from rcil_utils import *
import warnings
warnings.filterwarnings(
    "ignore",
    message=r".*inplace_abn_sync is being called, but torch\.distributed is not initialized.*"
)



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
    15: "in the pelvis, inferior to the bladder neck and anterior to the rectum/in the central pelvis, posterior to the bladder and anterior to the rectum",
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

def line_color_from_palette(cls_id: int, pal):
    rgb = pal[int(cls_id)].astype(np.float32) / 255.0
    return tuple(rgb.tolist())

def get_testset(opts, step):

    test_transform = PreprocessorTransformtest(
        hu_clip=None,
        zscore=True,
        augment=False
    )

    labels, labels_old, _ = task_self1.get_task_labels(opts.dataset, opts.task, step)
    labels_cum = labels_old + labels

    if opts.dataset == 'voc':
        dataset = VOCSegmentationIncremental
    elif opts.dataset == 'ade':
        dataset = adeSegmentationIncremental
    elif opts.dataset == 'rib':
        dataset = RibFracSegmentationIncremental
    elif opts.dataset == 'amos':
        dataset = AmosFracSegmentationIncremental
    elif opts.dataset == 'hip':
        dataset = hipFracSegmentationIncremental
    else:
        raise NotImplementedError



    image_set = 'train' if opts.val_on_trainset else 'val'
    test_dst = dataset(
        root=opts.data_root,
        train=opts.val_on_trainset,
        transform=test_transform,
        labels=list(labels_cum),
        disable_background=opts.disable_background,
        test_on_val=opts.test_on_val,
        step=step,
        ignore_test_bg=opts.ignore_test_bg,
        task_name=opts.task,
        split_root="splits_fcl",
    )




    return test_dst, len(labels_cum)

def _to_uint8_tensor(x):
    if x is None:
        return None
    if isinstance(x, torch.ByteTensor):
        return x
    if isinstance(x, torch.Tensor):
        return x.to(dtype=torch.uint8)
    try:
        return torch.tensor(x, dtype=torch.uint8)
    except Exception:
        return None


def _fcmodel_pack_state(c):
    if hasattr(c, "state_dict") and callable(getattr(c, "state_dict")):
        try:
            return c.state_dict()
        except Exception:
            pass

    return {
        "client_index": getattr(c, "client_index", None),
        "learned_step": getattr(c, "learned_step", -1),
        "signal": getattr(c, "signal", False),
        "last_entropy": getattr(c, "last_entropy", -1),
        "cur_epoch": getattr(c, "cur_epoch", 0),
        "last_learning_rate": getattr(c, "last_learning_rate", 0.0),
        "trainer_state": getattr(c, "trainer_state", None),
        "entropy_threshold": getattr(c, "entropy_threshold", None),
    }


def _fcmodel_load_state(c, sd):
    if sd is None:
        return

    if hasattr(c, "load_state_dict") and callable(getattr(c, "load_state_dict")):
        try:
            c.load_state_dict(sd)
            return
        except Exception:
            pass


    if "learned_step" in sd:
        c.learned_step = sd["learned_step"]
    if "signal" in sd:
        c.signal = sd["signal"]
    if "last_entropy" in sd:
        c.last_entropy = sd["last_entropy"]
    if "cur_epoch" in sd:
        c.cur_epoch = sd["cur_epoch"]
    if "last_learning_rate" in sd:
        c.last_learning_rate = sd["last_learning_rate"]
    if "trainer_state" in sd:
        c.trainer_state = sd["trainer_state"]
    if "entropy_threshold" in sd and sd["entropy_threshold"] is not None:
        c.entropy_threshold = sd["entropy_threshold"]


    c.current_trainset = None


def save_fed_ckpt(path, ep_g_next, old_step, model_g, clients, clients_index=None, extra=None):

    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "ep_g_next": int(ep_g_next),
        "old_step": int(old_step),
        "model_g": model_g.state_dict(),
        "clients_state": [_fcmodel_pack_state(c) for c in clients],
        "extra": dict(extra or {}),
        "clients_index": list(clients_index) if clients_index is not None else None,
        "rng_py": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }

    tmp_path = path + ".tmp"
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, path)


def load_fed_ckpt(path, model_g, clients, device, verbose=True):

    ckpt = torch.load(path, map_location=device)
    model_g.load_state_dict(ckpt["model_g"], strict=False)

    for c, st in zip(clients, ckpt.get("clients_state", [])):
        _fcmodel_load_state(c, st)

    def _vprint(msg):
        if verbose:
            print(msg)

    try:
        if ckpt.get("rng_py") is not None:
            random.setstate(ckpt["rng_py"])
    except Exception as e:
        _vprint(f"[Resume] skip python RNG restore: {e}")

    try:
        if ckpt.get("rng_numpy") is not None:
            np.random.set_state(ckpt["rng_numpy"])
    except Exception as e:
        _vprint(f"[Resume] skip numpy RNG restore: {e}")

    try:
        rng_t = _to_uint8_tensor(ckpt.get("rng_torch"))
        if rng_t is not None:
            torch.set_rng_state(rng_t)
        else:
            _vprint("[Resume] skip torch RNG restore: incompatible type")
    except Exception as e:
        _vprint(f"[Resume] skip torch RNG restore: {e}")

    try:
        if torch.cuda.is_available() and ckpt.get("rng_cuda") is not None:
            fixed = []
            for s in ckpt["rng_cuda"]:
                t = _to_uint8_tensor(s)
                if t is None:
                    fixed = None
                    break
                fixed.append(t)
            if fixed is not None:
                torch.cuda.set_rng_state_all(fixed)
            else:
                _vprint("[Resume] skip cuda RNG restore: incompatible type")
    except Exception as e:
        _vprint(f"[Resume] skip cuda RNG restore: {e}")

    start_ep_g = int(ckpt.get("ep_g_next", 0))
    old_step = int(ckpt.get("old_step", -1))
    extra = ckpt.get("extra", {}) or {}
    resume_clients_index = ckpt.get("clients_index", None)

    return start_ep_g, old_step, extra, resume_clients_index

def record_head_adr(
    adr_vec,
    step_s: int,
    ep_g: int,
    args,
    classes_name_map: dict,
    adr_hist: dict,
    writer=None,
    prefix="oada/ADR",
):
    labels, labels_old, _ = task_self1.get_task_labels(args.dataset, args.task, step_s)
    Cout = len(adr_vec)

    if Cout == len(labels):
        ch2cls = labels
    elif Cout == len(labels) + 1:
        ch2cls = [0] + labels
    else:
        ch2cls = list(range(Cout))

    added = 0
    for i, adr in enumerate(adr_vec):
        cls_id = ch2cls[i] if i < len(ch2cls) else i
        if cls_id in (255,):
            continue

        organ_name = classes_name_map.get(cls_id, f"class{cls_id}")
        adr_hist[(cls_id, organ_name)].append((ep_g, float(adr)))
        added += 1

        if writer is not None and cls_id not in (0, 255):
            writer.add_scalar(f"{prefix}/upto_step{args.pending_step}/{cls_id:02d}_{organ_name}", float(adr), ep_g)

    return added



def color_from_cls_id(cls_id: int):
    hue = (cls_id * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.95, 0.95)
    return (r, g, b)

def main(args):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        distributed.init_process_group(backend='nccl', init_method='env://')
        rank = distributed.get_rank()
        world_size = distributed.get_world_size()

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device_id = local_rank
        torch.cuda.set_device(device_id)
        print(f"[Distributed] rank = {rank}, world_size = {world_size}, local_rank = {local_rank}")
    else:
        print("Single process / single GPU: skip init_process_group()")
        rank = 0
        world_size = 1
        device_id = 0
        torch.cuda.set_device(device_id)

    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    writer = None
    if rank == 0:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        writer = SummaryWriter(log_dir=f"{args.log_dir}/{args.log_name}_{timestamp}")


    setup_seed(args.seed)

    args.inital_nb_classes = task_self1.get_per_task_classes(args.dataset, args.task, step=0)[0]

    if args.name != 'RCIL':
        model_g = make_model(args, classes=task_self1.get_per_task_classes(args.dataset, args.task, step=0))
    else:
        model_g = make_model_rcil(args, classes=task_self1.get_per_task_classes(args.dataset, args.task, step=0))

    if args.fix_bn:
        model_g.fix_bn()

    num_clients = args.num_clients
    models = []
    for client_index in range(3):
        model_temp = FC_model(client_index, args.batch_size, args.num_workers, args.loss_de, args.pod, world_size, rank,
                              device, args.entropy_threshold)
        models.append(model_temp)

    resume_path = os.path.join(args.resume, "last_fed.pt")
    start_ep_g = 0
    old_step = -1

    args.force_step_setup = False
    args.resume_clients_index = None
    adr_hist = defaultdict(list) if rank == 0 else None
    pal = color_map("amos") if rank == 0 else None

    if getattr(args, "reset", False):
        if rank == 0:
            print("[Reset] ignore last_fed.pt and start from scratch")
    elif getattr(args, "resume1", False) and os.path.exists(resume_path):
        start_ep_g, old_step, extra, resume_clients_index = load_fed_ckpt(
            resume_path, model_g, models, device, verbose=(rank == 0)
        )
        args.adr_ema_state = extra.get("adr_ema_state", {})
        args.force_step_setup = True
        args.resume_clients_index = resume_clients_index
        if old_step >= 0:
            old_step = old_step - 1
            if rank == 0:
                print(f"[Resume-Fix] shift old_step -> {old_step} to force head expansion")

        if rank == 0:
            print(f"[Resume] loaded {resume_path} | start_ep_g={start_ep_g}, resume_step={old_step + 1}, old_step={old_step}")
        
    else:
        if rank == 0:
            print("[Start] no resume (default start from scratch)")
    try:
        for ep_g in range(start_ep_g, args.epochs_global):

            current_step = ep_g // args.steps_global

            need_step_setup = (current_step != old_step) or getattr(args, "force_step_setup", False)
            if need_step_setup:
                args.force_step_setup = False

                if not hasattr(args, "text_encoder"):
                    args.text_encoder = ClinicalTextEncoder(device=str(device))
                    args.task_memory = TaskMemory()
                    args.class_text_emb = args.text_encoder.encode_class_embeddings(
                        classes,
                        class_location=class_location,
                        class_hu=class_hu,
                        cache_name="amos_16class_text_emb.pt"
                    )

                labels, labels_old, _ = task_self1.get_task_labels(args.dataset, args.task, current_step)
                labels_cum = labels_old + labels
                task_classes_orig = [c for c in (labels_old + labels) if c not in (0, 255)]

                args.pending_step = current_step
                args.pending_task_classes_orig = task_classes_orig
                args.pending_q_txt = ClinicalTextEncoder.task_embedding(args.class_text_emb, task_classes_orig)

                args.relevance_done_step = -1
                args.class_relevance_cont = None

                test_dst, n_classes = get_testset(args, current_step)
                if distributed.is_available() and distributed.is_initialized():
                    test_loader = data.DataLoader(
                        test_dst,
                        batch_size=args.batch_size if args.crop_val else 1,
                        sampler=DistributedSampler(test_dst, num_replicas=world_size, rank=rank),
                        num_workers=args.num_workers
                    )
                else:
                    test_loader = data.DataLoader(
                        test_dst,
                        batch_size=args.batch_size if args.crop_val else 1,
                        num_workers=args.num_workers
                    )

                inc_ds = test_loader.dataset
                old_classes_cont = len([x for x in labels_old if x not in (0, 255)]) + 1
                args.text_prior_mat_cont = build_text_prior_mat_cont(
                    class_text_emb_orig=args.class_text_emb,
                    labels_cum=labels_cum,
                    old_classes_cont=old_classes_cont,
                    device=device,
                    txt_temp=getattr(args, "txt_prior_tau", 0.07),
                )
                val_metrics = StreamSegMetrics(n_classes)

            if current_step != old_step and old_step != -1:
                args.base_weights = False

                for i in range(num_clients):
                    models[i].last_entropy = -1

                if args.name != 'RCIL':
                    model_g1 = make_model(args, classes=task_self1.get_per_task_classes(args.dataset, args.task,
                                                                                        current_step))
                    model_g1.load_state_dict(model_g.state_dict(), strict=False)
                    if args.init_balanced:
                        model_g1.init_new_classifier(device)
                    model_g = model_g1
                else:
                    model_g1 = make_model_rcil(args, classes=task_self1.get_per_task_classes(args.dataset, args.task,
                                                                                             current_step))

                    for name, mm in model_g1.named_modules():
                        if hasattr(mm, 'convs'):
                            mm.convs.conv2.bias = nn.Parameter(
                                torch.zeros(mm.convs.conv2.weight.shape[0]).to(mm.convs.conv2.weight.device))
                        if hasattr(mm, 'map_convs'):
                            for kk in range(4):
                                mm.map_convs[kk].bias = nn.Parameter(
                                    torch.zeros(mm.map_convs[kk].weight.shape[0]).to(mm.map_convs[kk].weight.device))

                    model_g1.load_state_dict(model_g.state_dict(), strict=False)

                    if args.init_balanced:
                        model_g1.init_new_classifier(device)
                    model_g = model_g1

                    model_g = convert_model(model_g, None)

            if rank == 0:
                print('federated global round: {}, step: {}'.format(ep_g, current_step))

            w_local = []
            if getattr(args, "resume_clients_index", None) is not None:
                clients_index = args.resume_clients_index
                args.resume_clients_index = None
                if rank == 0:
                    print(f"[Resume] use saved clients_index: {clients_index}")
            else:
                clients_index = random.sample(range(num_clients), args.local_clients)

            if rank == 0:
                print('select part of clients to conduct local training')
                print(clients_index)

            for c in clients_index:
                local_model = local_train(args, models, c, model_g, current_step, ep_g,
                                          writer=writer if rank == 0 else None)
                w_local.append(local_model)

            if rank == 0:
                print('federated aggregation...')

            if not hasattr(args, "adr_ema_state"):
                args.adr_ema_state = {}

            if args.base_weights == False:
                w_prev = model_g.state_dict()

                w_g_new, args.adr_ema_state, oada_log = FedAvg_OADA(
                    models=w_local,
                    w_global_prev=w_prev,
                    adr_ema_state=args.adr_ema_state,
                    rho=getattr(args, "oada_rho", 0.95),
                    gamma=getattr(args, "oada_gamma", 0.5),
                    beta_min=getattr(args, "oada_beta_min", 1.0),
                    beta_max=getattr(args, "oada_beta_max", 3.0),
                    server_lr=getattr(args, "oada_server_lr", 1.0),
                    only_keys_prefix=("cls.",),
                    use_bias=True,
                )

                model_g.load_state_dict(w_g_new, strict=False)

                if rank == 0 and oada_log["adr_mean"] is not None:
                    print(
                        f"[OADA] ADR min/mean/max = {oada_log['adr_min']:.4f}/{oada_log['adr_mean']:.4f}/{oada_log['adr_max']:.4f} | "
                        f"beta min/mean/max = {oada_log['beta_min']:.3f}/{oada_log['beta_mean']:.3f}/{oada_log['beta_max']:.3f}")

                val_score = model_global_eval(
                    args, model_g, test_loader, current_step, val_metrics, device, rank,
                    writer=writer if rank == 0 else None
                )
            else:
                if ((ep_g + 1) % args.steps_global) == 0:
                    base_ckpt_path = f"{args.checkpoint}/{args.dataset}_{args.task}_base_step_0.pth"
                    w_g_new = torch.load(base_ckpt_path)
                    model_g.load_state_dict(w_g_new)
                    val_score = model_global_eval(args, model_g, test_loader, current_step, val_metrics, device, rank,
                                                  writer)

            if rank == 0:
                if ((ep_g + 1) % args.steps_global) == 0:
                    with open(f"{args.results_path}/{args.date}_{args.dataset}_{args.task}_{args.name}.csv", "a+") as f:
                        classes_iou = ','.join(
                            [str(val_score['Class IoU'].get(c, 'x')) for c in range(args.num_classes)])
                        f.write(f"{current_step},{classes_iou},{val_score['Mean IoU']}\n")

                    torch.save(model_g.state_dict(),
                               f"{args.checkpoint}/{args.dataset}_{args.task}_{args.name}_step_{current_step}.pth")
                    print(
                        f"{current_step}权重已经保存在{args.checkpoint}/{args.dataset}_{args.task}_{args.name}_step_{current_step}.pth")

                    if current_step == 0 and args.name != "RCIL" and args.base_weights == False:
                        torch.save(model_g.state_dict(),
                                   f"{args.checkpoint}/{args.dataset}_{args.task}_base_step_{current_step}.pth")
                        print(
                            f"{current_step}权重已经保存在{args.checkpoint}/{args.dataset}_{args.task}_base_step_{current_step}.pth")

            if rank == 0:
                global_step = ep_g
                writer.add_scalar("val/MeanIoU", float(val_score["Mean IoU"]), global_step)
                writer.add_scalar("val/MeanDice", float(val_score["Mean Dice"]), global_step)
                writer.add_scalar("val/MeanASD", float(val_score["Mean ASD"]), global_step)
                writer.add_scalar("val/MeanHD95", float(val_score["Mean HD95"]), global_step)
                writer.add_scalar("val/OverallAcc", float(val_score["Overall Acc"]), global_step)
                writer.add_scalar("val/MeanAcc", float(val_score["Mean Acc"]), global_step)
                writer.add_scalar("val/FreqWAcc", float(val_score["FreqW Acc"]), global_step)
                writer.add_scalar("val/TotalSamples", float(val_score["Total samples"]), global_step)

                for cls_id, iou in val_score["Class IoU"].items():
                    if isinstance(iou, str):
                        continue
                    writer.add_scalar(f"val/ClassIoU/{cls_id:03d}", float(iou), global_step)

                class_iou_scalars = {f"{k:03d}": float(v) for k, v in val_score["Class IoU"].items() if
                                     not isinstance(v, str)}
                writer.add_scalars("val/ClassIoU_all", class_iou_scalars, global_step)

                for cls_id, hd95 in val_score["Class HD95"].items():
                    if isinstance(hd95, str):
                        continue
                    writer.add_scalar(f"val/ClassHD95/{cls_id:03d}", float(hd95), global_step)

                class_hd95_scalars = {
                    f"{k:03d}": float(v)
                    for k, v in val_score["Class HD95"].items()
                    if not isinstance(v, str)
                }
                writer.add_scalars("val/ClassHD95_all", class_hd95_scalars, global_step)

            old_step = current_step

            if rank == 0:
                save_fed_ckpt(
                    resume_path,
                    ep_g_next=ep_g + 1,
                    old_step=old_step,
                    model_g=model_g,
                    clients=models,
                    extra={"current_step": current_step,
                           "adr_ema_state": args.adr_ema_state,},
                    clients_index=clients_index,
                )

    except KeyboardInterrupt:
        if rank == 0:
            save_fed_ckpt(
                resume_path.replace("last_fed.pt", "interrupt_fed.pt"),
                ep_g_next=ep_g,
                old_step=old_step,
                model_g=model_g,
                clients=models,
                clients_index=clients_index,
                extra={"note": "interrupted"},
            )
            print("[Interrupted] saved interrupt_fed.pt")
        raise

if __name__ == '__main__':

    args = args_parser()
    args = modify_command_options(args)

    args.results_path = f"results/seed_{args.seed}"
    args.checkpoint = f"{args.checkpoint}/seed_{args.seed}"

    if args.overlap:
        args.results_path += "-ov"
        args.checkpoint += "-ov"

    os.makedirs(args.results_path, exist_ok=True)
    os.makedirs(args.checkpoint, exist_ok=True)

    main(args)