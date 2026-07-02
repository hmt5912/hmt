
import torch
import torch.nn.functional as F
from typing import List

class TaskMemory:
    def __init__(self):
        self.task_text = {}
        self.task_grad = {}
        self.task_classes = {}

def cosine(a, b):
    return float(F.cosine_similarity(a, b, dim=0))

def _pad_to_same(a: torch.Tensor, b: torch.Tensor):
    na, nb = a.numel(), b.numel()
    if na == nb:
        return a, b
    if na < nb:
        a = torch.cat([a, torch.zeros(nb - na, dtype=a.dtype)], dim=0)
    else:
        b = torch.cat([b, torch.zeros(na - nb, dtype=b.dtype)], dim=0)
    return a, b

def compute_task_relevance(
    step: int,
    task_classes_orig: List[int],
    q_txt: torch.Tensor,
    g_sig: torch.Tensor,
    memory: TaskMemory,
    eta: float = 0.7,
    tau: float = 0.5,
    topk: int = 3
):
    if len(memory.task_text) == 0:
        memory.task_text[step] = q_txt
        memory.task_grad[step] = g_sig
        memory.task_classes[step] = list(task_classes_orig)
        return None

    scores = {}
    for k in memory.task_text.keys():
        r_sem = float(F.cosine_similarity(q_txt, memory.task_text[k], dim=0).item())
        g1, g2 = _pad_to_same(g_sig, memory.task_grad[k])  # 关键
        r_grd = float(F.cosine_similarity(g1, g2, dim=0).item())
        scores[k] = eta * r_sem + (1 - eta) * r_grd

    selected = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:topk]
    w = torch.softmax(torch.tensor([scores[k] for k in selected], dtype=torch.float32) / tau, dim=0)

    R = {}
    for i, k in enumerate(selected):
        wk = float(w[i].item())
        for c in memory.task_classes[k]:
            R[c] = max(R.get(c, 0.0), wk)

    memory.task_text[step] = q_txt
    memory.task_grad[step] = g_sig
    memory.task_classes[step] = list(task_classes_orig)

    return R