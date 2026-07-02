
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List

def build_prompt(organ: str, location = None, hu_range = None) -> str:
    s = f"The {organ}"
    if location:
        s += f" located in the {location}"
    if hu_range:
        lo, hi = hu_range
        s += f" typically has CT HU values between {lo} and {hi}"
    s += "."
    return s

class ClinicalTextEncoder:
    def __init__(
        self,
        model_name: str = "/mnt/newdisk/hmt/kt2-12.4/FISS-self-linux/ClinicalBERT",
        device: str = "cuda",
        cache_dir: str = "text_cache"
    ):
        self.device = device
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, text: str) -> torch.Tensor:
        tok = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        out = self.model(**tok).last_hidden_state[:, 0]
        out = F.normalize(out, dim=1)
        return out.squeeze(0).cpu()

    def encode_class_embeddings(
        self,
        classes_id2name: dict,
        class_location= None,
        class_hu = None,
        cache_name: str = "amos_class_text_emb.pt",
        ignore_ids: set = {0, 255},
    ) -> dict:
        cache_path = os.path.join(self.cache_dir, cache_name)
        if os.path.exists(cache_path):
            return torch.load(cache_path, map_location="cpu")

        class_location = class_location or {}
        class_hu = class_hu or {}

        emb = {}
        for cid, name in classes_id2name.items():
            if cid in ignore_ids:
                continue
            prompt = build_prompt(name, class_location.get(cid), class_hu.get(cid))
            emb[cid] = self.encode(prompt)

        torch.save(emb, cache_path)
        return emb

    @staticmethod
    def task_embedding(class_emb: dict, class_ids: List[int]) -> torch.Tensor:
        vecs = [class_emb[c] for c in class_ids if c in class_emb]
        if len(vecs) == 0:
            raise ValueError("No class embeddings found for task classes.")
        m = torch.stack(vecs, 0).mean(0)
        return F.normalize(m, dim=0)

@torch.no_grad()
def build_text_prior_mat_cont(
    class_text_emb_orig,
    labels_cum,
    old_classes_cont,
    device,
    txt_temp=0.5,
    eps=1e-8,
    background_id=0,
):
    labels_cum = list(labels_cum)
    C_cont = len(labels_cum)
    assert C_cont > 0, "labels_cum is empty"

    if isinstance(class_text_emb_orig, dict):
        if background_id not in class_text_emb_orig and str(background_id) in class_text_emb_orig:
            new_dict = {}
            for k, v in class_text_emb_orig.items():
                try:
                    new_dict[int(k)] = v
                except Exception:
                    new_dict[k] = v
            class_text_emb_orig = new_dict

    has_bg = (background_id in labels_cum)
    labels_wo_bg = [x for x in labels_cum if x != background_id] if has_bg else labels_cum
    C_eff = len(labels_wo_bg)

    if C_eff == 0:
        prior_full = torch.zeros((C_cont, C_cont), device=device, dtype=torch.float32)
        prior_full[0, 0] = 1.0
        return prior_full

    if isinstance(class_text_emb_orig, dict):
        emb_list = []
        for oid in labels_wo_bg:
            if oid not in class_text_emb_orig:
                raise KeyError(
                    f"orig id {oid} not in class_text_emb_orig "
                    f"(available keys sample: {list(class_text_emb_orig.keys())[:10]})"
                )
            emb_list.append(class_text_emb_orig[oid])
        E = torch.stack(emb_list, dim=0)
    else:
        idx = torch.tensor(labels_wo_bg, device=class_text_emb_orig.device, dtype=torch.long)
        E = class_text_emb_orig[idx]

    E = E.to(device=device, dtype=torch.float32)
    E = F.normalize(E, dim=1)

    sim = (E @ E.t()) / max(float(txt_temp), eps)
    prior_eff = F.softmax(sim, dim=1)
    if not has_bg:
        return prior_eff

    prior_full = torch.zeros((C_cont, C_cont), device=device, dtype=prior_eff.dtype)

    bg_pos = labels_cum.index(background_id)
    map_pos = [i for i, oid in enumerate(labels_cum) if oid != background_id]
    mp = torch.tensor(map_pos, device=device)

    prior_full[mp[:, None], mp[None, :]] = prior_eff
    prior_full[bg_pos, bg_pos] = 1.0 

    return prior_full
