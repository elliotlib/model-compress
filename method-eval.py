import os
import gc
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

SEED = int(os.environ.get("SEED", 42))
MODEL_NAME = "Qwen/Qwen2-0.5B"
GROUP = 128
SEQ_LEN = 128
TRAIN_BLOCKS = 256
VALID_BLOCKS = 32
TEST_BLOCKS = 32
EPOCHS = 3
BIT_LR, SCALE_LR, SCALAR_LR = 1e-3, 1e-4, 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
print("device:", DEVICE, "| seed:", SEED)

# ---------------- DATA ----------------

ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
tok = AutoTokenizer.from_pretrained(MODEL_NAME)


def ids_of(split):
    return tok("\n".join(ds[split]["text"]), return_tensors="pt").input_ids[0]


train_ids = ids_of("train")
valid_ids = ids_of("validation")
test_ids = ids_of("test")


def make_blocks(ids, count, seed=None):
    if seed is None:
        return torch.stack([ids[i * SEQ_LEN:(i + 1) * SEQ_LEN] for i in range(count)])
    rng = random.Random(seed)
    starts = rng.sample(range(len(ids) - SEQ_LEN), count)
    return torch.stack([ids[s:s + SEQ_LEN] for s in starts])


train_blocks = make_blocks(train_ids, TRAIN_BLOCKS, seed=SEED)
valid_blocks = make_blocks(valid_ids, VALID_BLOCKS)
test_blocks = make_blocks(test_ids, TEST_BLOCKS)

# ---------------- EVAL ----------------


@torch.no_grad()
def perplexity(model, blocks):
    model.eval()
    nll, n = 0.0, 0
    for b in blocks:
        ids = b[None].to(DEVICE)
        logits = model(input_ids=ids).logits[:, :-1].float()
        labels = ids[:, 1:]
        nll += F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="sum"
        ).item()
        n += labels.numel()
    loss = nll / n
    return loss, float(np.exp(loss))


# ---------------- MODEL HELPERS ----------------


def load_model():
    m = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    m.config.use_cache = False
    return m.to(DEVICE)


def target_linears(model):
    out = []
    for name, mod in model.named_modules():
        if name and isinstance(mod, nn.Linear) and name != "lm_head":
            if mod.in_features % GROUP == 0:
                out.append((name, mod))
    return out


def replace_linear(model, name, new_layer):
    parent_name, child = name.rsplit(".", 1)
    setattr(model.get_submodule(parent_name), child, new_layer)


# ---------------- METHOD A: BINARY QAT ----------------


class BinaryQATLinear(nn.Module):
    def __init__(self, orig):
        super().__init__()
        w = orig.weight.detach().float()
        self.out_f, self.in_f = w.shape
        self.n_groups = self.in_f // GROUP
        self.sign_logits = nn.Parameter(w.clone())
        grouped = w.reshape(self.out_f, self.n_groups, GROUP)
        self.scales = nn.Parameter(grouped.abs().mean(-1).clamp_min(1e-8))
        self.bias = None if orig.bias is None else nn.Parameter(orig.bias.detach().clone(), requires_grad=False)

    def trainable(self):
        return [self.sign_logits], [self.scales]

    def forward(self, x):
        hard = torch.where(self.sign_logits >= 0, 1.0, -1.0).to(self.sign_logits.dtype)
        sign_ste = self.sign_logits + (hard - self.sign_logits).detach()
        w = sign_ste * self.scales.repeat_interleave(GROUP, dim=1)
        return F.linear(x, w, self.bias)


# ---------------- METHOD B: ONE SCALAR PER GROUP ----------------


class ScalarGroupLinear(nn.Module):
    def __init__(self, orig):
        super().__init__()
        w = orig.weight.detach().float()
        self.out_f, self.in_f = w.shape
        self.n_groups = self.in_f // GROUP
        signs = torch.where(w >= 0, 1.0, -1.0)
        scales = w.reshape(self.out_f, self.n_groups, GROUP).abs().mean(-1)
        init = (signs.reshape(self.out_f, self.n_groups, GROUP) * scales.unsqueeze(-1)).mean(-1)
        self.p = nn.Parameter(init.clone())
        self.bias = None if orig.bias is None else nn.Parameter(orig.bias.detach().clone(), requires_grad=False)

    def trainable(self):
        return [self.p]

    def forward(self, x):
        # имитация BF16-округления константы через straight-through estimator
        p_bf16 = self.p.to(torch.bfloat16).float()
        p_eff = self.p + (p_bf16 - self.p).detach()
        xg = x.reshape(*x.shape[:-1], self.n_groups, GROUP).sum(-1)  # (..., n_groups)
        y = torch.matmul(xg, p_eff.T)
        return y + self.bias if self.bias is not None else y


# ---------------- TRAIN ----------------


def train(model, lr_groups, label):
    opt = torch.optim.AdamW(lr_groups, weight_decay=0.0)
    for ep in range(EPOCHS):
        model.train()
        total = 0.0
        for i in range(TRAIN_BLOCKS):
            ids = train_blocks[i:i + 1].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = model(input_ids=ids, labels=ids).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"{label}: non-finite loss at epoch {ep + 1}, block {i}")
            loss.backward()
            opt.step()
            total += loss.item()
        v_loss, v_ppl = perplexity(model, valid_blocks)
        print(f"[{label}] epoch {ep + 1}/{EPOCHS} | train loss {total / TRAIN_BLOCKS:.4f} | valid ppl {v_ppl:.2f}")


def run_binary():
    torch.manual_seed(SEED)
    model = load_model()
    sign_ps, scale_ps = [], []
    for name, mod in target_linears(model):
        layer = BinaryQATLinear(mod).to(DEVICE)
        replace_linear(model, name, layer)
        s, c = layer.trainable()
        sign_ps += s
        scale_ps += c
    train(model, [{"params": sign_ps, "lr": BIT_LR}, {"params": scale_ps, "lr": SCALE_LR}], "binary")
    return model


def run_scalar():
    torch.manual_seed(SEED)
    model = load_model()
    ps = []
    for name, mod in target_linears(model):
        layer = ScalarGroupLinear(mod).to(DEVICE)
        replace_linear(model, name, layer)
        ps += layer.trainable()
    train(model, [{"params": ps, "lr": SCALAR_LR}], "scalar")
    return model


# ---------------- MAIN ----------------

results = {"seed": SEED}

base = load_model()
v_loss, v_ppl = perplexity(base, valid_blocks)
t_loss, t_ppl = perplexity(base, test_blocks)
results["fp32"] = {"valid_ppl": v_ppl, "test_ppl": t_ppl, "test_loss": t_loss}
print(f"FP32 | valid ppl {v_ppl:.2f} | test ppl {t_ppl:.2f}")
del base
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None

for key, runner in [("binary", run_binary), ("scalar", run_scalar)]:
    model = runner()
    v_loss, v_ppl = perplexity(model, valid_blocks)
    t_loss, t_ppl = perplexity(model, test_blocks)  # test открывается один раз, после обучения
    results[key] = {"valid_ppl": v_ppl, "test_ppl": t_ppl, "test_loss": t_loss}
    print(f"{key} | valid ppl {v_ppl:.2f} | test ppl {t_ppl:.2f}")
    del model
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

out_name = f"results_seed{SEED}.json"
with open(out_name, "w") as f:
    json.dump(results, f, indent=2)
print("saved:", out_name)
print(json.dumps(results, indent=2))
