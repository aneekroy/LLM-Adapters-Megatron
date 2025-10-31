#!/usr/bin/env python
# RadialRouter-style 3-model router with vLLM generation (no LoRA).
# - Precompute predictions from 3 base models
# - Train RadialFormer router with KL + (optional) query-query contrastive
# - Route each query to one model and evaluate Perf/Cost/Score
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding

import os, re, gc, json, glob, math, time, copy, argparse, hashlib, datetime, warnings
from collections import defaultdict, Counter
from statistics import median
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import wandb
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams

from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding

try:
    from sklearn.cluster import KMeans
    _HAVE_SK = True
except Exception:
    _HAVE_SK = False
    warnings.warn("scikit-learn not found; contrastive loss will use in-batch positives.")

# ───────────────────────── CLI ─────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["AddSub","MultiArith","SingleEq","gsm8k","AQuA","SVAMP"], required=True)

    # Exactly 3 base models (HF ids or local dirs). Globs allowed; must resolve to 3.
    p.add_argument("--models", required=True,
                   help="Comma-separated THREE model ids/paths, e.g. m1,m2,/local/m3")

    # Cost ($ per 1M tokens) for the 3 models, comma-separated (e.g., 0.562,7.185,0.439)
    p.add_argument("--model_costs", required=False, default=None,
                   help="Comma-separated 3 floats. If omitted, all zeros.")

    # Performance-cost tradeoff (Eq.7)   score = perf - alpha * cost
    p.add_argument("--alpha", type=float, default=0.02, help="0=Performance First; 0.02=Balance; 0.1=Cost First")

    # vLLM generation knobs
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--tp_size", type=int, default=1)
    p.add_argument("--max_model_len", type=int, default=None)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--max_tokens", type=int, default=256)

    # Router training
    p.add_argument("--router_project", default="radialrouter_eval")
    p.add_argument("--router_train_frac", type=float, default=0.8)
    p.add_argument("--router_epochs", type=int, default=30)
    p.add_argument("--router_lr", type=float, default=5e-5)
    p.add_argument("--router_batch", type=int, default=64)
    p.add_argument("--router_model", default="/home/models/mdeberta-v3-base")  # encoder E(x)

    # RadialFormer config
    p.add_argument("--rf_layers", type=int, default=6)
    p.add_argument("--rf_hidden", type=int, default=768)
    p.add_argument("--rf_heads", type=int, default=4)
    p.add_argument("--rf_head_dim", type=int, default=32)  # per-head dim

    # Contrastive loss (optional)
    p.add_argument("--contrastive_lambda", type=float, default=0.5)
    p.add_argument("--contrastive_temp", type=float, default=0.1)
    p.add_argument("--clusters", type=int, default=8, help="KMeans clusters for q–q contrastive; ignored if sklearn not present")

    # Misc
    p.add_argument("--tolerance", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

# ───────────────────────── Data & utils ─────────────────────────
def _ensure_pad_token(tokenizer, model=None):
    """
    Ensure tokenizer has a pad token.
    Prefer reusing eos_token; otherwise add [PAD] and resize model embeddings if provided.
    Also set left padding, which is safer for causal LMs.
    """
    # safer default for causal LMs; harmless for encoders
    try:
        tokenizer.padding_side = "left"
    except Exception:
        pass

    if getattr(tokenizer, "pad_token", None) is None:
        if getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            if model is not None and hasattr(model, "resize_token_embeddings"):
                model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_data(name: str):
    path = f"dataset/{name}/test.json"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing dataset file {path}")
    return json.load(open(path))

def generate_prompt(instr: str) -> str:
    return ("Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instr}\n\n### Response:\n")

def batches(lst, bs):
    for i in range(0, len(lst), bs):
        yield lst[i : i + bs]

def _expand_models(spec: str):
    out = []
    for token in spec.split(","):
        raw = token.strip()
        if not raw:
            continue
        if any(ch in raw for ch in "*?["):
            out.extend(sorted(glob.glob(raw)))
        else:
            out.append(raw.rstrip("/"))
    if len(out) != 3:
        raise ValueError(f"--models must resolve to exactly 3 entries, got {len(out)}: {out}")
    return out

def parse_costs(costs_str, n=3):
    if costs_str is None:
        return [0.0]*n
    vals = [float(x) for x in costs_str.split(",")]
    if len(vals) != n:
        raise ValueError(f"--model_costs must have {n} floats")
    return vals

# Answer extraction
NUM_REGEX = re.compile(r"-?\d+\.?\d*")
def extract_answer(dataset: str, text: str):
    text = text.replace(",", "")
    if dataset.lower() == "aqua":
        m = re.search(r"[ABCDE]", text)
        return m.group(0) if m else ""
    nums = NUM_REGEX.findall(text)
    if not nums:
        return float("inf")
    try:
        return float(nums[-1])
    except ValueError:
        return float("inf")

def numeric_equal(a: float, b: float, tol: float):
    return abs(a - b) <= tol

# ───────────────────────── vLLM inference ─────────────────────────

def build_llm(model_id_or_path: str, args):
    llm = LLM(
        model=model_id_or_path,
        tensor_parallel_size=args.tp_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        trust_remote_code=True,
    )
    sp = SamplingParams(temperature=0.1, top_p=0.75, top_k=40, max_tokens=args.max_tokens)
    return llm, sp

def vllm_generate(llm, sp, instructions):
    prompts = [generate_prompt(i) for i in instructions]
    outs = llm.generate(prompts, sp)
    return [o.outputs[0].text.strip() if o.outputs else "" for o in outs]

# ───────────────────────── Router (Encoder + RadialFormer) ─────────────────────────

class RadialFormerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, head_dim: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(d_model, n_heads*head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads*head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads*head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads*head_dim, d_model, bias=False)
        self.norm_s = nn.LayerNorm(d_model)
        self.norm_r = nn.LayerNorm(d_model)

    def _mh_attn(self, q: torch.Tensor, ctx: torch.Tensor):
        # q: (B, D); ctx: (B, L, D) -> out: (B, D)
        B, L, D = ctx.shape
        qh = self.q_proj(q).view(B, self.n_heads, self.head_dim)        # (B,H,Hd)
        kh = self.k_proj(ctx).view(B, L, self.n_heads, self.head_dim)   # (B,L,H,Hd)
        vh = self.v_proj(ctx).view(B, L, self.n_heads, self.head_dim)   # (B,L,H,Hd)

        # scores: (B,H,L)
        scores = torch.einsum("bhd,blhd->bhl", qh, kh) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)                            # (B,H,L)
        # weighted sum: (B,H,Hd)
        out = torch.einsum("bhl,blhd->bhd", attn, vh)
        out = out.reshape(B, self.n_heads*self.head_dim)
        return self.o_proj(out)  # (B,D)

    def forward(self, r, S, M):
        # r: (B,D) relay state; S: (B,N,D) satellite states; M: (N,D) learnable model embeddings
        B, N, D = S.shape
        # Update satellites independently with ctx=[s_{t-1}, m_i, r_{t-1}]
        new_S = []
        r_ctx = r.unsqueeze(1).expand(B, N, D)         # (B,N,D)
        M_rep = M.unsqueeze(0).expand(B, N, D)         # (B,N,D)
        for i in range(N):
            s_prev = S[:, i, :]                        # (B,D)
            ctx_i = torch.stack([S[:, i, :], M_rep[:, i, :], r_ctx[:, i, :]], dim=1)  # (B,3,D)
            s_upd = self._mh_attn(s_prev, ctx_i)
            s_upd = F.relu(s_upd)
            new_S.append(self.norm_s(s_upd))
        S_new = torch.stack(new_S, dim=1)              # (B,N,D)

        # Update relay with ctx=[r_{t-1}; S_t]
        ctx_r = torch.cat([r.unsqueeze(1), S_new], dim=1)  # (B,1+N,D)
        r_upd = self._mh_attn(r, ctx_r)
        r_upd = self.norm_r(F.relu(r_upd))
        return r_upd, S_new

class RadialFormer(nn.Module):
    def __init__(self, d_model: int, n_models: int, n_layers: int, n_heads: int, head_dim: int):
        super().__init__()
        self.n_models = n_models
        self.layers = nn.ModuleList([RadialFormerLayer(d_model, n_heads, head_dim) for _ in range(n_layers)])
        # learnable per-model embeddings m_i
        self.model_emb = nn.Parameter(torch.randn(n_models, d_model) * 0.02)

    def forward(self, q_embed: torch.Tensor):
        # q_embed: (B,D)  -> returns final satellite states S_T (B,N,D)
        B, D = q_embed.shape
        N = self.n_models
        r = q_embed
        S = self.model_emb.unsqueeze(0).expand(B, N, D).clone()
        for layer in self.layers:
            r, S = layer(r, S, self.model_emb)
        return S  # (B,N,D)

class Router(nn.Module):
    def __init__(self, encoder_name: str, d_model: int, n_models: int, n_layers: int, n_heads: int, head_dim: int):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name, use_fast=True)
        self.encoder = AutoModel.from_pretrained(encoder_name)
        _ensure_pad_token(self.tokenizer, self.encoder)
        self.rf = RadialFormer(d_model, n_models, n_layers, n_heads, head_dim)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    @torch.no_grad()
    def encode(self, input_ids, attention_mask):
        # [CLS] embedding (works for DeBERTa/MDeBERTa)
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h_cls = out.last_hidden_state[:, 0, :]    # (B,D)
        return h_cls

    def forward(self, input_ids, attention_mask):
        h_cls = self.encode(input_ids, attention_mask)   # (B,D)
        S = self.rf(h_cls)                                # (B,N,D)
        logits = self.mlp(S).squeeze(-1)                 # (B,N)
        probs = torch.softmax(logits, dim=-1)            # (B,N)
        return probs, h_cls

# ───────────────────────── Training data ─────────────────────────

class RouterDataset(Dataset):
    def __init__(self, texts, target_probs, tokenizer, max_len=512, cluster_ids=None):
        self.texts = texts
        self.target = torch.tensor(target_probs, dtype=torch.float32)   # (M,N)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.cluster_ids = cluster_ids if cluster_ids is not None else [-1]*len(texts)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return {
            "text": self.texts[i],
            "target": self.target[i],
            "cluster": int(self.cluster_ids[i]),
        }

def collate_fn(batch, tokenizer):
    texts = [b["text"] for b in batch]
    enc = tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
    target = torch.stack([b["target"] for b in batch], dim=0)
    cluster = torch.tensor([b["cluster"] for b in batch], dtype=torch.long)
    return enc["input_ids"], enc["attention_mask"], target, cluster

# ───────────────────────── Losses ─────────────────────────

def kl_loss(p_pred, q_true, eps=1e-12):
    # D_KL(p||q) = sum p log(p/q)
    p = torch.clamp(p_pred, eps, 1.0)
    q = torch.clamp(q_true, eps, 1.0)
    return torch.mean(torch.sum(p * (torch.log(p) - torch.log(q)), dim=-1))

def contrastive_q_q(h, cluster, temp=0.1):
    # Simple in-batch NT-Xent using cluster ids as positives.
    # h: (B,D) L2-normalized
    h = F.normalize(h, p=2, dim=-1)
    sim = torch.matmul(h, h.t()) / temp            # (B,B)
    B = h.size(0)
    mask = torch.eye(B, device=h.device).bool()
    sim = sim.masked_fill(mask, -1e9)

    # positives: same cluster id
    c = cluster.view(-1,1)
    pos_mask = (c == c.t()) & (~mask)
    # For samples with no positive in batch, skip via small epsilon
    exp_sim = torch.exp(sim)
    pos_exp = torch.where(pos_mask, exp_sim, torch.zeros_like(exp_sim))
    pos_sum = pos_exp.sum(dim=1) + 1e-12
    denom = exp_sim.sum(dim=1) + 1e-12
    loss = -torch.log(pos_sum / denom)
    # If no positives -> that row contributes ~ -log(ε/denom); we can downweight by ignoring those rows
    valid = (pos_mask.sum(dim=1) > 0).float()
    if valid.sum() > 0:
        return (loss * valid).sum() / valid.sum()
    else:
        return loss.mean()

# ───────────────────────── Main pipeline ─────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)

    data = load_data(args.dataset)
    instructions = [d["instruction"] for d in data]
    labels = [d["answer"] for d in data]
    is_aqua = (args.dataset.lower() == "aqua")
    if not is_aqua:
        labels = [float(x) if not isinstance(x, (int,float)) else x for x in labels]

    models = _expand_models(args.models)
    costs = parse_costs(args.model_costs, n=3)
    model_names = [os.path.basename(m).strip("/") or m for m in models]

    # Run all three models once to collect predictions
    preds_by_model = []
    for m_idx, (model_id, nice) in enumerate(zip(models, model_names), start=1):
        print(f"\n>> [{m_idx}/3] Generating with {nice} …")
        llm, sp = build_llm(model_id, args)
        all_preds = []
        for batch in tqdm(list(batches(instructions, args.batch_size)), leave=False):
            outs = vllm_generate(llm, sp, batch)
            all_preds.extend([extract_answer(args.dataset, o) for o in outs])
        preds_by_model.append(all_preds)
        del llm; gc.collect()

    # Compute per-query correctness and routing targets (score = perf - alpha*cost)
    N = len(data)
    perf = torch.zeros(N, 3, dtype=torch.float32)
    for i in range(N):
        for j in range(3):
            if is_aqua:
                perf[i, j] = 1.0 if preds_by_model[j][i] == labels[i] else 0.0
            else:
                p = preds_by_model[j][i]
                perf[i, j] = 1.0 if (isinstance(p, (int,float)) and p != float("inf")
                                     and numeric_equal(p, labels[i], args.tolerance)) else 0.0
    cost_vec = torch.tensor(costs, dtype=torch.float32).view(1,3).expand(N,3)
    scores = perf - args.alpha * cost_vec
    target_probs = torch.softmax(scores, dim=1).numpy()   # q_true for KL

    # Split for router train/val
    split = int(args.router_train_frac * N)
    train_texts = instructions[:split]
    val_texts   = instructions[split:]
    train_q = target_probs[:split]
    val_q   = target_probs[split:]
    train_perf = perf[:split]
    val_perf   = perf[split:]

    # Optional clustering for contrastive positives
    train_clusters = None
    if _HAVE_SK and args.clusters > 1:
        print("Clustering train queries for contrastive positives (KMeans)…")
        # quick text embeddings via encoder tokenizer -> later real embeddings are used in loss,
        # but for clusters we can embed with a fresh tiny pass
        tmp_tok = AutoTokenizer.from_pretrained(args.router_model, use_fast=True)
        tmp_enc = AutoModel.from_pretrained(args.router_model).eval()
        _ensure_pad_token(tmp_tok, tmp_enc)
        with torch.no_grad():
            all_emb = []
            for batch in batches(train_texts, 64):
                enc = tmp_tok(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
                out = tmp_enc(**enc)
                all_emb.append(out.last_hidden_state[:,0,:].cpu())
            X = torch.cat(all_emb, dim=0).numpy()
        km = KMeans(n_clusters=min(args.clusters, len(train_texts)), n_init="auto", random_state=args.seed)
        train_clusters = km.fit_predict(X).tolist()

    # Build router
    device = "cuda" if torch.cuda.is_available() else "cpu"
    router = Router(
        encoder_name=args.router_model,
        d_model=args.rf_hidden,
        n_models=3,
        n_layers=args.rf_layers,
        n_heads=args.rf_heads,
        head_dim=args.rf_head_dim,
    ).to(device)

    # Datasets + loaders
    train_ds = RouterDataset(train_texts, train_q, router.tokenizer, cluster_ids=train_clusters)
    val_ds   = RouterDataset(val_texts,   val_q,   router.tokenizer, cluster_ids=[-1]*len(val_texts))
    coll = lambda b: collate_fn(b, router.tokenizer)
    train_dl = DataLoader(train_ds, batch_size=args.router_batch, shuffle=True, num_workers=2, collate_fn=coll)
    val_dl   = DataLoader(val_ds,   batch_size=args.router_batch, shuffle=False, num_workers=2, collate_fn=coll)

    # Optim
    optim = torch.optim.AdamW(router.parameters(), lr=args.router_lr)

    # Logging
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = hashlib.sha1(args.models.encode()).hexdigest()[:8]
    unique_names = set(os.path.basename(m) for m in args.models.split(","))

    run_id = f"{args.dataset}-radialrouter-{unique_names}-{ts}-{tag}"
    wandb.init(project=args.router_project, name=run_id, config=vars(args), reinit=True)

    # Train
    for epoch in range(1, args.router_epochs+1):
        router.train()
        tr_kl, tr_con, tr_total = 0.0, 0.0, 0.0
        for input_ids, attn, q_true, cluster in train_dl:
            input_ids = input_ids.to(device); attn = attn.to(device)
            q_true = q_true.to(device)

            optim.zero_grad()
            p_pred, h_cls = router(input_ids, attn)      # p_pred: (B,3), h_cls: (B,D)
            loss_kl = kl_loss(p_pred, q_true)
            loss_con = contrastive_q_q(h_cls, cluster.to(device), temp=args.contrastive_temp) if args.contrastive_lambda > 0 else 0.0
            loss = loss_kl + args.contrastive_lambda * loss_con
            loss.backward()
            optim.step()

            tr_kl += float(loss_kl) * len(input_ids)
            tr_con += (float(loss_con) if isinstance(loss_con, torch.Tensor) else float(loss_con)) * len(input_ids)
            tr_total += float(loss) * len(input_ids)

        # Val KL only (no contrastive on val)
        router.eval()
        va_kl = 0.0; count = 0
        with torch.no_grad():
            for input_ids, attn, q_true, cluster in val_dl:
                input_ids = input_ids.to(device); attn = attn.to(device)
                q_true = q_true.to(device)
                p_pred, _ = router(input_ids, attn)
                va_kl += float(kl_loss(p_pred, q_true)) * len(input_ids)
                count += len(input_ids)

        wandb.log({
            "epoch": epoch,
            "train_total": tr_total / max(1, len(train_ds)),
            "train_kl": tr_kl / max(1, len(train_ds)),
            "train_contrastive": tr_con / max(1, len(train_ds)),
            "val_kl": va_kl / max(1, count),
        })

    # Inference: route all queries, then pick that model's previously generated prediction
    router.eval()
    all_probs = []
    with torch.no_grad():
        for batch in batches(instructions, args.router_batch):
            enc = router.tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
            p, _ = router(enc["input_ids"].to(device), enc["attention_mask"].to(device))
            all_probs.append(p.cpu())
    P = torch.cat(all_probs, dim=0)          # (N,3)
    choices = torch.argmax(P, dim=1).tolist()

    final_preds = []
    for i, c in enumerate(choices):
        final_preds.append(preds_by_model[c][i])

    # Evaluate Perf, Cost, Score
    correct = []
    for i, p in enumerate(final_preds):
        if is_aqua:
            correct.append(1 if p == labels[i] else 0)
        else:
            ok = (isinstance(p, (int,float)) and p != float("inf")
                  and numeric_equal(p, labels[i], args.tolerance))
            correct.append(1 if ok else 0)
    perf_overall = sum(correct) / len(correct)
    chosen_counts = Counter(choices)
    avg_cost = sum(chosen_counts[j] * costs[j] for j in range(3)) / len(choices)
    score = perf_overall - args.alpha * avg_cost

    print("\n==== ROUTED RESULTS (RadialRouter-style) ====")
    print(f"Performance (accuracy): {perf_overall:.4f}")
    print(f"Average cost ($/1M tokens): {avg_cost:.3f}")
    print(f"Score (perf - alpha*cost): {score:.4f}")
    print("Model usage:", {model_names[j]: chosen_counts[j] for j in range(3)})

    wandb.log({
        "final_perf": perf_overall,
        "final_cost": avg_cost,
        "final_score": score,
        "usage_m0": chosen_counts[0] / len(choices),
        "usage_m1": chosen_counts[1] / len(choices),
        "usage_m2": chosen_counts[2] / len(choices),
    })

    # Save per-record outputs
    out = copy.deepcopy(data)
    for i, rec in enumerate(out):
        rec["router_probs"] = P[i].tolist()
        rec["routed_model_idx"] = int(choices[i])
        rec["routed_model_name"] = model_names[choices[i]]
        rec["prediction"] = final_preds[i]
        rec["correct"] = bool(correct[i])

    os.makedirs("experiment", exist_ok=True)
    tag_models = "__".join([re.sub(r'[^A-Za-z0-9._-]+','-', os.path.basename(m)) for m in models])
    out_path = f"experiment/{args.dataset}-radialrouter-{tag_models}-{ts}.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print("Saved →", out_path)
    wandb.finish()

if __name__ == "__main__":
    main()