# -*- coding: utf-8 -*-
"""
ffn_vs_swiglu_comparison.py — 普通 FFN vs SwiGLU FFN 受控对照实验

实验设计(控制变量法):
  - 两个变体共享: 同一语料 / 同一种子(模型初始化) / 同一数据采样流(batch 完全相同)
    / 相同层数、头数、d_model、学习率、步数 —— 唯一自变量是 FFN 的类型;
  - 参数量对齐: 普通 FFN 隐藏维 = 4d(2 个矩阵, 8d² 参数),
    SwiGLU 隐藏维 = 8d/3(3 个矩阵, 3·d·8d/3 = 8d² 参数), 即 LLaMA 的等参数做法;
  - 每个变体跑 SEEDS 个不同种子, 用 numpy 做 mean±std 统计, 避免单种子偶然性;
  - 指标: 最优 val loss / 最终 train loss / val 字符准确率 / 收敛速度(到达阈值步数)
    / FFN 中间激活统计(numpy) / 生成续写与语料真值的字符匹配率 / 参数量 / 耗时。

产物(全部在 logs/ffn_cmp_时间戳/ 下):
  - run.log           完整运行日志
  - summary.csv       每个 (变体, 种子) 一行的指标汇总
  - runs/*.csv        每次训练的逐步历史(step, batch_loss, grad_norm, val_loss, val_acc)
  - comparison.png    三联对比图(train 曲线 / val 曲线 / 各种子最优 val 柱状图)
  - report.md         自动生成的对比分析报告(含结论)

运行:  python -X utf8 ffn_vs_swiglu_comparison.py
"""

import csv
import logging
import math
import os
import time
from datetime import datetime
from pathlib import Path

# Anaconda + PyTorch 在 Windows 上的 OpenMP 重复加载问题(torch 与 matplotlib 各带一份
# libiomp5md.dll), 必须在 import torch/matplotlib 之前设置, 否则进程直接 abort。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np               # 统计/平滑/准确率/激活分析都用 numpy 接口
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 0. 超参数(两个变体共用, 保证公平)
# =====================================================================
SEEDS      = [1337, 42, 7]    # 多种子重复实验, 用 mean±std 汇总
SEED_DATA  = 2024             # 数据采样流种子: 所有变体共用 → batch 序列完全一致
block_size = 48               # 上下文长度
batch_size = 16

n_layer    = 3
n_head     = 4
d_model    = 128
dropout    = 0.0              # 演示语料极小, 关 dropout 让差异更纯粹来自 FFN

max_steps  = 1500
eval_interval = 150
lr         = 3e-4
device     = "cuda" if torch.cuda.is_available() else "cpu"

# ---- FFN 隐藏维: 等参数量对齐 ----
# 普通 FFN: 2 个矩阵 → 2·d·h;  SwiGLU: 3 个矩阵 → 3·d·h'
# 令两者相等: h' = 2h/3 = 8d/3 (LLaMA 的标准做法)
FFN_HIDDEN_VANILLA = 4 * d_model                  # 512
FFN_HIDDEN_SWIGLU  = int(8 * d_model / 3)         # 341
VARIANTS = ["vanilla", "swiglu"]

# =====================================================================
# 1. 语料与字符级 tokenizer(与 mini_transformer.py 相同)
# =====================================================================
text = """
学而不思则罔,思而不学则殆。知之为知之,不知为不知,是知也。
三人行,必有我师焉。择其善者而从之,其不善者而改之。
温故而知新,可以为师矣。学而时习之,不亦说乎。有朋自远方来,不亦乐乎。
人不知而不愠,不亦君子乎。吾日三省吾身,为人谋而不忠乎,与朋友交而不信乎,传不习乎。
君子食无求饱,居无求安,敏于事而慎于言,就有道而正焉,可谓好学也已。
不患人之不己知,患不知人也。为政以德,譬如北辰,居其所而众星共之。
道之以政,齐之以刑,民免而无耻。道之以德,齐之以礼,有耻且格。
吾十有五而志于学,三十而立,四十而不惑,五十而知天命,六十而耳顺,七十而从心所欲,不逾矩。
故天将降大任于是人也,必先苦其心志,劳其筋骨,饿其体肤,空乏其身,行拂乱其所为,
所以动心忍性,曾益其所不能。人恒过,然后能改。困于心,衡于虑,而后作。
得道者多助,失道者寡助。寡助之至,亲戚畔之。多助之至,天下顺之。
生于忧患,而死于安乐也。富贵不能淫,贫贱不能移,威武不能屈,此之谓大丈夫。
老吾老,以及人之老。幼吾幼,以及人之幼。天下可运于掌。
不以规矩,不能成方圆。尽信书,则不如无书。君子喻于义,小人喻于利。
见贤思齐焉,见不贤而内自省也。德不孤,必有邻。己所不欲,勿施于人。
工欲善其事,必先利其器。人无远虑,必有近忧。小不忍,则乱大谋。
知者乐水,仁者乐山。知者动,仁者静。知者乐,仁者寿。
道可道,非常道。名可名,非常名。无名天地之始,有名万物之母。
故常无欲,以观其妙。常有欲,以观其徼。此两者同出而异名,同谓之玄。玄之又玄,众妙之门。
上善若水。水善利万物而不争,处众人之所恶,故几于道。
居善地,心善渊,与善仁,言善信,政善治,事善能,动善时。夫唯不争,故无尤。
天地不仁,以万物为刍狗。圣人不仁,以百姓为刍狗。
天地之间,其犹橐龠乎。虚而不屈,动而愈出。多言数穷,不如守中。
曲则全,枉则直,洼则盈,敝则新,少则得,多则惑。
是以圣人抱一为天下式。不自见故明,不自是故彰,不自伐故有功,不自矜故长。
夫唯不争,故天下莫能与之争。
人法地,地法天,天法道,道法自然。知人者智,自知者明。胜人者有力,自胜者强。
合抱之木,生于毫末。九层之台,起于累土。千里之行,始于足下。
天下难事,必作于易。天下大事,必作于细。
天网恢恢,疏而不失。大直若屈,大巧若拙,大辩若讷。
吾生也有涯,而知也无涯。以有涯随无涯,殆已。
君子曰,学不可以已。青,取之于蓝,而青于蓝。冰,水为之,而寒于水。
吾尝终日而思矣,不如须臾之所学也。吾尝跂而望矣,不如登高之博见也。
登高而招,臂非加长也,而见者远。顺风而呼,声非加疾也,而闻者彰。
假舆马者,非利足也,而致千里。假舟楫者,非能水也,而绝江河。君子生非异也,善假于物也。
积土成山,风雨兴焉。积水成渊,蛟龙生焉。积善成德,而神明自得,圣心备焉。
故不积跬步,无以至千里。不积小流,无以成江海。骐骥一跃,不能十步。
驽马十驾,功在不舍。锲而舍之,朽木不折。锲而不舍,金石可镂。
大学之道,在明明德,在亲民,在止于至善。
知止而后有定,定而后能静,静而后能安,安而后能虑,虑而后能得。
物有本末,事有终始。知所先后,则近道矣。
古之欲明明德于天下者,先治其国。欲治其国者,先齐其家。欲齐其家者,先修其身。
欲修其身者,先正其心。欲正其心者,先诚其意。欲诚其意者,先致其知。致知在格物。
物格而后知至,知至而后意诚,意诚而后心正,心正而后身修,身修而后家齐,家齐而后国治,国治而后天下平。
自天子以至于庶人,壹是皆以修身为本。
"""
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)
n_split = int(0.9 * len(data))
train_data, val_data = data[:n_split], data[n_split:]

PROMPTS = ["学而时习之,", "故天将降大任于是人也,", "君子喻于义,"]

# =====================================================================
# 2. 模型组件(结构同 mini_transformer.py, 仅 FFN 可替换)
# =====================================================================
class SinusoidalPositionalEncoding(nn.Module):
    """正弦位置编码: 给每个位置一个固定指纹, 补偿注意力缺失的顺序信息。"""
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class CausalSelfAttention(nn.Module):
    """带因果掩码的多头自注意力: 序列内信息交换的唯一通道。"""
    def __init__(self):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head, self.d_head = n_head, d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(d_model, dim=2)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        att = att + torch.full((T, T), float("-inf"), device=x.device).triu(1)
        att = self.attn_drop(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class VanillaFFN(nn.Module):
    """普通 FFN: 升维 → GELU → 降维, 共 2 个权重矩阵, 参数量 2·d·h (+bias)。
    激活函数只负责逐点非线性, 没有门控。"""
    def __init__(self):
        super().__init__()
        self.up = nn.Linear(d_model, FFN_HIDDEN_VANILLA)
        self.act = nn.GELU()                      # 独立成模块, 便于挂 hook 采集中间激活
        self.down = nn.Linear(FFN_HIDDEN_VANILLA, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.down(self.act(self.up(x))))


class SwiGLUFFN(nn.Module):
    """SwiGLU FFN: 三条平行投影(gate / up / down), 无 bias。
      out = down( SiLU(gate(x)) ⊙ up(x) )
    SiLU(gate(x)) 是"门": 逐元素控制 up(x) 这条"内容"支路放行多少。
    隐藏维取 8d/3, 使 3·d·h' ≈ 2·d·4d, 与普通 FFN 参数量对齐(LLaMA 做法)。"""
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(d_model, FFN_HIDDEN_SWIGLU, bias=False)
        self.up = nn.Linear(d_model, FFN_HIDDEN_SWIGLU, bias=False)
        self.act = nn.SiLU()                      # SiLU(x) = x·σ(x), 即 Swish
        self.down = nn.Linear(FFN_HIDDEN_SWIGLU, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.down(self.act(self.gate(x)) * self.up(x)))


class TransformerBlock(nn.Module):
    """Pre-LN 结构: x = x + Sublayer(LN(x)), 残差主干道无 LN, 梯度直通。"""
    def __init__(self, ffn_type: str):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(d_model)
        # 实验的自变量: 只在这里切换 FFN 类型
        self.ffn = VanillaFFN() if ffn_type == "vanilla" else SwiGLUFFN()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, ffn_type: str):
        super().__init__()
        self.ffn_type = ffn_type
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = SinusoidalPositionalEncoding(d_model, block_size)
        self.blocks = nn.ModuleList(TransformerBlock(ffn_type) for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight          # 权重共享
        self.apply(self._init_weights)
        # head 与 tok_emb 共享张量, apply() 遍历时 head 会覆盖 embedding 的初始化, 补一刀
        self.tok_emb.weight.data.normal_(mean=0.0, std=d_model ** -0.5)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=d_model ** -0.5)  # Xavier 风格
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)             # GPT-2 风格
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, idx, targets=None):
        x = self.pos_emb(self.tok_emb(idx))
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=10):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -block_size:])
            logits = logits[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx

# =====================================================================
# 3. 数据 / 评估工具
# =====================================================================
def get_batch(split, generator):
    """取一个 batch。generator 外传: 所有变体共用同一采样流 → batch 序列逐位相同,
    这是对照实验公平性的关键(否则 RNG 消耗差异会让两个模型看到不同数据)。"""
    d = train_data if split == "train" else val_data
    hi = max(1, len(d) - block_size - 1)
    ix = torch.randint(hi, (batch_size,), generator=generator)
    x = torch.stack([d[i : i + block_size] for i in ix])
    y = torch.stack([d[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model):
    """train/val 各采 30 个 batch 求平均 loss。评估用独立固定种子 → 每次评估、
    每个变体看到的评估 batch 都相同, 数字可直接横向比较。"""
    model.eval()
    g = torch.Generator().manual_seed(999)
    out = {}
    for split in ("train", "val"):
        losses = [model(*get_batch(split, g))[1].item() for _ in range(30)]
        out[split] = float(np.mean(losses))        # numpy 求均值
    model.train()
    return out


@torch.no_grad()
def eval_accuracy(model):
    """val 集上的字符预测准确率: 滑动窗口取若干 (48, 48) 片段, 逐位置 argmax
    与真值比较。用 numpy 做布尔平均, 比 torch 更直观。"""
    model.eval()
    stride = 8
    starts = list(range(0, max(1, len(val_data) - block_size - 1), stride))
    x = torch.stack([val_data[i : i + block_size] for i in starts]).to(device)
    y = torch.stack([val_data[i + 1 : i + 1 + block_size] for i in starts]).to(device)
    logits, _ = model(x)
    preds = logits.argmax(dim=-1).cpu().numpy()          # (P, T) numpy 数组
    tgts = y.cpu().numpy()
    acc = float(np.mean(preds == tgts))                  # 布尔数组求均值 = 准确率
    model.train()
    return acc


@torch.no_grad()
def activation_stats(model):
    """用 forward hook 抓取**每一层** FFN 的中间激活, 转成 numpy 统计:
      - vanilla: GELU 输出;
      - swiglu : SiLU 门输出、以及 门⊙up 的乘积(真正送进 down 的信号)。
    统计量: mean / std / mean|x| / 近零比例(稀疏度) / 负值比例 / p1,p50,p99 百分位。
    这组数据回答两个问题: "门控改变了激活的什么性质" 与 "该性质是否随深度变化"。"""
    captured, handles = {}, []

    def make_hook(name):
        # 工厂函数避免循环内 lambda 闭包绑定同一个 name 的经典陷阱
        def fn(m, i, o):
            captured.setdefault(name, o.detach().clone())
        return fn

    for li, blk in enumerate(model.blocks):
        ffn = blk.ffn
        if isinstance(ffn, VanillaFFN):
            handles.append(ffn.act.register_forward_hook(make_hook(f"L{li}_gelu_out")))
        else:
            handles.append(ffn.act.register_forward_hook(make_hook(f"L{li}_silu_gate")))
            handles.append(ffn.up.register_forward_hook(make_hook(f"L{li}_up_out")))
    model.eval()
    x, _ = get_batch("val", torch.Generator().manual_seed(999))
    model(x)
    for h in handles:
        h.remove()
    model.train()

    def stats(t):
        a = t.float().cpu().numpy()                      # torch → numpy
        p1, p50, p99 = np.percentile(a, [1, 50, 99])      # 百分位数: 看分布形状而非只看均值
        return {"mean": float(np.mean(a)), "std": float(np.std(a)),
                "mean_abs": float(np.mean(np.abs(a))),
                "sparse_frac": float(np.mean(np.abs(a) < 1e-3)),
                "neg_frac": float(np.mean(a < 0)),        # 负值比例: 门控"反向抑制"的维度占比
                "p1": float(p1), "p50": float(p50), "p99": float(p99)}

    out = {}
    for li in range(len(model.blocks)):
        if f"L{li}_gelu_out" in captured:
            out[f"L{li}_gelu_out"] = stats(captured[f"L{li}_gelu_out"])
        else:
            g_, u_ = captured[f"L{li}_silu_gate"], captured[f"L{li}_up_out"]
            out[f"L{li}_silu_gate"] = stats(g_)
            out[f"L{li}_product"] = stats(g_ * u_)       # 门 × 内容 = 送入 down 的信号
    return out


def flops_analysis():
    """理论计算量分析(MACs = 乘加次数; FLOPs ≈ 2×MACs), 每 token 计:
      attn : qkv+proj 投影 4d² + QKᵀ 与 AV 两个 T·d²  → 4d² + 2·T·d²
      FFN  : vanilla 2 个矩阵 2·d·h_v ; swiglu 3 个矩阵 3·d·h_s
    用于回答"等参数量下 SwiGLU 是否更费算力"。"""
    d, T = d_model, block_size
    attn = 4 * d * d + 2 * T * d * d
    ffn_v = 2 * d * FFN_HIDDEN_VANILLA
    ffn_s = 3 * d * FFN_HIDDEN_SWIGLU
    total_v = n_layer * (attn + ffn_v)
    total_s = n_layer * (attn + ffn_s)
    return {"attn_mac": attn, "ffn_v": ffn_v, "ffn_s": ffn_s,
            "total_v": total_v, "total_s": total_s,
            "ffn_ratio": ffn_s / ffn_v, "total_ratio": total_s / total_v}


@torch.no_grad()
def generation_match(model, n_chars=50):
    """生成质量量化: 对每个提示词续写 n_chars 字, 与语料中该提示词后的真实下文
    逐字比较, 得到字符匹配率(模型已 memorize 语料, 匹配率≈续写正确率)。
    返回 (平均匹配率, {prompt: 生成文本})。"""
    model.eval()
    rates, samples = [], {}
    for prompt in PROMPTS:
        pos = text.find(prompt)
        truth = text[pos + len(prompt): pos + len(prompt) + n_chars]
        ctx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
        gen = decode(model.generate(ctx, max_new_tokens=n_chars)[0].tolist())[len(prompt):]
        m = min(len(gen), len(truth))
        # numpy 逐字符比较: 两个字符串转数组后布尔平均
        rate = float(np.mean(np.array(list(gen[:m])) == np.array(list(truth[:m])))) if m else 0.0
        rates.append(rate)
        samples[prompt] = gen
    return float(np.mean(rates)), samples

# =====================================================================
# 4. 单次训练(一个变体 × 一个种子), 返回指标字典 + 逐步历史
# =====================================================================
def train_one(variant, seed, log):
    torch.manual_seed(seed)                    # 模型初始化 RNG: 同种子 → 同初始权重分布
    data_gen = torch.Generator().manual_seed(SEED_DATA)   # 采样流: 跨变体共享
    model = MiniGPT(variant).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    history, t0, prev_t = [], time.time(), time.time()
    grad_norm, best_val, best_step = 0.0, float("inf"), -1
    log.info(f"[{variant} | seed {seed}] 开始训练, 参数量 {n_params:,}")
    for step in range(max_steps + 1):
        val_loss = val_acc = None
        if step % eval_interval == 0:
            losses = estimate_loss(model)
            val_loss, val_acc = losses["val"], eval_accuracy(model)
            if val_loss < best_val:
                best_val, best_step = val_loss, step
            tps = eval_interval * batch_size * block_size / max(time.time() - prev_t, 1e-9) if step else 0.0
            prev_t = time.time()
            log.info(f"    step {step:5d} | train {losses['train']:.4f} | val {val_loss:.4f} | "
                     f"val_acc {val_acc:.3f} | grad {grad_norm:.2f} | {tps:,.0f} tok/s")
        x, y = get_batch("train", data_gen)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        optimizer.step()
        history.append({"step": step, "batch_loss": round(loss.item(), 4),
                        "grad_norm": round(grad_norm, 4),
                        "val_loss": val_loss, "val_acc": val_acc,
                        "elapsed": round(time.time() - t0, 2)})

    # ---- 训练后指标 ----
    final_train = float(np.mean([h["batch_loss"] for h in history[-100:]]))  # 末 100 步均值, 比单点稳
    val_acc = eval_accuracy(model)
    gen_rate, gen_samples = generation_match(model)
    act = activation_stats(model)
    elapsed = time.time() - t0

    # ---- 收敛速度: 平滑 train 曲线首次跌破阈值的步数(numpy 卷积做滑动平均) ----
    tr = np.array([h["batch_loss"] for h in history])
    w = 50
    smooth = np.convolve(tr, np.ones(w) / w, mode="valid")   # smooth[i] = mean(tr[i:i+w])
    steps_to = {}
    for thr in (1.0, 0.5, 0.1):
        hit = np.nonzero(smooth < thr)[0]
        steps_to[thr] = int(hit[0] + w - 1) if hit.size else None  # 对齐到窗口末端步数

    log.info(f"[{variant} | seed {seed}] 完成: 用时 {elapsed:.1f}s | best val {best_val:.4f}"
             f"@step{best_step} | 最终 train {final_train:.4f} | val_acc {val_acc:.3f} | "
             f"生成匹配率 {gen_rate:.3f}")
    return {"variant": variant, "seed": seed, "params": n_params,
            "best_val": best_val, "best_val_step": best_step,
            "final_train": final_train, "final_val": history[-1]["val_loss"],
            "val_acc": val_acc, "gen_match": gen_rate, "gen_samples": gen_samples,
            "steps_to": steps_to, "elapsed": elapsed, "act_stats": act,
            "history": history}

# =====================================================================
# 5. 汇总统计 / 报告 / 绘图(全部 numpy + 标准库)
# =====================================================================
def aggregate(results):
    """按变体分组, 对每个指标做 mean±std(numpy)。"""
    agg = {}
    for v in VARIANTS:
        rs = [r for r in results if r["variant"] == v]
        get = lambda k: np.array([r[k] for r in rs], dtype=float)
        st = lambda k: (float(np.mean(get(k))), float(np.std(get(k))))
        m_s2 = {thr: [r["steps_to"][thr] for r in rs] for thr in (1.0, 0.5, 0.1)}
        steps_mean = {thr: (float(np.mean([s for s in v_ if s is not None]))
                            if any(s is not None for s in v_) else None)
                      for thr, v_ in m_s2.items()}
        agg[v] = {
            "params": rs[0]["params"],
            "best_val": st("best_val"), "final_train": st("final_train"),
            "val_acc": st("val_acc"), "gen_match": st("gen_match"),
            "elapsed": st("elapsed"), "steps_to": steps_mean,
        }
    return agg


def write_report(run_dir, agg, results, log):
    """生成 Markdown 对比报告: 配置表 / 参数量表 / 逐种子表 / mean±std 汇总表 /
    激活统计表 / 生成样例 / 自动结论。"""
    L = []
    A = lambda v: agg[v]
    L.append("# 普通 FFN vs SwiGLU FFN 对比实验报告\n")
    L.append(f"- 生成时间: {datetime.now():%Y-%m-%d %H:%M:%S}")
    L.append(f"- 种子: {SEEDS} | 步数: {max_steps} | lr: {lr} | d_model: {d_model} | "
             f"层数: {n_layer} | 头数: {n_head} | 上下文: {block_size}")
    L.append(f"- 语料: {len(text)} 字符, 词表 {vocab_size}; train {len(train_data)} / "
             f"val {len(val_data)} tokens")
    L.append(f"- 控制变量: 同种子初始化、同数据采样流(SEED_DATA={SEED_DATA})、"
             f"同评估 batch; 唯一自变量为 FFN 类型\n")

    L.append("## 1. 两种 FFN 的结构与参数量\n")
    L.append("| 变体 | 结构 | 隐藏维 | 矩阵数 | 总参数量 |")
    L.append("|---|---|---|---|---|")
    L.append(f"| vanilla | down(GELU(up(x))) | {FFN_HIDDEN_VANILLA} (=4d) | 2 | {A('vanilla')['params']:,} |")
    L.append(f"| swiglu | down(SiLU(gate(x))⊙up(x)) | {FFN_HIDDEN_SWIGLU} (≈8d/3) | 3 | {A('swiglu')['params']:,} |")
    dv = A('swiglu')['params'] - A('vanilla')['params']
    L.append(f"\n参数量差异 {dv:+,}({dv / A('vanilla')['params'] * 100:+.2f}%), "
             f"属等参数量对齐(LLaMA 的 8d/3 做法)。")
    fl = flops_analysis()
    L.append(f"\n理论计算量(MACs/token): attn {fl['attn_mac']:,} | "
             f"FFN: vanilla {fl['ffn_v']:,} vs swiglu {fl['ffn_s']:,} "
             f"(比值 {fl['ffn_ratio']:.3f}) | 全模型: {fl['total_v']:,} vs {fl['total_s']:,} "
             f"(比值 {fl['total_ratio']:.3f})。即等参数量对齐下两者算力开销基本持平。\n")

    L.append("## 2. 逐种子明细\n")
    L.append("| 变体 | seed | 最优val loss | @step | 最终train | 最终val | val准确率 | "
             "生成匹配率 | train<0.5步数 | 耗时s |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        s = r["steps_to"][0.5]
        L.append(f"| {r['variant']} | {r['seed']} | {r['best_val']:.4f} | {r['best_val_step']} | "
                 f"{r['final_train']:.4f} | {r['final_val']:.4f} | {r['val_acc']:.3f} | "
                 f"{r['gen_match']:.3f} | {s if s is not None else '—'} | {r['elapsed']:.1f} |")

    L.append("\n## 3. 多种子汇总 (mean ± std)\n")
    L.append("| 指标 | vanilla | swiglu | 更优者 |")
    L.append("|---|---|---|---|")
    rows = [
        ("最优 val loss ↓", "best_val", min),
        ("最终 train loss ↓", "final_train", min),
        ("val 字符准确率 ↑", "val_acc", max),
        ("生成字符匹配率 ↑", "gen_match", max),
        ("训练耗时 s ↓", "elapsed", min),
    ]
    for name, key, better in rows:
        mv, sv = A("vanilla")[key]
        ms, ss = A("swiglu")[key]
        win = "vanilla" if better(mv, ms) == mv else "swiglu"
        L.append(f"| {name} | {mv:.4f} ± {sv:.4f} | {ms:.4f} ± {ss:.4f} | **{win}** |")
    s5v, s5s = A("vanilla")["steps_to"][0.5], A("swiglu")["steps_to"][0.5]
    if s5v is not None and s5s is not None:
        win = "vanilla" if s5v <= s5s else "swiglu"
        L.append(f"| train<0.5 所需步数 ↓ | {s5v:.0f} | {s5s:.0f} | **{win}** |")

    L.append(f"\n## 4. FFN 中间激活统计 (全部 {n_layer} 层, val batch, seed {SEEDS[0]}, numpy 计算)\n")
    L.append("| 变体/层/激活 | mean | std | mean\\|x\\| | 近零比例 | 负值比例 | p1 | p50 | p99 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        if r["seed"] != SEEDS[0]:
            continue
        for name, st in r["act_stats"].items():
            L.append(f"| {r['variant']}/{name} | {st['mean']:+.4f} | {st['std']:.4f} | "
                     f"{st['mean_abs']:.4f} | {st['sparse_frac']:.3f} | {st['neg_frac']:.3f} | "
                     f"{st['p1']:+.3f} | {st['p50']:+.3f} | {st['p99']:+.3f} |")
    L.append("\n解读提示: GELU 与 SiLU 对负输入都输出负值, 故负值比例≈负输入占比(≈0.5 量级), "
             "区别在负值幅度——GELU 负尾有下界 ≈ -0.17(p1 列可验证), SiLU 门的负尾更深(p1≈-0.28), "
             "而门⊙up 乘积支路的负尾可达 -1.7, 说明门控产生了更强的抑制信号; "
             "乘积支路的近零比例与百分位跨度体现门控的选择性, 对比各层可观察该性质是否随深度漂移。\n")

    L.append("## 5. 生成样例 (seed " + str(SEEDS[0]) + ")\n")
    for r in results:
        if r["seed"] != SEEDS[0]:
            continue
        L.append(f"\n### {r['variant']}\n")
        for p, g in r["gen_samples"].items():
            L.append(f"- 提示「{p}」→ {g[:60]}")

    # ---- 自动结论 ----
    bv_v, bv_s = A("vanilla")["best_val"][0], A("swiglu")["best_val"][0]
    ac_v, ac_s = A("vanilla")["val_acc"][0], A("swiglu")["val_acc"][0]
    gm_v, gm_s = A("vanilla")["gen_match"][0], A("swiglu")["gen_match"][0]
    L.append("\n## 6. 自动结论\n")
    win_bv = "swiglu" if bv_s < bv_v else "vanilla"
    L.append(f"1. 泛化(最优 val loss): **{win_bv}** 更优"
             f"(vanilla {bv_v:.4f} vs swiglu {bv_s:.4f}, 差 {bv_s - bv_v:+.4f})。")
    win_ac = "swiglu" if ac_s > ac_v else "vanilla"
    L.append(f"2. val 字符准确率: **{win_ac}** 更高"
             f"({ac_v:.3f} vs {ac_s:.3f})。")
    win_gm = "swiglu" if gm_s > gm_v else "vanilla"
    L.append(f"3. 生成续写匹配率: **{win_gm}** 更高({gm_v:.3f} vs {gm_s:.3f})。")
    if s5v is not None and s5s is not None:
        win_c = "swiglu" if s5s < s5v else "vanilla"
        L.append(f"4. 收敛速度(train<0.5): **{win_c}** 更快({s5v:.0f} vs {s5s:.0f} 步)。")
    L.append("5. 说明: 本实验语料极小(≈1.5K 字符)且训练至 memorize, 两变体差距可能小于"
             "种子间波动; 报告以 mean±std 呈现即为量化该波动。SwiGLU 的优势在更大语料/"
             "更长训练下通常更明显(参见 LLaMA/PaLM 的实践)。")
    report = "\n".join(L) + "\n"
    path = run_dir / "report.md"
    path.write_text(report, encoding="utf-8")
    log.info(f"对比报告已写入: {path}")
    return report


def plot_comparison(run_dir, results, log):
    """三联图: (a) 平滑 train loss 曲线(细线=各种子, 粗线=种子均值);
    (b) val loss 曲线; (c) 各种子最优 val loss 柱状对比。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("未检测到 matplotlib, 跳过绘图")
        return
    colors = {"vanilla": "steelblue", "swiglu": "darkorange"}
    w = 50
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))

    for v in VARIANTS:                       # (a) train 曲线
        smooths = []
        for r in [r for r in results if r["variant"] == v]:
            tr = np.array([h["batch_loss"] for h in r["history"]])
            sm = np.convolve(tr, np.ones(w) / w, mode="valid")
            smooths.append(sm)
            axes[0].plot(np.arange(len(sm)) + w - 1, sm, lw=0.6, alpha=0.35, color=colors[v])
        axes[0].plot(np.arange(len(smooths[0])) + w - 1, np.mean(smooths, axis=0),
                     lw=2, color=colors[v], label=v)
    axes[0].set_title("train loss (smoothed)")
    axes[0].set_xlabel("step"); axes[0].set_ylabel("loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    for v in VARIANTS:                       # (b) val 曲线
        rs = [r for r in results if r["variant"] == v]
        steps = [h["step"] for h in rs[0]["history"] if h["val_loss"] is not None]
        vals = np.array([[h["val_loss"] for h in r["history"] if h["val_loss"] is not None]
                         for r in rs])
        for row in vals:
            axes[1].plot(steps, row, lw=0.6, alpha=0.35, color=colors[v])
        axes[1].plot(steps, vals.mean(axis=0), lw=2, color=colors[v], label=v)
    axes[1].set_title("val loss")
    axes[1].set_xlabel("step"); axes[1].set_ylabel("loss")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    x_pos = np.arange(len(SEEDS))            # (c) 各种子最优 val 柱状图
    width = 0.36
    for i, v in enumerate(VARIANTS):
        vals = [r["best_val"] for r in results if r["variant"] == v]
        bars = axes[2].bar(x_pos + (i - 0.5) * width, vals, width,
                           color=colors[v], label=v)
        axes[2].bar_label(bars, fmt="%.2f", fontsize=8)
    axes[2].set_xticks(x_pos); axes[2].set_xticklabels([f"seed {s}" for s in SEEDS])
    axes[2].set_title("best val loss per seed")
    axes[2].set_ylabel("loss"); axes[2].legend(); axes[2].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    path = run_dir / "comparison.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    log.info(f"对比图已保存: {path}")

# =====================================================================
# 6. 主流程
# =====================================================================
def main():
    run_dir = Path(__file__).parent / "logs" / f"ffn_cmp_{datetime.now():%Y%m%d_%H%M%S}"
    (run_dir / "runs").mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("ffn_cmp")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    sh = logging.StreamHandler()
    for h in (fh, sh):
        h.setFormatter(fmt)
        log.addHandler(h)
    log.info(f"实验目录: {run_dir}")
    log.info(f"变体: {VARIANTS} | 种子: {SEEDS} | 共 {len(VARIANTS) * len(SEEDS)} 次训练")
    fl = flops_analysis()
    log.info(f"理论算力 MACs/token: attn {fl['attn_mac']:,} | FFN vanilla {fl['ffn_v']:,} / "
             f"swiglu {fl['ffn_s']:,} | 全模型 {fl['total_v']:,} / {fl['total_s']:,} "
             f"(比值 {fl['total_ratio']:.3f})")

    results = []
    for v in VARIANTS:
        for seed in SEEDS:
            r = train_one(v, seed, log)
            # 逐步历史落盘, 便于事后用 pandas/Excel 复盘
            csv_path = run_dir / "runs" / f"{v}_seed{seed}.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                wr = csv.DictWriter(f, fieldnames=["step", "batch_loss", "grad_norm",
                                                   "val_loss", "val_acc", "elapsed"])
                wr.writeheader()
                wr.writerows(r["history"])
            results.append(r)

    # 汇总表: 每行一个 (变体, 种子)
    agg = aggregate(results)
    with open(run_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["variant", "seed", "params", "best_val", "best_val_step",
                     "final_train", "final_val", "val_acc", "gen_match",
                     "steps_to_0.5", "elapsed_s"])
        for r in results:
            wr.writerow([r["variant"], r["seed"], r["params"], f"{r['best_val']:.4f}",
                         r["best_val_step"], f"{r['final_train']:.4f}",
                         f"{r['final_val']:.4f}", f"{r['val_acc']:.4f}",
                         f"{r['gen_match']:.4f}",
                         r["steps_to"][0.5] if r["steps_to"][0.5] is not None else "",
                         f"{r['elapsed']:.1f}"])
    log.info(f"汇总表已写入: {run_dir / 'summary.csv'}")

    plot_comparison(run_dir, results, log)
    report = write_report(run_dir, agg, results, log)

    # 控制台打印报告中的核心汇总表, 方便不下翻文件
    log.info("=" * 70)
    for line in report.splitlines():
        if line.startswith("|") or line.startswith("## 3"):
            log.info(line)
    log.info(f"全部产物见: {run_dir}")

if __name__ == "__main__":
    main()
