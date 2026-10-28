# -*- coding: utf-8 -*-
"""
mini_transformer.py — 从零实现一个最小可运行的 Decoder-only Transformer(迷你 GPT)

结构对照(与之前讲解的一致):
    token → Embedding + 位置编码
      ↓
    ┌──────────── × N 层 TransformerBlock ────────────┐
    │ Pre-LN → 带因果掩码的多头自注意力 → 残差相加      │
    │ Pre-LN → 前馈网络 FFN          → 残差相加      │
    └─────────────────────────────────────────────────┘
      ↓
    LayerNorm → Linear(输出头, 与 embedding 共享权重) → Softmax → 下一个 token 概率

任务:字符级语言模型。在一段古典中文语料上训练,学会"给定前文,预测下一个字",
训练完成后可以根据提示词续写文本。CPU 上几分钟即可跑完。

运行:  python -X utf8 mini_transformer.py
每次运行会在 logs/run_时间戳/ 目录下归档三份产物, 便于事后分析:
  - train.log        完整运行日志(配置/参数表/每次评估指标/生成对照)
  - loss_history.csv 每一步的训练指标(step、batch loss、梯度范数、val loss、耗时)
  - loss_curve.png   train/val loss 曲线图(需安装 matplotlib, 无则自动跳过)
"""

import csv
import logging
import math
import os
import time
from datetime import datetime
from pathlib import Path

# Anaconda + PyTorch 在 Windows 上的常见问题: torch 和 matplotlib 各自携带一份
# OpenMP 运行时(libiomp5md.dll), 同时加载会报 "OMP: Error #15" 并中止进程。
# 必须在 import torch/matplotlib 之前设置该环境变量, 让两份 OpenMP 共存。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 0. 超参数配置(模型很小,保证 CPU 上能快速训练)
# =====================================================================
seed       = 1337     # 随机种子,保证结果可复现
block_size = 48       # 上下文长度:模型一次最多"看到"多少个字符
batch_size = 16       # 每步训练同时用多少个样本

n_layer    = 3        # TransformerBlock 堆叠层数 N
n_head     = 4        # 多头注意力的头数 h
d_model    = 128      # 每个 token 的隐藏向量维度 d
dropout    = 0.0      # 演示用微小语料,关掉 dropout 让拟合更干净(真实训练中取 0.1)

max_steps  = 2000     # 训练步数
eval_interval = 200   # 每隔多少步评估一次验证集 loss
lr         = 3e-4     # 学习率(Adam 的经典默认值)
device     = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(seed)

# =====================================================================
# 1. 准备语料 & 字符级 Tokenizer
# ---------------------------------------------------------------------
# 真正的 LLM 用 BPE 等子词分词,这里为了聚焦模型结构,用最简单的字符级:
#   每个汉字 = 一个 token,词表 = 语料中出现过的所有不重复字符。
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
chars = sorted(set(text))            # 语料中所有不重复的字符 → 词表
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}   # 字符 → 整数 id(编码)
itos = {i: ch for i, ch in enumerate(chars)}   # 整数 id → 字符(解码)
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]      # 90% 训练 / 10% 验证

# =====================================================================
# 2. 位置编码:正弦/余弦函数(原论文方案)
# ---------------------------------------------------------------------
#   PE(pos, 2i)   = sin(pos / 10000^(2i/d))
#   PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
# 注意力本身不知道 token 的先后顺序,必须把位置信息"加"进 embedding。
# 不同维度用不同频率的三角函数,相当于给每个位置一个独一无二的指纹。
# (现代 LLM 多用 RoPE 旋转位置编码,思想类似:把相对位置编进 Q·K 内积。)
# =====================================================================
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)                       # (max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()         # (max_len, 1)
        # div_term = 1 / 10000^(2i/d),用 exp(log) 计算避免大数溢出
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)   # 偶数维用 sin
        pe[:, 1::2] = torch.cos(pos * div)   # 奇数维用 cos
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d),非训练参数

    def forward(self, x):                    # x: (B, T, d_model)
        return x + self.pe[:, : x.size(1)]   # 直接相加

# =====================================================================
# 3. 子层一:带因果掩码的多头自注意力
# ---------------------------------------------------------------------
#   Attention(Q, K, V) = softmax(Q·Kᵀ / √d_head + mask) · V
#
#   Q(Query):我想找什么  K(Key):我能提供什么  V(Value):我实际携带的信息
#   因果掩码:把"未来位置"的得分设为 -inf,softmax 后权重为 0,
#            保证位置 i 只能看到 ≤ i 的 token(不偷看答案)。
#   多头:把 d_model 切成 n_head 份并行计算,让不同的头关注不同类型的关系。
# =====================================================================
class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head, self.d_head = n_head, d_model // n_head
        # 三个投影矩阵 W_Q / W_K / W_V(合并成一个 Linear 一次算完,更高效)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)          # 多头拼接后的输出投影 W_O
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x):                                # x: (B, T, d_model)
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(d_model, dim=2)      # 各 (B, T, C)
        # 拆成多头: (B, T, C) → (B, n_head, T, d_head),让每个头独立做注意力
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        # 注意力得分: Q·Kᵀ / √d_head  → (B, n_head, T, T)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        # 因果掩码:上三角(未来位置)填 -inf
        mask = torch.full((T, T), float("-inf"), device=x.device).triu(1)
        att = att + mask
        att = F.softmax(att, dim=-1)                     # 每行归一化成权重分布
        att = self.attn_drop(att)
        y = att @ v                                      # 加权求和 Value → (B, nh, T, dh)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # 拼接所有头 → (B, T, C)
        return self.resid_drop(self.proj(y))

# =====================================================================
# 4. 子层二:前馈网络 FFN(逐位置的知识加工)
# ---------------------------------------------------------------------
#   FFN(x) = W2 · GELU(W1·x + b1) + b2,  维度变化 d → 4d → d
#   注意力负责 token 之间"交换信息",FFN 对每个位置独立做非线性变换,
#   负责"消化信息",模型的世界知识主要存储在这里(类似键值存储器)。
# =====================================================================
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),   # 升维 4 倍
            nn.GELU(),                         # 非线性激活(GPT 系列用 GELU)
            nn.Linear(4 * d_model, d_model),   # 降回原维度
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# =====================================================================
# 5. TransformerBlock:一个完整的"编码器层/解码器层"
# ---------------------------------------------------------------------
#   采用现代主流的 Pre-LN 写法: x = x + Sublayer(LayerNorm(x))
#   残差连接: 提供梯度直通路径,让每个子层只学"增量修正"
#   LayerNorm: 稳定每层输入的数值分布,加速收敛
# =====================================================================
class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))    # 子层1: 信息交换(注意力)+ 残差
        x = x + self.ffn(self.ln2(x))     # 子层2: 信息加工(FFN)+ 残差
        return x

# =====================================================================
# 6. 完整模型:迷你 GPT
# =====================================================================
class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)   # token id → 向量
        self.pos_emb = SinusoidalPositionalEncoding(d_model, block_size)
        self.blocks = nn.ModuleList(TransformerBlock() for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(d_model)                  # 最后一层归一化
        self.head = nn.Linear(d_model, vocab_size, bias=False)  # 映射到词表 logits
        # 权重共享: 输出头复用 token embedding 的权重,减少参数、提升效果
        self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)
        # 注意: head 与 tok_emb 共享同一权重张量, apply() 按注册顺序遍历时
        # head(Linear, std=0.02) 会覆盖 tok_emb(Embedding) 的初始化,
        # 因此最后重新按 embedding 的标准设置一次共享权重。
        self.tok_emb.weight.data.normal_(mean=0.0, std=d_model ** -0.5)

    def _init_weights(self, module):
        # 初始化对训练影响很大,这里是一个很好的教学点:
        #  - nn.Embedding 默认 N(0,1) 太大: 共享权重后 logits=x·Eᵀ 的方差 ≈ d_model,
        #    初始 loss 会爆炸(远大于理论值 ln(词表大小));
        #  - 全部用 GPT-2 的 N(0,0.02) 又太小: logits 起步过小,收敛很慢;
        #  - Xavier 风格 std=1/√d_model 恰好让初始 logits ≈ N(0,1),
        #    初始 loss ≈ ln(词表大小),梯度也足够大。
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=d_model ** -0.5)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, idx, targets=None):
        # idx: (B, T) 的 token id 序列
        x = self.tok_emb(idx)                # (B, T) → (B, T, d_model)
        x = self.pos_emb(x)                  # 注入位置信息
        for blk in self.blocks:              # 依次通过 N 层 TransformerBlock
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)                # (B, T, vocab_size) 每个位置的下一字预测
        loss = None
        if targets is not None:
            # 交叉熵损失: 比较 logits 与真实的"下一个字符"
            loss = F.cross_entropy(
                logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=10):
        """自回归生成: 每次预测下一个 token, 把它拼回输入, 循环生成。"""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]          # 上下文超长时截断
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # 只取最后一个位置; 温度控制随机性
            if top_k:                                # top-k 采样: 只在概率最高的 k 个字里选
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # 按概率抽样
            idx = torch.cat([idx, next_id], dim=1)
        return idx

# =====================================================================
# 7. 训练辅助: 取一个 batch、评估 loss
# =====================================================================
def get_batch(split):
    d = train_data if split == "train" else val_data
    # 若该 split 长度不足一个完整上下文,则从有效范围内取样(保护性写法)
    hi = max(1, len(d) - block_size - 1)
    ix = torch.randint(hi, (batch_size,))
    # x 是输入序列, y 是"x 右移一位"——每个位置的目标就是它的下一个字符
    x = torch.stack([d[i : i + block_size] for i in ix])
    y = torch.stack([d[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
    model.eval()                       # 评估模式: 关闭 dropout
    out = {}
    for split in ("train", "val"):
        losses = [model(*get_batch(split))[1].item() for _ in range(30)]
        out[split] = sum(losses) / len(losses)
    model.train()
    return out

# =====================================================================
# 8. 日志系统
# ---------------------------------------------------------------------
# 同时输出到 控制台 + logs/run_时间戳/train.log, 并附带:
#   loss_history.csv — 每一步的 batch loss / 梯度范数 / val loss / 耗时
#   loss_curve.png   — loss 曲线图(train 原始 + 平滑, val 单独一条)
# =====================================================================
def setup_logging():
    run_dir = Path(__file__).parent / "logs" / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("mini_transformer")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    sh = logging.StreamHandler()
    for h in (fh, sh):
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger, run_dir


def plot_loss(history, run_dir, log):
    """把 loss 曲线画成 PNG。图中文字用英文, 避免 matplotlib 中文字体缺字问题。"""
    try:
        import matplotlib
        matplotlib.use("Agg")                 # 无界面后端, 只存文件不弹窗
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("未检测到 matplotlib, 跳过绘制 loss 曲线")
        return
    steps = [r["step"] for r in history]
    train = [r["batch_loss"] for r in history]
    # 滑动平均平滑 train 曲线(单 batch loss 噪声大, 平滑后趋势更清晰)
    w = 50
    smooth = [sum(train[max(0, i - w + 1) : i + 1]) / min(w, i + 1)
              for i in range(len(train))]
    val_s = [r["step"] for r in history if r["val_loss"] is not None]
    val_v = [r["val_loss"] for r in history if r["val_loss"] is not None]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, train, lw=0.6, alpha=0.4, color="steelblue", label="train (per-batch)")
    ax.plot(steps, smooth, lw=2, color="steelblue", label=f"train (smooth {w})")
    ax.plot(val_s, val_v, "o-", ms=4, lw=1.5, color="crimson", label="val")
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("Mini-GPT training curve")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = run_dir / "loss_curve.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    log.info(f"loss 曲线已保存: {path}")

# =====================================================================
# 9. 主流程: 建模 → 训练 → 生成(全程写日志)
# =====================================================================
def main():
    log, run_dir = setup_logging()
    log.info(f"本次运行目录: {run_dir}")

    # ---------- 数据与配置摘要 ----------
    log.info("=" * 64)
    log.info(f"语料: {len(text)} 字符 | 词表: {vocab_size} | "
             f"train {len(train_data)} tokens / val {len(val_data)} tokens")
    log.info(f"模型: n_layer={n_layer}, n_head={n_head}, d_model={d_model}, "
             f"block_size={block_size}, dropout={dropout}")
    log.info(f"训练: max_steps={max_steps}, batch_size={batch_size}, lr={lr}, "
             f"seed={seed}, device={device}")

    # ---------- 模型与参数量明细 ----------
    model = MiniGPT().to(device)
    # model.parameters() 会自动去重共享权重, 所以 total < 各模块简单相加
    total = sum(p.numel() for p in model.parameters())
    log.info(f"总参数量(去重后): {total:,} ({total / 1e3:.1f}K), 各模块明细:")
    for name, mod in model.named_children():
        n = sum(p.numel() for p in mod.parameters())
        if n:
            note = "  ← 与 tok_emb 共享权重, 总数中只计一次" if name == "head" else ""
            log.info(f"    {name:<10} {type(mod).__name__:<32} {n:>9,} params{note}")

    # ---------- 训练前生成(对照组: 未学习的模型应输出乱码) ----------
    model.eval()
    prompt = "学而时习之,"
    ctx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    out = decode(model.generate(ctx, max_new_tokens=40)[0].tolist())
    log.info(f"[训练前生成] 提示「{prompt}」→ 续写: {out}")
    model.train()

    # ---------- 训练循环 ----------
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history = []                       # 每步指标, 训练结束写入 CSV
    t0 = prev_t = time.time()
    grad_norm = 0.0
    best_val, final_val = float("inf"), float("nan")
    log.info("开始训练...")
    for step in range(max_steps + 1):
        val_loss = None
        if step % eval_interval == 0:  # 定期评估: train/val loss + 吞吐量
            losses = estimate_loss(model)
            val_loss = losses["val"]
            best_val = min(best_val, val_loss)
            final_val = val_loss
            now = time.time()
            tps = eval_interval * batch_size * block_size / (now - prev_t) if step else 0.0
            prev_t = now
            log.info(f"step {step:5d} | train {losses['train']:.4f} | val {val_loss:.4f} | "
                     f"grad_norm {grad_norm:.3f} | {tps:,.0f} tok/s | 已用 {now - t0:.1f}s")
        x, y = get_batch("train")
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()                             # 反向传播
        # clip_grad_norm_ 的返回值 = 裁剪前的全局梯度范数, 是重要的健康指标:
        #   持续 >1 说明梯度常被裁剪(可考虑降 lr); 突然飙升是训练不稳定的信号
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        optimizer.step()                            # 更新参数
        history.append({"step": step,
                        "batch_loss": round(loss.item(), 4),
                        "grad_norm": round(grad_norm, 4),
                        "val_loss": val_loss,
                        "elapsed": round(time.time() - t0, 2)})

    log.info(f"训练完成, 总用时 {time.time() - t0:.1f}s")
    log.info(f"最优 val loss {best_val:.4f} | 最终 val loss {final_val:.4f} | "
             f"最终 train loss {history[-1]['batch_loss']:.4f}")
    if final_val > best_val * 1.2:
        log.info("观察: val loss 明显高于最优点 → 典型过拟合(语料太小, 模型背下了训练集)")

    # ---------- 落盘: CSV + 曲线图 ----------
    csv_path = run_dir / "loss_history.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "batch_loss", "grad_norm",
                                               "val_loss", "elapsed"])
        writer.writeheader()
        writer.writerows(history)
    log.info(f"逐步指标已写入: {csv_path}")
    plot_loss(history, run_dir, log)

    # ---------- 训练后生成(实验组) ----------
    model.eval()
    log.info("[训练后生成] (temperature=0.8, top_k=10)")
    for prompt in ["学而时习之,", "故天将降大任于是人也,", "君子喻于义,"]:
        ctx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
        out = decode(model.generate(ctx, max_new_tokens=50)[0].tolist())
        log.info(f"    提示「{prompt}」")
        log.info(f"    续写: {out}")
    log.info(f"全部产物见: {run_dir}")

if __name__ == "__main__":
    main()
