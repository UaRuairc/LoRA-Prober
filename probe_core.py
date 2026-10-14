"""Model-side logic for the LoRA Prober (Qwen-Image 2.1, Krea 2 and MiniMax H3).

Nothing in this file imports ComfyUI, so it can be tested on its own:
model profiles, LoRA key parsing, weight-change norms, ablation variants, metrics,
the suggested subset and the ai-toolkit config.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import defaultdict, namedtuple
from dataclasses import dataclass, field

import numpy as np
import torch

NAN = float("nan")


def finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x)) if finite(x) else lo


# ---------------------------------------------------------------------- profiles


@dataclass(frozen=True)
class Layer:
    short: str  # name used in the probe ("q", "mlp_up", ...)
    path: str  # ComfyUI / ai-toolkit name inside a block
    aliases: tuple  # the same layer under other trainers' names
    shape: tuple  # (in, out) features in the main stack, for LoRA size estimates
    ignore: str  # ai-toolkit ignore_if_contains pattern for this layer alone
    default: bool = True  # trained by ai-toolkit's default config for the model (MiniMax H3 leaves adaln out)
    optional: bool = False  # only some checkpoints of the model have it (MiniMax H3's to_gate_compress: FastH3)


@dataclass(frozen=True)
class SideGroup:
    """Blocks outside the main stack that LoRAs also train (Krea 2's text fusion, MiniMax H3's token refiner)."""

    key: str
    label: str
    blocks: tuple  # ComfyUI / ai-toolkit block paths, in column order
    labels: tuple  # short column labels
    regex: str  # LoRA key -> (kind, index, rest)
    offsets: dict  # kind -> column offset
    shapes: dict  # short -> (in, out); layers missing here do not exist in these blocks
    pattern: str  # ai-toolkit only_if_contains pattern selecting all of it


@dataclass(frozen=True)
class Profile:
    key: str
    label: str
    comfy_classes: tuple  # ComfyUI model_base class names
    block_attr: str  # block list on the ComfyUI diffusion model
    default_blocks: int
    layers: tuple
    sets: dict  # "attn" / "mlp" / "kv" -> shorts
    group_ignores: dict  # "attn" / "mlp" -> ai-toolkit pattern for the whole set
    block_regex: str  # LoRA key -> (block index, rest)
    block_pattern: str  # ai-toolkit only_if_contains pattern for one block
    all_blocks_pattern: str  # ai-toolkit pattern for every main block, or "" if not needed
    canonical_block: str  # ai-toolkit module path of block b
    non_block_linears: tuple
    transformer_block_names: tuple  # what ai-toolkit's transformer_only keeps
    side: SideGroup = None
    arch: str = ""  # ai-toolkit model arch
    encode: str = "plain"  # how prompts are encoded: "qwen21" or "plain" (CLIPTextEncode)
    rank: int = 32  # rank of the default LoRA in ai-toolkit's preset
    alpha_note: str = ""  # comment on linear_alpha in the suggested config
    always_ignore: tuple = ()  # ai-toolkit ignore_if_contains patterns the default config always has
    video: bool = False  # renders are video clips (+ audio)

    @property
    def order(self):
        return [l.short for l in self.layers]

    @property
    def default_order(self):
        """Layer types ai-toolkit's default config for this model trains."""
        return [l.short for l in self.layers if l.default]

    def layer(self, short):
        return next(l for l in self.layers if l.short == short)

    def named_sets(self):
        """(set key, label) for the layer sets this model has, e.g. attention, MLP, k,v."""
        return [(k, lab) for k, lab in (("attn", "attention"), ("mlp", "MLP"), ("kv", "k,v")) if k in self.sets]


QWEN21 = Profile(
    key="qwen21",
    label="Qwen-Image 2.1",
    comfy_classes=("QwenImage21",),
    block_attr="transformer_blocks",
    default_blocks=32,
    layers=(
        Layer("q", "attn.to_q", (), (4096, 4096), "attn.to_q"),
        Layer("k", "attn.to_k", (), (4096, 4096), "attn.to_k"),
        Layer("v", "attn.to_v", (), (4096, 4096), "attn.to_v"),
        Layer("o", "attn.to_out.0", ("attn.to_out",), (4096, 4096), "attn.to_out"),
        # diffusers splits the fused SwiGLU input into gate_layer + proj
        Layer("mlp_in", "img_mlp.gate_up", ("img_mlp.gate_layer", "img_mlp.proj"), (4096, 24576), "img_mlp.gate_up"),
        Layer("mlp_out", "img_mlp.out", (), (12288, 4096), "img_mlp.out"),
    ),
    sets={"attn": ("q", "k", "v", "o"), "mlp": ("mlp_in", "mlp_out"), "kv": ("k", "v")},
    group_ignores={"attn": ".attn.", "mlp": "img_mlp."},
    block_regex=r"(?:^|[._])transformer_blocks[._](\d+)[._](.+)$",
    block_pattern="transformer_blocks.{b}.",
    all_blocks_pattern="",
    canonical_block="transformer_blocks.{b}",
    non_block_linears=(
        "time_text_embed.timestep_embedder.linear_1",
        "time_text_embed.timestep_embedder.linear_2",
        "txt_in.in_layer",
        "txt_in.out_layer",
        "img_in",
        "modulation.1",
        "norm_out.linear",
        "proj_out",
    ),
    transformer_block_names=("transformer_blocks",),
    arch="qwen_image_2",
    encode="qwen21",
    alpha_note="ignored for this model's LoRA (alpha = rank)",
)

KREA2 = Profile(
    key="krea2",
    label="Krea 2",
    comfy_classes=("Krea2",),
    block_attr="blocks",
    default_blocks=28,
    layers=(
        Layer("q", "attn.wq", ("attn.to_q",), (6144, 6144), ".attn.wq"),
        Layer("k", "attn.wk", ("attn.to_k",), (6144, 1536), ".attn.wk"),
        Layer("v", "attn.wv", ("attn.to_v",), (6144, 1536), ".attn.wv"),
        Layer("gate", "attn.gate", ("attn.to_gate",), (6144, 6144), ".attn.gate"),
        Layer("o", "attn.wo", ("attn.to_out.0", "attn.to_out"), (6144, 6144), ".attn.wo"),
        Layer("mlp_gate", "mlp.gate", ("ff.gate",), (6144, 16384), ".mlp.gate"),
        Layer("mlp_up", "mlp.up", ("ff.up",), (6144, 16384), ".mlp.up"),
        Layer("mlp_down", "mlp.down", ("ff.down",), (16384, 6144), ".mlp.down"),
    ),
    sets={"attn": ("q", "k", "v", "gate", "o"), "mlp": ("mlp_gate", "mlp_up", "mlp_down"), "kv": ("k", "v")},
    group_ignores={"attn": ".attn.", "mlp": ".mlp."},
    block_regex=r"(?:^|[._])(?:transformer_)?blocks[._](\d+)[._](.+)$",
    # with the "transformer." prefix so it cannot match txtfusion.layerwise_blocks.N
    block_pattern="transformer.blocks.{b}.",
    all_blocks_pattern="transformer.blocks.",
    canonical_block="blocks.{b}",
    non_block_linears=("txtfusion.projector", "first", "tmlp.0", "tmlp.2", "txtmlp.1", "txtmlp.3",
                       "last.linear", "tproj.1"),
    transformer_block_names=("blocks",),  # a substring test, so it also keeps the text-fusion blocks
    side=SideGroup(
        key="txtfusion",
        label="text fusion",
        blocks=("txtfusion.layerwise_blocks.0", "txtfusion.layerwise_blocks.1",
                "txtfusion.refiner_blocks.0", "txtfusion.refiner_blocks.1"),
        labels=("L0", "L1", "R0", "R1"),
        regex=r"(?:txtfusion|text_fusion)[._](layerwise|refiner)_blocks[._](\d+)[._](.+)$",
        offsets={"layerwise": 0, "refiner": 2},
        shapes={"q": (2560, 2560), "k": (2560, 2560), "v": (2560, 2560), "gate": (2560, 2560),
                "o": (2560, 2560), "mlp_gate": (2560, 6912), "mlp_up": (2560, 6912), "mlp_down": (6912, 2560)},
        pattern="txtfusion.",
    ),
    arch="krea2",
    encode="plain",
    alpha_note="ignored for this model's LoRA (alpha = rank)",
)

# MiniMax H3 (33B audio-video DiT): 50 blocks, hidden 5376, 56 heads of 128, SwiGLU ffn 14336, text through a
# 2-block token refiner. The Comfy-Org "pruned" checkpoints keep every block weight but feed the adaln linears an
# 8-dim time table instead of the 2688-dim time embedding, so adaln LoRAs only fit the checkpoint they were
# trained on; ai-toolkit's MiniMax-H3 preset leaves adaln out (ignore_if_contains: adaln_proj).
# to_gate_compress only exists in the VSA-trained FastH3 checkpoints.
_H3_SHAPES = {"qkv": (5376, 21504), "o": (7168, 5376), "mlp_in": (5376, 28672), "mlp_out": (14336, 5376)}
MINIMAX_H3 = Profile(
    key="minimax_h3",
    label="MiniMax H3",
    comfy_classes=("MiniMaxH3",),
    block_attr="blocks",
    default_blocks=50,
    layers=(
        Layer("qkv", "attn.qkv_proj", ("attn.to_qkv",), _H3_SHAPES["qkv"], ".attn.qkv_proj"),
        Layer("o", "attn.out_proj", ("attn.to_out.0", "attn.to_out"), _H3_SHAPES["o"], ".attn.out_proj"),
        Layer("gate_compress", "attn.to_gate_compress", (), (5376, 7168), ".attn.to_gate_compress", optional=True),
        Layer("mlp_in", "mlp.fc1", (), _H3_SHAPES["mlp_in"], ".mlp.fc1"),
        Layer("mlp_out", "mlp.fc2", (), _H3_SHAPES["mlp_out"], ".mlp.fc2"),
        Layer("adaln", "adaln_proj.linear", (), (2688, 96768), "adaln_proj", default=False),
    ),
    sets={"attn": ("qkv", "o", "gate_compress"), "mlp": ("mlp_in", "mlp_out")},
    group_ignores={"attn": ".attn.", "mlp": ".mlp."},
    block_regex=r"(?:^|[._])(?:transformer_)?blocks[._](\d+)[._](.+)$",
    block_pattern="transformer.blocks.{b}.",
    all_blocks_pattern="transformer.blocks.",
    canonical_block="blocks.{b}",
    non_block_linears=("video_patch_proj", "audio_patch_proj", "condition_proj", "time_embedder.proj_in",
                       "time_embedder.proj_out", "final_layer.adaln_proj.linear", "final_layer.video_out",
                       "final_layer.audio_out"),
    transformer_block_names=("blocks",),  # a substring test, so it also keeps token_refiner.blocks.N
    side=SideGroup(
        key="token_refiner",
        label="token refiner",
        blocks=("token_refiner.blocks.0", "token_refiner.blocks.1"),
        labels=("R0", "R1"),
        regex=r"(token_refiner)[._]blocks[._](\d+)[._](.+)$",
        offsets={"token_refiner": 0},
        shapes=dict(_H3_SHAPES),
        pattern="token_refiner.",
    ),
    arch="minimax_h3",
    encode="plain",
    rank=16,
    always_ignore=("adaln_proj",),
    video=True,
)

PROFILES = {p.key: p for p in (QWEN21, KREA2, MINIMAX_H3)}
FAMILY_CHOICES = ["auto"] + [p.label for p in PROFILES.values()]


def profile_by_label(label):
    for p in PROFILES.values():
        if p.label == label or p.key == label:
            return p
    return None


def profile_for_model_class(name):
    for p in PROFILES.values():
        if name in p.comfy_classes:
            return p
    return None


# --------------------------------------------------------------------------- keys

Key = namedtuple("Key", "group block short path suffix")  # group: "main", a side key, or None

_REGEX_CACHE = {}


def _re(pattern):
    if pattern not in _REGEX_CACHE:
        _REGEX_CACHE[pattern] = re.compile(pattern)
    return _REGEX_CACHE[pattern]


def _layer_forms(profile):
    key = ("forms", profile.key)
    if key not in _REGEX_CACHE:
        forms = []
        for l in profile.layers:
            for path in (l.path,) + tuple(l.aliases):
                forms.append((path, l.short, path))
                forms.append((path.replace(".", "_"), l.short, path))
        forms.sort(key=lambda f: -len(f[0]))  # "attn.to_out.0" before "attn.to_out"
        _REGEX_CACHE[key] = forms
    return _REGEX_CACHE[key]


def _match_layer(profile, rest):
    for form, short, path in _layer_forms(profile):
        if rest.startswith(form + "."):
            return short, path, rest[len(form) + 1 :]
    return None, None, rest


def parse_key(profile: Profile, name: str) -> Key:
    """Tensor name -> (group, block, layer short name, layer path, rest of the name).

    Handles ai-toolkit / ComfyUI names ("diffusion_model.blocks.3.attn.wq.lora_A.weight"),
    diffusers / PEFT names ("transformer.transformer_blocks.3.attn.to_q...") and kohya
    names ("lora_unet_blocks_3_attn_wq.lora_down.weight").
    """
    if profile.side is not None:
        m = _re(profile.side.regex).search(name)
        if m:
            block = profile.side.offsets[m.group(1)] + int(m.group(2))
            short, path, suffix = _match_layer(profile, m.group(3))
            return Key(profile.side.key, block, short, path, suffix)
    m = _re(profile.block_regex).search(name)
    if m is None:
        return Key(None, None, None, None, name)
    short, path, suffix = _match_layer(profile, m.group(2))
    return Key("main", int(m.group(1)), short, path, suffix)


def detect_profile(sd_keys):
    """Profile whose layer names match the most LoRA tensors (None if nothing matches)."""
    best, best_n = None, 0
    for p in PROFILES.values():
        n = sum(1 for k in sd_keys if parse_key(p, k).short is not None)
        if n > best_n:
            best, best_n = p, n
    return best


def recognised_share(profile, sd_keys):
    keys = list(sd_keys)
    return sum(1 for k in keys if parse_key(profile, k).short is not None) / max(1, len(keys))


# ------------------------------------------------------------------ selections


@dataclass
class Selection:
    blocks: list  # main-stack block indices
    modules: list  # layer shorts
    side: bool  # include the side group (Krea 2 text fusion)


def parse_selection(profile, spec, nb):
    """Blocks spec -> (main blocks, include side). Accepts all, main/dit, txt, ranges like 8-15."""
    text = str(spec if spec is not None else "").strip().lower()
    if text in ("", "all"):
        return list(range(nb)), profile.side is not None
    out, side = set(), False
    for part in text.replace(" ", "").split(","):
        if not part:
            continue
        if part in ("all",):
            out.update(range(nb))
            side = profile.side is not None
        elif part in ("main", "dit"):
            out.update(range(nb))
        elif part in ("txt", "text", "txtfusion", "refiner", "token_refiner"):
            if profile.side is None:
                raise ValueError(f"{profile.label} has no separate text blocks ('{part}')")
            side = True
        else:
            try:
                if "-" in part:
                    a, b = (int(x) for x in part.split("-", 1))
                    out.update(range(min(a, b), max(a, b) + 1))
                else:
                    out.add(int(part))
            except ValueError:
                raise ValueError(f"can't read blocks {part!r}: use numbers and ranges like 12 or 8-15, 'all'"
                                 + (f", or 'txt' for the {profile.side.label}" if profile.side is not None else "")) from None
    bad = sorted(b for b in out if not 0 <= b < nb)
    if bad:
        raise ValueError(f"block index out of range 0-{nb - 1}: {bad}")
    return sorted(out), side


def parse_blocks(spec, nb: int, profile: Profile = QWEN21) -> list:
    return parse_selection(profile, spec, nb)[0]


def parse_modules(profile, spec) -> list:
    if spec is None or str(spec).strip().lower() in ("", "all"):
        return list(profile.order)
    picked = set()
    for part in str(spec).split(","):
        part = part.strip().lower()
        if not part:
            continue
        if part in profile.sets:
            picked.update(profile.sets[part])
        elif part in profile.order:
            picked.add(part)
        else:
            sets = ", ".join(k for k, _ in profile.named_sets())
            raise ValueError(f"unknown layer '{part}' for {profile.label} "
                             f"(use {', '.join(profile.order)}, {sets} or all)")
    return [m for m in profile.order if m in picked]


_EXCEPT_RE = re.compile(r"(?:^|\s+)except(?:\s+|$)", re.I)


def _parse_parts(profile, text, nb) -> list:
    if not text.strip():  # nothing written: every block, every layer
        blocks, side = parse_selection(profile, "", nb)
        return [Selection(blocks, list(profile.order), side)]
    parts = []
    for chunk in text.split("+"):
        chunk = chunk.strip()
        if not chunk:
            raise ValueError(f"empty part in {text!r} (parts are joined with +)")
        bspec, _, mspec = chunk.partition(":")
        blocks, side = parse_selection(profile, bspec, nb)
        parts.append(Selection(blocks, parse_modules(profile, mspec), side))
    return parts


def parse_variant_selection(profile, spec, nb):
    """A selection written as text -> (include, exclude), lists of Selection; include None = everything.

    BLOCKS[:LAYERS]        e.g. 12:mlp (only block 12's MLP), 8-15 (blocks 8-15, all layers)
    PART + PART            union, e.g. 8-15:mlp + 12:attn
    -PART[+PART]           everything except, e.g. -12:mlp (the whole LoRA without block 12's MLP)
    PARTS except PARTS     e.g. 8-23 except 12:mlp
    """
    text = str(spec if spec is not None else "").strip()
    pieces = _EXCEPT_RE.split(text, maxsplit=2)
    if len(pieces) > 2:
        raise ValueError(f"use 'except' once in {text!r}")
    inc, exc = pieces[0].strip(), (pieces[1].strip() if len(pieces) == 2 else "")
    if inc.startswith("-"):
        if len(pieces) == 2:
            raise ValueError(f"write either -X or A except X, not both: {text!r}")
        inc, exc = "", inc[1:].strip()
        if not exc:
            raise ValueError(f"nothing after '-' in {text!r}")
    if len(pieces) == 2 and not exc:
        raise ValueError(f"nothing after 'except' in {text!r}")
    exclude = _parse_parts(profile, exc, nb) if exc else []
    if exclude and inc.lower().replace(" ", "") in ("", "all", "all:all"):
        return None, exclude  # the whole LoRA minus the excluded parts
    return _parse_parts(profile, inc, nb), exclude


def describe_part(profile, nb, sel, show_side=True) -> str:
    side = bool(sel.side) and profile.side is not None and show_side
    if not sel.blocks and side:
        where = profile.side.label
    elif side and len(sel.blocks) == nb:
        where = "all blocks"
    elif side:
        where = f"{describe_blocks(sel.blocks, nb)} + {profile.side.label}"
    else:
        where = describe_blocks(sel.blocks, nb)
    return f"{where} {describe_modules(profile, sel.modules)}"


def compact_ranges(nums) -> str:
    nums = sorted(set(nums))
    if not nums:
        return "none"
    out, start, prev = [], nums[0], nums[0]
    for n in nums[1:] + [None]:
        if n is not None and n == prev + 1:
            prev = n
            continue
        out.append(str(start) if start == prev else f"{start}-{prev}")
        if n is not None:
            start = prev = n
    return ",".join(out)


def describe_modules(profile, mods) -> str:
    s = set(mods)
    if s == set(profile.order):
        return "all layers"
    for name, label in profile.named_sets():
        if s == set(profile.sets[name]):
            return label
    if s == set(profile.default_order):
        off = [l.short for l in profile.layers if not l.default]
        return "all layers except " + ",".join(off)
    return ",".join(m for m in profile.order if m in s) or "none"


def describe_blocks(blocks, nb: int) -> str:
    if len(blocks) == nb:
        return "all blocks"
    if not blocks:
        return "no blocks"
    if len(blocks) == 1:
        return f"block {blocks[0]}"
    return "blocks " + compact_ranges(blocks)


def describe_selection(profile, nb, blocks, side, modules, show_side=True) -> str:
    if profile.side is None or not show_side:
        where = describe_blocks(blocks, nb)
    elif not blocks and side:
        where = f"{profile.side.label} only"
    elif side:
        where = describe_blocks(blocks, nb) + f" + {profile.side.label}"
    else:
        where = describe_blocks(blocks, nb) + f", no {profile.side.label}"
    return f"{where}, {describe_modules(profile, modules)}"


# ------------------------------------------------------------- LoRA structure


@dataclass
class LoraInfo:
    profile: Profile
    modules: dict  # (group, block, short, path) -> {suffix: tensor name}
    stray: list  # inside a block, but not a known layer
    other: list  # outside the blocks
    sq: dict  # (group, block, short) -> squared Frobenius norm of the weight change
    kinds: set
    ranks: set
    notes: list

    @property
    def blocks(self):
        return sorted({b for g, b, _ in self.sq if g == "main"})

    @property
    def has_side(self):
        return any(g != "main" for g, _, _ in self.sq)

    def energy(self, group, block):
        return sum(v for (g, b, _), v in self.sq.items() if g == group and b == block and finite(v))

    def block_energy(self, nb):
        return [self.energy("main", b) for b in range(nb)]

    def side_energy(self):
        if self.profile.side is None:
            return []
        return [self.energy(self.profile.side.key, i) for i in range(len(self.profile.side.blocks))]

    def module_energy(self, short):
        return sum(v for (_, _, s), v in self.sq.items() if s == short and finite(v))

    def total_energy(self):
        return sum(v for v in self.sq.values() if finite(v)) or 1.0


def _get(parts, *names):
    for n in names:
        if n in parts:
            return parts[n]
    return None


def _t(sd, key):
    return sd[key].detach().to("cpu", torch.float64)


def _module_delta_sq(sd, parts, notes):
    """Squared Frobenius norm of the weight change one layer's adapter applies."""
    alpha_key = _get(parts, "alpha")
    alpha = float(_t(sd, alpha_key).reshape(-1)[0]) if alpha_key else None

    a_key = _get(parts, "lora_A.weight", "lora_down.weight")
    b_key = _get(parts, "lora_B.weight", "lora_up.weight")
    if a_key and b_key:
        A, B = _t(sd, a_key), _t(sd, b_key)
        r = A.shape[0]
        A = A.reshape(r, -1)
        B = B.reshape(B.shape[0], -1)
        scale = alpha / r if alpha is not None else 1.0
        if "dora_scale" in parts:
            notes.add("DoRA magnitude vectors are ignored; norms cover the low-rank part only.")
        sq = float(((B.T @ B) * (A @ A.T)).sum())  # ||B A||^2 on r x r matrices
        return max(sq, 0.0) * scale * scale, "lora", r

    def factor(full, fa, fb):
        if full in parts:
            return _t(sd, parts[full]), None
        if fa in parts and fb in parts:
            a, b = _t(sd, parts[fa]), _t(sd, parts[fb])
            return a.reshape(a.shape[0], -1) @ b.reshape(b.shape[0], -1), a.shape[1]
        return None, None

    w1, r1 = factor("lokr_w1", "lokr_w1_a", "lokr_w1_b")
    w2, r2 = factor("lokr_w2", "lokr_w2_a", "lokr_w2_b")
    if w1 is not None and w2 is not None:
        r = r2 or r1
        scale = alpha / r if (alpha is not None and r) else 1.0
        notes.add("LoKr: scale taken as alpha/rank for factorised weights, 1 otherwise.")
        return float((w1 * w1).sum()) * float((w2 * w2).sum()) * scale * scale, "lokr", r

    d = _get(parts, "diff")
    if d:
        return float((_t(sd, d) ** 2).sum()), "full", None

    notes.add(f"unrecognised adapter tensors skipped: {sorted(parts)}")
    return NAN, "unknown", None


def analyze_lora(profile: Profile, sd: dict) -> LoraInfo:
    modules = defaultdict(dict)
    stray, other = [], []
    for name in sd:
        k = parse_key(profile, name)
        if k.group is None:
            other.append(name)
        elif k.short is None:
            stray.append(name)
        else:
            modules[(k.group, k.block, k.short, k.path)][k.suffix] = name
    notes = set()
    sq = defaultdict(float)
    kinds, ranks = set(), set()
    for (g, b, short, _path), parts in sorted(modules.items()):
        v, kind, r = _module_delta_sq(sd, parts, notes)
        sq[(g, b, short)] += v
        kinds.add(kind)
        if r:
            ranks.add(r)
    return LoraInfo(profile, dict(modules), stray, other, dict(sq), kinds, ranks, sorted(notes))


# -------------------------------------------------------------------- variants


@dataclass
class Variant:
    name: str
    kind: str  # base | full | drop | keep | modules | side | custom | suggest
    blocks: list
    modules: list
    side: bool = False  # include the side group (Krea 2 text fusion)
    group: list = None
    keep_other: bool = False
    keys: list = field(default_factory=list)
    param_share: float = NAN
    include: list = None  # Selection parts, when the variant is not one blocks x layers rectangle
    exclude: list = field(default_factory=list)  # Selection parts removed from include (None = everything)
    spec: str = ""  # the text a custom variant was written as

    @property
    def is_parts(self):
        return self.include is not None or bool(self.exclude)

    def describe(self, profile, nb, show_side=True):
        if self.kind == "base":
            return "no LoRA"
        if self.is_parts:
            inc = "everything" if self.include is None else \
                " + ".join(describe_part(profile, nb, p, show_side) for p in self.include)
            if not self.exclude:
                return inc
            exc = " + ".join(describe_part(profile, nb, p, show_side) for p in self.exclude)
            return f"all except {exc}" if self.include is None else f"{inc} except {exc}"
        return describe_selection(profile, nb, self.blocks, self.side, self.modules, show_side)


def group_tag(group) -> str:
    return f"b{group[0]:02d}-{group[-1]:02d}"


SWEEPS = ["drop + keep groups", "keep-only groups", "drop-only groups", "no block groups"]


def build_variants(profile, nb, sweep, group_size, module_variants, custom_text="", has_side=False) -> list:
    all_blocks = list(range(nb))
    order = list(profile.order)
    side_all = profile.side is not None
    out = [
        Variant("base", "base", [], []),
        Variant("full", "full", all_blocks, order, side=side_all, keep_other=True),
    ]
    do_drop = sweep in ("drop + keep groups", "drop-only groups")
    do_keep = sweep in ("drop + keep groups", "keep-only groups")
    g = max(1, int(group_size))
    for start in range(0, nb, g):
        grp = list(range(start, min(start + g, nb)))
        if do_drop:
            out.append(Variant(f"drop_{group_tag(grp)}", "drop", [b for b in all_blocks if b not in grp], order,
                               side=side_all, group=grp, keep_other=True))
        if do_keep:
            out.append(Variant(f"keep_{group_tag(grp)}", "keep", grp, order, group=grp))
    if module_variants:
        for label, _ in profile.named_sets():
            out.append(Variant(f"keep_{label}", "modules", all_blocks, list(profile.sets[label]), side=side_all))
        if profile.side is not None and has_side:
            key = profile.side.key
            out.append(Variant(f"keep_{key}", "side", [], order, side=True))
            out.append(Variant(f"drop_{key}", "side", all_blocks, order, side=False, keep_other=True))
    for line in (custom_text or "").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        name, sep, rest = line.partition("=")
        name = name.strip()
        if not sep or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError(f"custom variant must look like NAME=BLOCKS[:LAYERS], got {line!r}")
        out.append(custom_variant(profile, name, rest, nb))
    names = [v.name for v in out]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise ValueError(f"duplicate variant names: {sorted(dupes)}")
    return out


def custom_variant(profile, name, spec, nb, kind="custom") -> Variant:
    """A variant from a selection written as text (see parse_variant_selection)."""
    include, exclude = parse_variant_selection(profile, spec, nb)
    if include is not None and len(include) == 1 and not exclude:  # one rectangle: the simple form
        p = include[0]
        return Variant(name, kind, p.blocks, p.modules, side=p.side, spec=str(spec).strip())
    parts = include if include is not None else [Selection(list(range(nb)), list(profile.order), profile.side is not None)]
    blocks = sorted({b for p in parts for b in p.blocks})
    mods = [m for m in profile.order if any(m in p.modules for p in parts)]
    return Variant(name, kind, blocks, mods, side=any(p.side for p in parts), include=include, exclude=exclude,
                   keep_other=include is None, spec=str(spec).strip())


def _in_parts(parts, k, all_mods) -> bool:
    for p in parts:
        if (k.block not in p.blocks) if k.group == "main" else not p.side:
            continue
        if (set(p.modules) == all_mods) if k.short is None else (k.short in p.modules):
            return True
    return False


def select_keys(profile, sd_keys, variant: Variant) -> list:
    """Tensor names a variant keeps (whole adapters per layer: A, B, alpha together)."""
    if variant.kind == "base":
        return []
    if variant.is_parts:
        all_mods = set(profile.order)
        out = []
        for name in sd_keys:
            k = parse_key(profile, name)
            if k.group is None:
                keep = variant.keep_other
            else:
                keep = (variant.include is None or _in_parts(variant.include, k, all_mods)) and \
                    not _in_parts(variant.exclude, k, all_mods)
            if keep:
                out.append(name)
        return out
    blocks, mods = set(variant.blocks), set(variant.modules)
    all_mods = mods == set(profile.order)
    out = []
    for name in sd_keys:
        k = parse_key(profile, name)
        if k.group is None:
            keep = variant.keep_other
        elif (k.block not in blocks) if k.group == "main" else not variant.side:
            keep = False
        elif k.short is None:
            keep = all_mods
        else:
            keep = k.short in mods
        if keep:
            out.append(name)
    return out


def numel(t) -> int:
    return int(t.numel()) if hasattr(t, "numel") else 0


def assign_keys(profile, sd: dict, variants: list) -> None:
    total = sum(numel(t) for t in sd.values()) or 1
    keys = list(sd.keys())
    for v in variants:
        v.keys = select_keys(profile, keys, v)
        v.param_share = sum(numel(sd[k]) for k in v.keys) / total if v.kind != "base" else 0.0


# --------------------------------------------------------------------- metrics


def _mean_se(xs):
    xs = [x for x in xs if finite(x)]
    if not xs:
        return NAN, NAN, 0
    n = len(xs)
    m = sum(xs) / n
    se = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n) if n > 1 else NAN
    return m, se, n


def summarize(records, variant_names, base="base", full="full") -> dict:
    """records: dicts with variant, kind ('id'|'ctrl'), sim, drift."""
    out = {}
    for v in variant_names:
        ids = [r for r in records if r["variant"] == v and r["kind"] == "id"]
        ctrls = [r for r in records if r["variant"] == v and r["kind"] == "ctrl"]
        m, se, n = _mean_se([r["sim"] for r in ids])
        lm, lse, ln = _mean_se([r["sim"] for r in ctrls])
        dm, _, _ = _mean_se([r["drift"] for r in ctrls])
        out[v] = dict(id_sim=m, id_se=se, id_faces=n, id_total=len(ids),
                      leak=lm, leak_se=lse, leak_faces=ln, ctrl_total=len(ctrls), drift=dm)
    b, f = out.get(base), out.get(full)
    gain = f["id_sim"] - b["id_sim"] if b and f and finite(f["id_sim"]) and finite(b["id_sim"]) else NAN
    lgain = f["leak"] - b["leak"] if b and f and finite(f["leak"]) and finite(b["leak"]) else NAN
    for s in out.values():
        s["id_kept"] = (s["id_sim"] - b["id_sim"]) / gain if finite(gain) and abs(gain) > 1e-6 and finite(s["id_sim"]) else NAN
        s["id_kept_se"] = s["id_se"] / abs(gain) if finite(gain) and abs(gain) > 1e-6 and finite(s["id_se"]) else NAN
        s["leak_kept"] = (s["leak"] - b["leak"]) / lgain if finite(lgain) and abs(lgain) > 1e-3 and finite(s["leak"]) else NAN
        fd = f["drift"] if f else NAN
        s["drift_rel"] = s["drift"] / fd if finite(fd) and fd > 1e-6 and finite(s["drift"]) else NAN
    return out


def variety(items) -> dict:
    """Prompt variety per variant: mean pairwise difference between the identity renders of different
    prompts (same seed), on small colour thumbnails. items: dicts with variant, kind, prompt, seed, lay
    (uint8 [h, w, 3] or None). Returns {variant: (mean, se, pairs)}."""
    by = defaultdict(lambda: defaultdict(dict))
    for it in items:
        lay = it.get("lay")
        if it.get("kind") == "id" and lay is not None:
            by[it["variant"]][it["seed"]][it["prompt"]] = np.asarray(lay, dtype=np.float32) / 255.0
    out = {}
    for v, seeds in by.items():
        ds = []
        for prompts in seeds.values():
            keys = sorted(prompts)
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    ds.append(float(np.abs(prompts[keys[i]] - prompts[keys[j]]).mean()))
        out[v] = _mean_se(ds)
    return out


def add_variety(summary: dict, items, base="base") -> dict:
    """Adds variety, variety_se and variety_rel (relative to the base model, 1.0 = as varied as base)."""
    vs = variety(items)
    b = vs.get(base, (NAN, NAN, 0))[0]
    ok = finite(b) and b > 1e-6
    for name, s in summary.items():
        m, se, n = vs.get(name, (NAN, NAN, 0))
        s["variety"], s["variety_se"], s["variety_pairs"] = m, se, n
        s["variety_rel"] = m / b if ok and finite(m) else NAN
        s["variety_rel_se"] = se / b if ok and finite(se) else NAN
    return summary


def paired_se(records, a: str, b: str, kind: str = "id") -> float:
    """Standard error of the mean per-render difference between two variants (same prompts and seeds)."""
    xa = {(r["prompt"], r["seed"]): r["sim"] for r in records if r["variant"] == a and r["kind"] == kind}
    xb = {(r["prompt"], r["seed"]): r["sim"] for r in records if r["variant"] == b and r["kind"] == kind}
    d = [xa[k] - xb[k] for k in xa if k in xb and finite(xa[k]) and finite(xb[k])]
    if len(d) < 2:
        return NAN
    m = sum(d) / len(d)
    return math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1) / len(d))


def _gain(summary):
    b, f = summary.get("base"), summary.get("full")
    gain = f["id_sim"] - b["id_sim"] if b and f and finite(f["id_sim"]) and finite(b["id_sim"]) else NAN
    return abs(gain) if finite(gain) and abs(gain) > 1e-6 else NAN


def _pair_row(summary, records, drop, keep, norm):
    s_drop = summary.get(drop.name) if drop else None
    sk = summary.get(keep.name) if keep else None
    nec = 1.0 - s_drop["id_kept"] if s_drop and finite(s_drop["id_kept"]) else NAN
    suf = sk["id_kept"] if sk else NAN
    leak = sk["leak_kept"] if sk else NAN
    drift = sk["drift_rel"] if sk else NAN
    side = max([x for x in (leak, drift) if finite(x)], default=NAN)
    nec_se = paired_se(records, drop.name, "full") / norm if records and drop and finite(norm) else NAN
    suf_se = paired_se(records, keep.name, "base") / norm if records and keep and finite(norm) else NAN
    return dict(necessity=nec, sufficiency=suf, leak=leak, drift=drift, side=side,
                params=keep.param_share if keep else NAN, nec_se=nec_se, suf_se=suf_se)


def group_rows(summary: dict, variants: list, records=None) -> list:
    norm = _gain(summary)
    groups = {}
    for v in variants:
        if v.kind in ("drop", "keep") and v.group:
            g = groups.setdefault(group_tag(v.group), {"group": v.group})
            g[v.kind] = v
    rows = []
    for tag in sorted(groups):
        g = groups[tag]
        row = _pair_row(summary, records, g.get("drop"), g.get("keep"), norm)
        row.update(tag=tag, group=g["group"])
        rows.append(row)
    return rows


def side_row(profile, summary, variants, records=None):
    """Necessity / sufficiency of the side group (Krea 2 text fusion), or None."""
    if profile.side is None:
        return None
    key = profile.side.key
    drop = next((v for v in variants if v.name == f"drop_{key}"), None)
    keep = next((v for v in variants if v.name == f"keep_{key}"), None)
    if drop is None and keep is None:
        return None
    row = _pair_row(summary, records, drop, keep, _gain(summary))
    row.update(tag=profile.side.label)
    return row


# ------------------------------------------------------------------ suggestion


def _score_group(r):
    nec, suf = clamp(r["necessity"]), clamp(r["sufficiency"])
    if not finite(r["sufficiency"]):
        return nec
    if not finite(r["necessity"]):
        nec = suf
    penalty = 0.25 * max(0.0, (r["side"] if finite(r["side"]) else 0.0) - suf)
    return 0.5 * (nec + suf) - penalty


def rank_groups(rows):
    for r in rows:
        r["score"] = _score_group(r)
    return sorted(rows, key=lambda r: -r["score"])


def group_effect(r) -> float:
    """Largest of necessity / sufficiency that is clear of twice its paired standard error."""
    n, s = clamp(r["necessity"]), clamp(r["sufficiency"])
    if finite(r.get("nec_se")) and n < 2 * r["nec_se"]:
        n = 0.0
    if finite(r.get("suf_se")) and s < 2 * r["suf_se"]:
        s = 0.0
    return max(n, s)


def predicted_retention(groups) -> float:
    """Noisy-OR of keep-alone sufficiency: rough guess of identity kept by a union of groups."""
    miss = 1.0
    for r in groups:
        miss *= 1.0 - clamp(r["sufficiency"])
    return 1.0 - miss


@dataclass
class Suggestion:
    blocks: list
    modules: list
    side: bool
    reasons: list
    remaining: list  # ranked groups not yet included
    module_kept: float = NAN
    predicted: float = NAN
    side_effect: float = NAN  # identity effect of the side group, if measured


def suggest(profile, summary: dict, rows: list, nb: int, target: float, side_info=None, has_side=False,
            min_effect: float = 0.05) -> Suggestion:
    """min_effect: a group counts only if removing it, or keeping it alone, moves identity by at least this much."""
    reasons = []
    order = list(profile.order)
    side, side_effect = bool(has_side), NAN

    # the side group (Krea 2 text fusion): alone enough, needed, or droppable?
    if has_side and side_info is not None:
        label = profile.side.label
        side_effect = group_effect(side_info)
        if finite(side_info["sufficiency"]) and side_info["sufficiency"] >= target:
            reasons.append(f"the {label} layers alone keep {100 * side_info['sufficiency']:.0f}% of the identity")
            return Suggestion([], order, True, reasons, [], NAN, side_info["sufficiency"], side_effect)
        if finite(side_info["necessity"]):
            if side_effect < min_effect:
                side = False
                reasons.append(f"removing the {label} layers costs less than {100 * min_effect:.0f}% "
                               "of the identity, so they are left out")
            else:
                reasons.append(f"removing the {label} layers loses {100 * clamp(side_info['necessity']):.0f}% "
                               "of the identity, so they stay")

    # layer types: smallest set that keeps the target on its own
    mods, mod_kept = order, NAN
    labels = [k for k in ("kv", "attn", "mlp") if k in profile.sets]
    for label in labels:
        s = summary.get(f"keep_{label}")
        if s and finite(s["id_kept"]) and s["id_kept"] >= target:
            mods, mod_kept = list(profile.sets[label]), s["id_kept"]
            reasons.append(f"{describe_modules(profile, mods)} alone keeps {100 * s['id_kept']:.0f}% "
                           "of the identity gain")
            break
    else:
        if any(f"keep_{label}" in summary for label in labels):
            reasons.append("no single layer type keeps the target on its own, so all layer types stay")

    # blocks: only groups with a measurable effect, greedily until the predicted retention reaches the target
    def done(blocks, remaining, pred=NAN):
        return Suggestion(blocks, mods, side, reasons, remaining, mod_kept, pred, side_effect)

    if not rows:
        reasons.append("no block-group sweep, so all blocks stay")
        return done(list(range(nb)), [])
    ranked = rank_groups(rows)
    have_suff = any(finite(r["sufficiency"]) for r in ranked)
    relevant = [r for r in ranked if group_effect(r) >= min_effect]
    if not relevant:
        reasons.append("no block group has a clear effect on its own; identity is redundant across blocks")
        return done(list(range(nb)), [])
    chosen = []
    if have_suff:
        for r in relevant:
            chosen.append(r)
            if predicted_retention(chosen) >= target:
                break
    else:  # drop-only sweep: cover most of the necessity mass
        total = sum(clamp(r["necessity"]) for r in relevant) or 1.0
        acc = 0.0
        for r in relevant:
            chosen.append(r)
            acc += clamp(r["necessity"])
            if acc >= 0.8 * total:
                break
    blocks = sorted(b for r in chosen for b in r["group"])
    remaining = [r for r in ranked if r not in chosen]
    pred = predicted_retention(chosen) if have_suff else NAN
    if len(blocks) > 0.75 * nb:
        reasons.append("identity is spread over most blocks; restricting blocks is unlikely to help")
        return done(list(range(nb)), [], pred)
    tags = ", ".join(r["tag"] for r in chosen)
    one = len(chosen) == 1
    if have_suff and pred >= target:
        reasons.append(f"{tags} {'is' if one else 'are'} estimated to keep {100 * pred:.0f}% "
                       f"of the identity on {'its' if one else 'their'} own")
    elif have_suff and pred < 0.5:
        reasons.append(f"removing {tags} costs identity, but {'it does' if one else 'they do'} little on "
                       f"{'its' if one else 'their'} own (blocks work together); the test render checks the subset")
    elif have_suff:
        reasons.append(f"{tags} {'is the group' if one else 'are the groups'} with a clear effect (estimated "
                       f"{100 * pred:.0f}% kept from solo results; the test render checks the actual subset)")
    else:
        reasons.append(f"{tags} {'is' if one else 'are'} where removing blocks hurts identity most")
    spare = [r["tag"] for r in relevant if r not in chosen]
    if spare:
        verb, aux = ("carries", "was") if len(spare) == 1 else ("carry", "were")
        reasons.append(f"{', '.join(spare)} also {verb} some of it but {aux} not needed for the target")
    if len(relevant) < len(ranked):
        reasons.append(f"the other groups change identity by less than {100 * min_effect:.0f}%")
    return done(blocks, remaining, pred)


def widen(profile, sel: Selection, remaining, nb, module_kept, predicted, has_side, side_effect=NAN):
    """Next candidate after a failed test. Returns (Selection, remaining, why) or None."""
    if has_side and not sel.side:
        return Selection(sel.blocks, sel.modules, True), remaining, f"added the {profile.side.label} layers"
    order = list(profile.order)
    est_block_loss = 1.0 - predicted if finite(predicted) else 0.5
    est_mod_loss = 1.0 - module_kept if finite(module_kept) else 0.0
    can_widen_mods = set(sel.modules) != set(order)
    if can_widen_mods and (est_mod_loss >= est_block_loss or not remaining):
        steps = [list(profile.sets[k]) for k in ("kv", "attn") if k in profile.sets] + [order]
        step = next(s for s in steps if set(sel.modules) < set(s))  # kv -> attention -> all; MLP -> all
        return Selection(sel.blocks, step, sel.side), remaining, f"widened layers to {describe_modules(profile, step)}"
    if remaining:
        nxt = remaining[0]
        return (Selection(sorted(set(sel.blocks) | set(nxt["group"])), sel.modules, sel.side), remaining[1:],
                f"added group {nxt['tag']}")
    if len(sel.blocks) < nb:
        return Selection(list(range(nb)), sel.modules, sel.side), [], "widened to all blocks"
    return None


def is_everything(profile, sel: Selection, nb, has_side) -> bool:
    return (len(sel.blocks) == nb and set(sel.modules) == set(profile.order)
            and (sel.side or not has_side))


# --------------------------------------------------------------- ai-toolkit config


def side_block_paths(profile):
    return list(profile.side.blocks) if profile.side is not None else []


def main_layers(profile, present=None):
    """Layers of a main block; present: layer shorts the loaded model has (None: all but the optional ones)."""
    return [l for l in profile.layers if (l.short in present if present is not None else not l.optional)]


def side_layers(profile):
    return [l for l in profile.layers if profile.side is not None and l.short in profile.side.shapes]


def canonical_linears(profile, nb: int, present=None):
    for b in range(nb):
        for l in main_layers(profile, present):
            yield f"{profile.canonical_block.format(b=b)}.{l.path}"
    for block in side_block_paths(profile):
        for l in side_layers(profile):
            yield f"{block}.{l.path}"
    yield from profile.non_block_linears


def expected_paths(profile, blocks, modules, side, present=None):
    mods = [m for m in modules if profile.layer(m).default]
    main = {l.short for l in main_layers(profile, present)}
    paths = {f"{profile.canonical_block.format(b=b)}.{profile.layer(m).path}" for b in blocks for m in mods
             if m in main}
    if side:
        sl = {l.short for l in side_layers(profile)}
        paths |= {f"{block}.{profile.layer(m).path}" for block in side_block_paths(profile) for m in mods if m in sl}
    return paths


def simulate_ai_toolkit(profile, only, ignore, transformer_only=True, nb=None, present=None):
    """Mirror toolkit/lora_special.py create_modules() filtering: substring tests on the dotted
    module name ("transformer." + path); only_if_contains also tests the "$$" form."""
    nb = profile.default_blocks if nb is None else nb
    chosen = []
    for path in canonical_linears(profile, nb, present):
        clean = "transformer." + path
        lora_name = clean.replace(".", "$$")
        if any(w in clean for w in ignore):
            continue
        if transformer_only and not any(name in clean for name in profile.transformer_block_names):
            continue
        if only is not None and not (any(w in clean for w in only) or any(w in lora_name for w in only)):
            continue
        chosen.append(path)
    return chosen


def ai_toolkit_patterns(profile, blocks, modules, side, nb: int):
    all_main = len(blocks) == nb
    side_all = profile.side is None or side
    if all_main and side_all:
        only = None
    else:
        only = []
        if all_main and profile.all_blocks_pattern:
            only.append(profile.all_blocks_pattern)
        else:
            only += [profile.block_pattern.format(b=b) for b in blocks]
        if profile.side is not None and side:
            only.append(profile.side.pattern)
        if not only:
            raise ValueError("the selection is empty")
    # layers the model's default config leaves out (MiniMax H3's adaln) are never suggested
    excluded = [m for m in profile.default_order if m not in modules]
    if not excluded:
        ignore = []
    elif set(excluded) == set(profile.sets["mlp"]):
        ignore = [profile.group_ignores["mlp"]]
    elif set(excluded) == set(profile.sets["attn"]):
        ignore = [profile.group_ignores["attn"]]
    else:
        ignore = [profile.layer(m).ignore for m in excluded]
    ignore += [p for p in profile.always_ignore if p not in ignore]
    return only, ignore


def _yaml_list(key: str, items, per_line: int = 2) -> list:
    """A YAML list: block style when short, a wrapped flow list ([a, b, ...]) when long."""
    quoted = [f'"{p}"' for p in items]
    if len(quoted) <= 4:
        return [f"{key}:"] + [f"{' ' * (len(key) - len(key.lstrip()) + 2)}- {q}" for q in quoted]
    head = f"{key}: ["
    pad = " " * len(head)
    rows = [", ".join(quoted[i : i + per_line]) for i in range(0, len(quoted), per_line)]
    out = [head + rows[0] + ("," if len(rows) > 1 else "]")]
    for j, r in enumerate(rows[1:], start=1):
        out.append(pad + r + ("," if j < len(rows) - 1 else "]"))
    return out


def lora_param_fraction(profile, blocks, modules, side, nb: int, present=None) -> float:
    """Size of a LoRA on these layers relative to ai-toolkit's default LoRA for the model."""
    main = {l.short: sum(l.shape) for l in main_layers(profile, present) if l.default}
    n_side = len(side_block_paths(profile))
    side_sz = {l.short: sum(profile.side.shapes[l.short]) for l in side_layers(profile) if l.default}
    full = nb * sum(main.values()) + n_side * sum(side_sz.values())
    part = len(blocks) * sum(main.get(m, 0) for m in modules)
    if side and n_side:
        part += n_side * sum(side_sz.get(m, 0) for m in modules)
    return part / full if full else NAN


ALWAYS_IGNORE_NOTE = {
    "adaln_proj": "adaln_proj stays out as in ai-toolkit's preset: an adaln LoRA only fits the checkpoint it was "
                  "trained on (the pruned files store adaln differently)",
}


def ai_toolkit_yaml(profile, blocks, modules, side, nb: int, rank: int = None, show_side=True, present=None) -> str:
    rank = profile.rank if rank is None else rank
    only, ignore = ai_toolkit_patterns(profile, blocks, modules, side, nb)
    chosen = simulate_ai_toolkit(profile, only, ignore, True, nb, present)
    if set(chosen) != expected_paths(profile, blocks, modules, side, present):
        raise RuntimeError("generated ai-toolkit patterns do not select exactly the requested layers")
    lines = [
        f"# ai-toolkit, model arch: {profile.arch}",
        "network:",
        '  type: "lora"',
        f"  linear: {rank}",
        f"  linear_alpha: {rank}" + (f"   # {profile.alpha_note}" if profile.alpha_note else ""),
        "  transformer_only: true",
    ]
    if only or ignore:
        lines.append("  network_kwargs:")
        if only:
            lines += _yaml_list("    only_if_contains", only)
        if ignore:
            lines += _yaml_list("    ignore_if_contains", ignore)
    lines.append(f"# ai-toolkit should log: create LoRA for U-Net: {len(chosen)} modules.")
    mods = [m for m in modules if profile.layer(m).default] or list(modules)
    lines.append(f"# {describe_selection(profile, nb, blocks, side, mods, show_side)}; "
                 f"{100 * lora_param_fraction(profile, blocks, modules, side, nb, present):.0f}% of the default "
                 "LoRA size")
    for p in profile.always_ignore:
        if p in ALWAYS_IGNORE_NOTE:
            lines.append(f"# {ALWAYS_IGNORE_NOTE[p]}")
    return "\n".join(lines)


# ----------------------------------------------------------------- checkpoints

_STEP_RE = re.compile(r"^(.*?)[_-](\d+)$")


def _stem(name: str) -> str:
    return os.path.splitext(os.path.basename(str(name).replace("\\", "/")))[0]


def checkpoint_step(name, metadata=None):
    """Training step of a saved LoRA: ai-toolkit metadata (training_info.step), kohya (ss_steps), else the
    number at the end of the file name (ai-toolkit saves name_000001500.safetensors). None if unknown."""
    md = metadata or {}
    ti = md.get("training_info")
    if ti:
        try:
            ti = json.loads(ti) if isinstance(ti, str) else ti
            if isinstance(ti, dict) and isinstance(ti.get("step"), (int, float)):
                return int(ti["step"])
        except (ValueError, TypeError):
            pass
    ss = md.get("ss_steps")
    if ss is not None and str(ss).strip().isdigit():
        return int(str(ss).strip())
    m = _STEP_RE.match(_stem(name))
    return int(m.group(2)) if m else None


def checkpoint_run(name, metadata=None) -> str:
    """Which training run a save belongs to: the job name in the metadata, else the file name without its step."""
    md = metadata or {}
    if md.get("ss_output_name"):
        return str(md["ss_output_name"])
    m = _STEP_RE.match(_stem(name))
    return m.group(1) if m else _stem(name)


def order_checkpoints(items) -> list:
    """items: dicts with name, step, run. Saves from one training run are put in step order (no step last);
    LoRAs from different runs keep the order they were given in."""
    items = list(items)
    if len(items) > 1 and len({it["run"] for it in items}) == 1:
        return sorted(items, key=lambda it: (it["step"] is None, it["step"] or 0))
    return items


def _abbrev(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def label_checkpoints(items) -> None:
    """Adds 'label' (e.g. 'step 1,500') and 'short' (x-axis label) to each item, unique within the list.
    Saves of one training run are labelled by step ('final' for the one without a step); LoRAs from
    different runs by run name and step, or by file name."""
    one_run = len({it["run"] for it in items}) == 1
    for it in items:
        stem, step = _stem(it["name"]), it["step"]
        if step is not None and one_run:
            it["label"], it["short"] = f"step {step:,}", f"{step:,}"
        elif step is not None:
            run = str(it["run"])
            it["label"], it["short"] = f"{run} · step {step:,}", f"{_abbrev(run, 7)} {step:,}"
        elif one_run and len(items) > 1:
            it["label"], it["short"] = "final", "final"
        else:
            it["label"], it["short"] = stem, _abbrev(stem, 12)
    for key in ("label", "short"):
        counts, seen = defaultdict(int), defaultdict(int)
        for it in items:
            counts[it[key]] += 1
        for it in items:
            base = it[key]
            if counts[base] > 1:
                seen[base] += 1
                it[key] = f"{base} #{seen[base]}"


@dataclass
class CheckpointProbe:
    """What the comparison needs from one checkpoint's probe."""

    name: str
    label: str
    short: str
    step: object
    info: LoraInfo
    nb: int
    variants: list
    summary: dict
    rows: list
    side_info: object
    what: str  # suggested subset in words
    tested_kept: float  # identity kept by the suggested subset in its test render (nan if untested)
    tested_pass: object  # True / False / None
    size_frac: float
    restricted: bool
    reliable: bool


def _full(c):
    return c.summary.get("full") or {}


def _get_num(d, key):
    v = (d or {}).get(key, NAN)
    return v if finite(v) else NAN


MAX_COMPARED_CUSTOM = 12  # custom variants shown in the comparison grid; longer lists stay per checkpoint


def compare_checkpoints(profile, cks, target, group_size, scorer_name, min_effect=0.05, sequence=True) -> dict:
    """Everything the checkpoint comparison shows: trend charts, reliance grids, summary table, verdict.
    sequence: the files are saves of one training run in step order (trends and overfitting signs make
    sense); otherwise they are compared side by side."""
    base = cks[0].summary.get("base") or {}
    b_id, b_leak = _get_num(base, "id_sim"), _get_num(base, "leak")
    labels, shorts = [c.label for c in cks], [c.short for c in cks]

    def series(key, se_key=None):
        return [(_get_num(_full(c), key), _get_num(_full(c), se_key) if se_key else NAN) for c in cks]

    charts = [
        dict(title="Likeness", sub=f"full LoRA, {scorer_name} similarity to the references",
             values=series("id_sim", "id_se"), ref=(b_id, "base"), fmt="num"),
        dict(title="Leak", sub="the same similarity on control prompts",
             values=series("leak", "leak_se"), ref=(b_leak, "base"), fmt="num"),
        dict(title="Drift", sub="how much control images change vs no LoRA",
             values=series("drift"), ref=(0.0, None), fmt="num", ymin=0.0),
        dict(title="Prompt variety", sub="identity renders vs each other; base = 100%",
             values=series("variety_rel", "variety_rel_se"), ref=(1.0, "base"), fmt="pct", ymin=0.0),
    ]
    for ch in charts:
        ch["connect"] = sequence

    # necessity / sufficiency per block group (+ the side group)
    groups = {}
    for c in cks:
        for r in c.rows:
            groups.setdefault(r["tag"], r["group"])
    tags = sorted(groups, key=lambda t: groups[t][0])
    has_side_rows = profile.side is not None and any(c.side_info for c in cks)

    def row_val(c, tag, key):
        r = next((r for r in c.rows if r["tag"] == tag), None)
        return r[key] if r and finite(r[key]) else NAN

    def side_val(c, key):
        return c.side_info[key] if c.side_info and finite(c.side_info[key]) else NAN

    rel_cols = tags + ([profile.side.label] if has_side_rows else [])
    nec = [[row_val(c, t, "necessity") for t in tags] + ([side_val(c, "necessity")] if has_side_rows else [])
           for c in cks]
    suf = [[row_val(c, t, "sufficiency") for t in tags] + ([side_val(c, "sufficiency")] if has_side_rows else [])
           for c in cks]

    # weight-change share per block group, from the LoRA files
    nb = cks[0].nb
    g = max(1, int(group_size))
    wgroups = [list(range(s, min(s + g, nb))) for s in range(0, nb, g)]
    wside = profile.side is not None and any(c.info.has_side for c in cks)
    wcols = [group_tag(gr) for gr in wgroups] + ([profile.side.label] if wside else [])
    wvals, norms = [], []
    for c in cks:
        tot = c.info.total_energy()
        be = c.info.block_energy(nb)
        row = [sum(be[b] for b in gr) / tot for gr in wgroups]
        if wside:
            row.append(sum(c.info.side_energy()) / tot)
        wvals.append(row)
        norms.append(math.sqrt(tot))
    wmax = max([v for row in wvals for v in row if finite(v)] or [1.0]) or 1.0

    # identity kept by layer types, the side group and custom subsets
    sub_cols = [(k, lab) for k, lab in (("keep_attn", "attention alone"), ("keep_mlp", "MLP alone"),
                                         ("keep_kv", "k,v alone")) if any(k in c.summary for c in cks)]
    if profile.side is not None:
        key = profile.side.key
        for k, lab in ((f"keep_{key}", f"{profile.side.label} alone"), (f"drop_{key}", f"without {profile.side.label}")):
            if any(k in c.summary for c in cks):
                sub_cols.append((k, lab))
    customs = []
    for c in cks:
        for v in c.variants:
            if v.kind == "custom" and v.name not in customs:
                customs.append(v.name)
    omitted = 0
    if len(customs) <= MAX_COMPARED_CUSTOM:
        sub_cols += [(n, n) for n in customs]
    else:  # a long custom sweep stays in each checkpoint's own diagnostic
        omitted = len(customs)
    sub_vals = [[_get_num(c.summary.get(k), "id_kept") for k, _ in sub_cols] for c in cks]
    if any(finite(c.tested_kept) for c in cks):
        sub_cols.append(("suggest", "suggested subset"))
        for row, c in zip(sub_vals, cks):
            row.append(c.tested_kept)

    nmax = max(norms) if norms else 1.0
    table = []
    for c, n in zip(cks, norms):
        f = _full(c)
        table.append(dict(name=os.path.basename(c.name), label=c.label, step=c.step,
                          change=n / nmax if nmax > 0 else NAN, id_sim=_get_num(f, "id_sim"), id_se=_get_num(f, "id_se"),
                          leak=_get_num(f, "leak"), drift=_get_num(f, "drift"), variety=_get_num(f, "variety_rel"),
                          what=c.what, kept=c.tested_kept, passed=c.tested_pass, size=c.size_frac,
                          restricted=c.restricted, reliable=c.reliable))

    verdict, pick = _comparison_verdict(profile, cks, base, labels, shorts, target, scorer_name, norms, wvals,
                                        len(wgroups), min_effect, sequence)
    return dict(sequence=sequence, labels=labels, shorts=shorts, names=[os.path.basename(c.name) for c in cks],
                weak=[not c.reliable for c in cks], charts=charts,
                grids=dict(necessity=dict(cols=rel_cols, values=nec), sufficiency=dict(cols=rel_cols, values=suf),
                           weights=dict(cols=wcols, values=wvals, vmax=wmax),
                           subsets=dict(cols=[lab for _, lab in sub_cols], keys=[k for k, _ in sub_cols], values=sub_vals,
                                        omitted=omitted)),
                table=table, verdict=verdict, pick=pick)


def _comparison_verdict(profile, cks, base, labels, shorts, target, scorer_name, norms, wvals, n_wgroups, min_effect,
                        sequence=True):
    out = []
    K = len(cks)
    b_id, b_leak = _get_num(base, "id_sim"), _get_num(base, "leak")
    full = [_full(c) for c in cks]
    ids = [_get_num(f, "id_sim") for f in full]
    gains = [x - b_id if finite(x) and finite(b_id) else NAN for x in ids]
    pick = None

    if any(finite(x) for x in ids):
        fin = [x for x in ids if finite(x)] + ([b_id] if finite(b_id) else [])
        spec = ".3f" if max(fin) - min(fin) < 0.03 else ".2f"  # enough decimals to tell them apart
        trail = (" → " if sequence else ", ").join(f"{s} {format(x, spec)}" if finite(x) else f"{s} –"
                                                   for s, x in zip(shorts, ids))
        out.append(("info", f"Likeness ({scorer_name}, full LoRA): {trail}; without the LoRA {fmt(b_id, spec)}."))

    valid = [i for i in range(K) if finite(gains[i]) and cks[i].reliable]
    best_i = max(valid, key=lambda i: gains[i]) if valid else None
    i0 = None
    if best_i is not None and gains[best_i] > 1e-6 and not sequence:
        pick = best_i
        if K > 1:
            out.append(("good", f"Highest likeness: {labels[best_i]}."))
    elif best_i is not None and gains[best_i] > 1e-6:
        i0 = min(i for i in valid if gains[i] >= 0.95 * gains[best_i])
        pick = i0
        if K == 1:
            pass
        elif i0 == K - 1:
            out.append(("info", f"Likeness is still rising at the last checkpoint ({labels[-1]}), so more training "
                                "may still help."))
        else:
            extra = f"; {labels[best_i]} scores highest" if best_i != i0 else ""
            out.append(("good", f"Earliest checkpoint within 5% of the best likeness: {labels[i0]}{extra}."))

    # overfitting signs after likeness levels off (one training run only)
    if sequence and i0 is not None and i0 < K - 1:
        L = K - 1
        f0, fL = full[i0], full[L]
        signs = []
        l0, lL = _get_num(f0, "leak"), _get_num(fL, "leak")
        if finite(l0) and finite(lL) and finite(b_leak):
            g0, gL = l0 - b_leak, lL - b_leak
            se = math.sqrt(_get_num(f0, "leak_se") ** 2 + _get_num(fL, "leak_se") ** 2) \
                if finite(_get_num(f0, "leak_se")) and finite(_get_num(fL, "leak_se")) else 0.0
            if gL - g0 > max(0.02, 2 * se, 0.15 * abs(g0)):
                signs.append(f"leak rises ({g0:+.2f} → {gL:+.2f} over no LoRA)")
        d0, dL = _get_num(f0, "drift"), _get_num(fL, "drift")
        if finite(d0) and finite(dL) and dL - d0 > max(0.02, 0.15 * d0):
            signs.append(f"drift rises ({d0:.2f} → {dL:.2f})")
        v0, vL = _get_num(f0, "variety_rel"), _get_num(fL, "variety_rel")
        if finite(v0) and finite(vL) and v0 - vL > 0.08:
            signs.append(f"prompt variety falls ({100 * v0:.0f}% → {100 * vL:.0f}% of the base model's)")
        if signs:
            out.append(("warn", f"From {labels[i0]} to {labels[L]}, likeness hardly improves, but "
                                + ", ".join(signs) + ": typical signs of overfitting."))
        else:
            out.append(("good", f"No sign of overfitting from {labels[i0]} to {labels[L]}: leak, drift and prompt "
                                "variety stay about the same."))

    # where the identity lives, and how that moves
    with_rows = [c for c in cks if c.rows and c.reliable]
    if with_rows:
        parts, stats = [], []
        for c in with_rows:
            rel = [r for r in c.rows if group_effect(r) >= min_effect]
            tops = [r for r in c.rows if finite(r["sufficiency"])]
            top = max(tops, key=lambda r: r["sufficiency"]) if tops else None
            parts.append(f"{c.short}: {len(rel)} of {len(c.rows)} groups"
                         + (f", best alone {top['tag']} {100 * top['sufficiency']:.0f}%" if top else ""))
            stats.append((len(rel), top))
        out.append(("info", "Where the identity lives: " + "; ".join(parts) + "."))
        if len(with_rows) > 1:
            (n_a, top_a), (n_z, top_z) = stats[0], stats[-1]
            a, z = with_rows[0].short, with_rows[-1].short
            if not sequence:
                pass
            elif n_z < n_a and (top_z is None or top_a is None or top_z["sufficiency"] >= top_a["sufficiency"]):
                out.append(("info", f"The identity concentrates in fewer blocks as training goes on ({n_a} → {n_z} "
                                    f"groups from {a} to {z})."))
            elif n_z > n_a:
                out.append(("info", f"The identity spreads over more blocks as training goes on ({n_a} → {n_z} "
                                    f"groups from {a} to {z})."))
            if top_a and top_z and top_a["tag"] != top_z["tag"]:
                if sequence:
                    out.append(("info", f"The strongest group moves from {top_a['tag']} ({a}) to {top_z['tag']} ({z})."))
                else:
                    out.append(("info", f"The strongest group differs: {top_a['tag']} in {a}, {top_z['tag']} in {z}."))

    # layer types and the side group
    reliable = [c for c in cks if c.reliable]
    if len(reliable) > 1:
        a, z = reliable[0], reliable[-1]
        cols = [("keep_attn", "attention alone"), ("keep_mlp", "MLP alone"), ("keep_kv", "k,v alone")]
        if profile.side is not None:
            cols += [(f"keep_{profile.side.key}", f"{profile.side.label} alone")]
        changes = []
        for k, lab in cols:
            x, y = _get_num(a.summary.get(k), "id_kept"), _get_num(z.summary.get(k), "id_kept")
            if finite(x) and finite(y) and abs(y - x) >= 0.15:
                changes.append(f"{lab} {100 * x:.0f}% → {100 * y:.0f}%")
        if changes:
            sep = " → " if sequence else " vs "
            out.append(("info", f"Identity kept by layer type, {a.short}{sep}{z.short}: "
                                + ", ".join(c.replace(" → ", sep) for c in changes) + "."))

    # weight change
    if K > 1 and finite(norms[0]) and norms[0] > 0 and finite(norms[-1]):
        top2 = [sum(sorted(row[:n_wgroups], reverse=True)[:2]) for row in wvals]
        ratio = norms[-1] / norms[0]
        if sequence:
            text = (f"The LoRA's total weight change {'grows' if ratio >= 1 else 'shrinks'} {ratio:.1f}× from "
                    f"{shorts[0]} to {shorts[-1]}")
            if abs(top2[-1] - top2[0]) >= 0.08:
                text += f", and the two largest block groups go from {100 * top2[0]:.0f}% to {100 * top2[-1]:.0f}% of it"
        else:
            text = f"The total weight change of {shorts[-1]} is {ratio:.1f}× that of {shorts[0]}"
            if abs(top2[-1] - top2[0]) >= 0.08:
                text += (f"; its two largest block groups hold {100 * top2[-1]:.0f}% of it, against "
                         f"{100 * top2[0]:.0f}% in {shorts[0]}")
        out.append(("info", text + "."))

    # suggestions
    sugg = [c for c in cks if c.reliable and c.what]
    if len(sugg) > 1:
        if len({c.what for c in sugg}) == 1:
            out.append(("info", f"Every checkpoint suggests the same subset: {sugg[0].what}."))
        else:
            out.append(("info", "Suggested subsets: " + "; ".join(
                f"{c.short}: {c.what}" + (f" ({100 * c.tested_kept:.0f}% in its test)" if finite(c.tested_kept) else "")
                for c in sugg) + "."))

    for c in cks:
        if not c.reliable:
            out.append(("warn", f"{c.label}: the LoRA barely changes {scorer_name} similarity, so its layer numbers "
                                "are noise."))
    out.append(("info", f"Necessity, sufficiency and the subset numbers are relative to each "
                        f"{'checkpoint' if sequence else 'file'}'s own full LoRA; the charts show absolute values."))
    return out, pick


def _pct(v, na="-"):
    return f"{round(100 * v)}%" if finite(v) else na


def comparison_text(cmp, settings="") -> str:
    out = ["LoRA Prober: checkpoint comparison", settings, ""]
    for kind, text in cmp["verdict"]:
        out.append(f"{'!' if kind in ('warn', 'bad') else '-'} {text}")
    out.append("")
    head = f"{'checkpoint' if cmp.get('sequence', True) else 'LoRA file':<34} {'likeness':>9} {'leak':>6} {'drift':>6} {'variety':>8} {'change':>7}  suggested subset"
    out.append(head)
    for r in cmp["table"]:
        kept = f" ({100 * r['kept']:.0f}% in test)" if finite(r["kept"]) else ""
        out.append(f"{(r['label'] + ' · ' + r['name'])[:34]:<34} {fmt(r['id_sim'], '.3f'):>9} {fmt(r['leak'], '.3f'):>6} "
                   f"{fmt(r['drift'], '.3f'):>6} {_pct(r['variety']):>8} {_pct(r['change']):>7}  {r['what'] or '-'}{kept}")
    for key, title in (("necessity", "Necessity (identity lost when the group is removed)"),
                       ("sufficiency", "Sufficiency (identity kept with only the group)"),
                       ("weights", "Share of the weight change"),
                       ("subsets", "Identity kept by layer types and subsets")):
        grid = cmp["grids"][key]
        if not grid["cols"] or not any(finite(v) for row in grid["values"] for v in row):
            continue
        out.append("")
        out.append(title)
        out.append(f"{'':<16}" + "".join(f"{c[:10]:>11}" for c in grid["cols"]))
        for lab, row in zip(cmp["shorts"], grid["values"]):
            out.append(f"{lab[:16]:<16}" + "".join(f"{_pct(v):>11}" for v in row))
    return "\n".join(out)


# ------------------------------------------------------------------ LoRA Prober Compare node

MAX_COMPARE_FILES = 32


def match_loras(pattern: str, available) -> list:
    """LoRA files for one line of the compare list: an exact name (extension optional), a pattern with * ? [
    (tested on the path and on the file name), else every file whose path contains the text. Case-insensitive
    when nothing matches exactly."""
    import fnmatch

    avail = list(available)
    norm = {a.replace("\\", "/"): a for a in avail}
    p = pattern.strip().replace("\\", "/")
    if not p:
        return []
    for cand in (p, p + ".safetensors"):
        if cand in norm:
            return [norm[cand]]
    low = {k.lower(): v for k, v in norm.items()}
    for cand in (p.lower(), p.lower() + ".safetensors"):
        if cand in low:
            return [low[cand]]
    if any(ch in p for ch in "*?["):
        pl = p.lower()
        hits = [v for k, v in norm.items()
                if fnmatch.fnmatchcase(k.lower(), pl) or fnmatch.fnmatchcase(k.lower().rsplit("/", 1)[-1], pl)
                or fnmatch.fnmatchcase(k.lower(), pl + ".safetensors")]
        return sorted(hits, key=lambda s: s.lower())
    return sorted((v for k, v in norm.items() if p.lower() in k.lower()), key=lambda s: s.lower())


def parse_lora_list(text: str, available, max_files: int = MAX_COMPARE_FILES) -> list:
    """The compare node's list: one entry per line, NAME [| strength] [| selection]; # starts a comment.
    Returns dicts with name, strength (None = the node's), spec, order."""
    import difflib

    avail = list(available)
    items, seen = [], set()
    for raw in (text or "").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        fields = [f.strip() for f in line.split("|")]
        pat, strength, spec = fields[0], None, ""
        for f in fields[1:]:
            if not f:
                continue
            try:
                value = float(f)
            except ValueError:
                value = None
            if (value is not None and strength is not None) or (value is None and spec):
                raise ValueError(f"Too many fields in {line!r}: write NAME | strength | selection.")
            if value is not None:
                strength = value
            else:
                spec = f
        hits = match_loras(pat, avail)
        exact = len(hits) == 1 and hits[0].replace("\\", "/").lower() in (pat.replace("\\", "/").lower(),
                                                                           pat.replace("\\", "/").lower() + ".safetensors")
        if not hits:
            close = difflib.get_close_matches(pat, avail, n=3, cutoff=0.5)
            raise ValueError(f"No LoRA file matches {pat!r}." + (f" Did you mean {', '.join(close)}?" if close else
                                                                 " Write part of the name, a folder, or a pattern "
                                                                 "with *."))
        for h in hits:
            k = (h, strength, spec)
            if k in seen:
                continue
            seen.add(k)
            items.append(dict(name=h, strength=strength, spec=spec, order=len(items), exact=exact))
    if not items:
        raise ValueError("List at least one LoRA file in loras (one per line).")
    if len(items) > max_files:
        raise ValueError(f"The list matches {len(items)} LoRA files; the limit is {max_files}. Narrow the patterns.")
    return items


def order_runs(items) -> list:
    """Saves grouped by training run (runs in the order they first appear), each run in step order (files
    without a step last); the same file at other strengths or layers stays next to it."""
    runs = []
    for it in items:
        if it["run"] not in runs:
            runs.append(it["run"])
    return sorted(items, key=lambda it: (runs.index(it["run"]), it["step"] is None, it["step"] or 0, it["order"]))


def label_compare_items(items, default_strength) -> None:
    """label / short per item: run and step of the file (as for checkpoints), plus the strength or layers when
    a line sets them; still-equal labels get #2, #3."""
    files = {}
    for it in items:
        files.setdefault(it["name"], dict(name=it["name"], step=it["step"], run=it["run"]))
    label_checkpoints(list(files.values()))
    per_run = defaultdict(list)  # labels within a run ("step 500", "final"), for charts and rows that name the run
    for f in files.values():
        per_run[f["run"]].append(dict(f))
    run_labels = {}
    for fs in per_run.values():
        label_checkpoints(fs)
        for f in fs:
            run_labels[f["name"]] = (f["label"], f["short"])
    for it in items:
        f = files[it["name"]]
        extra = []
        s = it.get("strength")
        if s is not None and abs(s - default_strength) > 1e-9:
            extra.append(f"×{s:g}")
        if it.get("spec"):
            extra.append(f"[{it['spec']}]")
        tail = (" " + " ".join(extra)) if extra else ""
        it["label"], it["short"] = f["label"] + tail, f["short"] + tail
        it["run_label"], it["run_short"] = run_labels[it["name"]][0] + tail, run_labels[it["name"]][1] + tail
    def unique(group, key):
        counts, seen = defaultdict(int), defaultdict(int)
        for it in group:
            counts[it[key]] += 1
        for it in group:
            base = it[key]
            if counts[base] > 1:
                seen[base] += 1
                if seen[base] > 1:
                    it[key] = f"{base} #{seen[base]}"

    for key in ("label", "short"):
        unique(items, key)
    for run in {it["run"] for it in items}:  # run labels only need to differ within their run
        for key in ("run_label", "run_short"):
            unique([it for it in items if it["run"] == run], key)


def compare_layout(items, include_base):
    """Grid positions. Several runs with several saves each: a row per run (base alone at the top left);
    otherwise None (the caller wraps everything in reading order)."""
    runs = []
    for it in items:
        if it["run"] not in runs:
            runs.append(it["run"])
    per_run = [[it for it in items if it["run"] == r] for r in runs]
    if len(runs) < 2 or max(len(p) for p in per_run) < 2:
        return None
    col0 = 1 if include_base else 0
    pos = []
    if include_base:
        pos.append((None, 0, 0))
    for r, its in enumerate(per_run):
        for j, it in enumerate(its):
            pos.append((it, r, col0 + j))
    return pos, len(runs), col0 + max(len(p) for p in per_run)


def compare_verdict(items, summary, has_refs, has_ctrl, scorer_name, base_name="base"):
    """Verdict lines for the compare node. items: ordered dicts with name (summary key), label, run, step."""
    out = []
    if not has_refs:
        out.append(("info", "No reference images connected, so there are no likeness scores: compare the "
                            "renders and videos by eye."))
        return out
    b = summary.get(base_name) or {}
    b_id = b.get("id_sim", NAN)
    ids = {it["key"]: summary[it["key"]]["id_sim"] for it in items if finite(summary[it["key"]]["id_sim"])}
    if not ids:
        out.append(("bad", f"{scorer_name} found nothing to score in the renders (no faces?). Use clearer references "
                           "or CLIP Vision."))
        return out
    best = max(items, key=lambda it: ids.get(it["key"], -1e9))
    bv = ids[best["key"]]
    best_lab = "the final save" if best["label"] == "final" else best["label"]
    below = finite(b_id) and bv <= b_id  # not even the best file beats the base model
    if below:
        out.append(("warn", f"No LoRA is more like the references than the base model (best: {best_lab} at {bv:.3f}, "
                            f"base model {b_id:.3f}): the character has not been learned, or the references do not "
                            "show it."))
    else:
        gain = f" (base model {b_id:.3f})" if finite(b_id) else ""
        out.append(("good", f"Highest likeness: {best_lab} at {bv:.3f}{gain}."))
    runs = []
    for it in items:
        if it["run"] not in runs:
            runs.append(it["run"])
    multi = len(runs) > 1

    def tag(it):  # a save as named on its run's line ("step 1,000", "the final save")
        lab = it.get("run_label", it["label"]) if multi else it["label"]
        return "the final save" if lab == "final" else lab

    for run in runs:
        its = [it for it in items if it["run"] == run and it["key"] in ids]
        # a trend needs saves of one run in step order (a final save without a step comes last)
        if len(its) < 2 or sum(1 for it in its if it["step"] is None) > 1 or \
                len({(it["step"], it.get("strength"), it.get("spec")) for it in its}) < len(its) or \
                len({it["name"] for it in its}) < 2:
            continue
        name = f"{run}: " if multi else ""

        def say(kind, text, name=name):  # lines about one run start with its name when there are several
            out.append((kind, name + text if name else text[0].upper() + text[1:]))

        top = max(ids[it["key"]] for it in its)
        lo = b_id if finite(b_id) else min(ids[it["key"]] for it in its)
        span = top - lo
        learned = span > 1e-6
        if not learned and not below and finite(b_id):
            say("warn", f"no save is more like the references than the base model ({top:.3f} vs {b_id:.3f}).")
        early = next((it for it in its if learned and ids[it["key"]] >= top - 0.05 * span), None)
        if early is not None:
            v = ids[early["key"]]
            if v >= top and early is its[0]:
                say("info", f"{tag(early)}, the first save, already has the run's best likeness ({top:.3f}).")
            elif v >= top:
                say("info", f"{tag(early)} has the run's best likeness ({top:.3f}); "
                            + ("the saves before it are" if its.index(early) > 1 else "the save before it is")
                            + " clearly lower.")
            else:
                say("info", f"{tag(early)} is the earliest save close to the run's best likeness ({v:.3f} vs "
                            f"{top:.3f}, within 5% of the gain over the base model).")
        last = its[-1]
        if learned and ids[last["key"]] < top - 0.15 * span and last is not early:
            say("warn", f"likeness falls after its peak ({ids[last['key']]:.3f} at {tag(last)}): a sign of "
                        "overtraining.")
        var = [summary[it["key"]].get("variety_rel", NAN) for it in its]
        if finite(var[0]) and finite(var[-1]) and var[-1] < var[0] - 0.15:
            say("warn", f"prompt variety drops from {100 * var[0]:.0f}% to {100 * var[-1]:.0f}% of the base model's: "
                        "later saves look more alike whatever the prompt.")
        if has_ctrl:
            lk = [summary[it["key"]].get("leak", NAN) for it in its]
            if finite(lk[0]) and finite(lk[-1]) and lk[-1] > lk[0] + 0.05:
                say("warn", f"leak rises from {lk[0]:.3f} to {lk[-1]:.3f}: the character shows up more in prompts "
                            "without the trigger.")
    if multi:
        bests = []
        for run in runs:
            its = [it for it in items if it["run"] == run and it["key"] in ids]
            if its:
                top = max(its, key=lambda it: ids[it["key"]])
                bests.append((ids[top["key"]], run, top, len(its)))
        bests.sort(key=lambda x: -x[0])
        out.append(("info", "Best of each run: " + "; ".join(f"{run} {v:.3f}" + (f" ({tag(it)})" if n > 1 else "")
                                                             for v, run, it, n in bests) + "."))
    return out


# ---------------------------------------------------------------------- report


def fmt(x, spec=".2f", na="-"):
    return format(x, spec) if finite(x) else na


def text_report(profile, lora_name, info: LoraInfo, nb, variants, summary, rows, side_info, final, settings,
                show_side=True) -> str:
    out = [f"LoRA Prober ({profile.label}): {lora_name}", settings, ""]
    total = info.total_energy()
    energy = info.block_energy(nb)
    out.append("Weight change by block (share of total):")
    out.append("  " + "  ".join(f"{b}:{100 * energy[b] / total:.1f}%" for b in range(nb)))
    if profile.side is not None and show_side:
        out.append(f"  {profile.side.label}: " + "  ".join(
            f"{lab}:{100 * e / total:.1f}%" for lab, e in zip(profile.side.labels, info.side_energy())))
    shown = [m for m in profile.order if info.module_energy(m) > 0
             or (profile.layer(m).default and not profile.layer(m).optional)]
    out.append("  by layer: " + "  ".join(f"{m} {100 * info.module_energy(m) / total:.0f}%" for m in shown))
    out.append("")
    out.append(f"{'variant':<16} {'what':<40} {'params':>6} {'id_sim':>7} {'id_kept':>8} {'leak':>6} {'leak_k':>7} "
               f"{'drift':>6} {'drift_r':>7} {'variety':>8}")
    for v in variants:
        s = summary.get(v.name)
        if not s:
            continue
        kept = fmt(s["id_kept"]) + (f"±{s['id_kept_se']:.2f}" if finite(s["id_kept_se"]) else "")
        var = s.get("variety_rel", NAN)
        out.append(f"{v.name:<16} {v.describe(profile, nb, show_side):<40.40} {fmt(100 * v.param_share, '.0f'):>5}% "
                   f"{fmt(s['id_sim'], '.3f'):>7} {kept:>8} {fmt(s['leak'], '.3f'):>6} {fmt(s['leak_kept']):>7} "
                   f"{fmt(s['drift'], '.3f'):>6} {fmt(s['drift_rel']):>7} "
                   f"{(fmt(100 * var, '.0f') + '%') if finite(var) else '-':>8}")
    if rows or side_info:
        out.append("")
        out.append(f"{'group':<12} {'necessity':>9} {'sufficiency':>11} {'leak(alone)':>11} {'drift(alone)':>12}")
        for r in list(rows) + ([side_info] if side_info else []):
            out.append(f"{r['tag']:<12} {fmt(r['necessity']):>9} {fmt(r['sufficiency']):>11} "
                       f"{fmt(r['leak']):>11} {fmt(r['drift']):>12}")
    out.append("")
    out.append(final["headline"])
    for line in final["reasons"]:
        out.append("  - " + line)
    out.append("")
    out.append(final["yaml"])
    out.append("")
    out.append("Inference-time ablation shows where this LoRA stores identity. Retrain with the config "
               "above and compare: a LoRA restricted at training time relocates what it learns.")
    return "\n".join(out)
