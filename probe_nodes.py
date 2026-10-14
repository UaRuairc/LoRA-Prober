"""ComfyUI nodes: LoRA Prober, LoRA Prober Weight Report, LoRA Prober Loader (Block Select).

Supported models: Qwen-Image 2.1, Krea 2 and MiniMax H3 (see probe_core.PROFILES). MiniMax H3 renders are short
video clips scored on their first frame. The probe can run on up to four LoRA files (e.g. checkpoints of one
training run) and compare them; compare_node.py has the LoRA Prober Compare node for any number of files and videos.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import os
import re
import time
import types
from collections import OrderedDict, defaultdict

import numpy as np
import torch
from PIL import Image

import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.sd
import comfy.utils
import folder_paths
import latent_preview
import nodes as comfy_nodes

from . import media
from . import probe_core as core
from . import render_cache as rc
from .identity import ArcFaceScorer, ClipVisionScorer, drift, gray_small, layout_thumb, reference_centroid, thumbnail
from .render import (BAD, BG, GOOD, LEAK, MUTED, WARN, render_comparison, render_contact_sheet, render_diagnostic,
                     render_weight_report)

log = logging.getLogger("lora_prober")
NAN = float("nan")
CATEGORY = "LoRA Prober"
MODELS_TEXT = ", ".join(p.label for p in core.PROFILES.values())
SHEET_ROWS = 36  # variants per contact sheet; more are split into parts, each with base and full on top
IMAGE_OUTPUT_PIXELS = 32_000_000  # cap for one IMAGE output batch
CACHE_TOLERANCE = 2.0  # mean difference (0-255) of 32x32 thumbnails above which a cached render is stale
CLIP_FRAME = 0  # MiniMax H3 probe clips are scored on this frame (the first has its own latent: the sharpest)
SELECTION_TIP = ("A selection is blocks:layers. 12:mlp = only block 12's MLP; -12:mlp = the whole LoRA without "
                 "block 12's MLP; join parts with + (8-15:mlp + 12:attn); 'A except B' (8-23 except 12:attn). "
                 "Blocks: numbers and ranges, all, and txt for the text-side blocks (Krea 2 text fusion, MiniMax H3 "
                 "token refiner). Layers: attn, mlp, kv (not H3), all, or layer names (Qwen-Image 2.1: "
                 "q,k,v,o,mlp_in,mlp_out; Krea 2: q,k,v,gate,o,mlp_gate,mlp_up,mlp_down; MiniMax H3: "
                 "qkv,o,gate_compress,mlp_in,mlp_out,adaln).")
CLIP_TIP = ("The model's text encoder (Qwen-Image 2.1: Qwen3-VL 8B, Load CLIP type qwen_image; Krea 2: Qwen3-VL 4B, "
            "type krea2; MiniMax H3: Qwen3-VL 32B, type minimax).")
VAE_TIP = "The model's VAE (MiniMax H3: the video VAE, minimax_h3_video_vae)."

DEFAULT_ID_PROMPTS = """photo of [trigger], close-up portrait, soft window light, plain background
photo of [trigger] smiling, three-quarter view, outdoors in a park, daylight
photo of [trigger] in a dark wool coat on a city street at night, neon reflections
photo of [trigger], side profile, studio lighting, grey backdrop
photo of [trigger] sitting at a cafe table holding a cup, warm afternoon light
photo of [trigger] laughing, candid snapshot on a beach, wind in the hair"""

DEFAULT_CTRL_PROMPTS = """photo of a person, close-up portrait, soft window light, plain background
photo of a person in a dark wool coat on a city street at night, neon reflections
photo of two friends talking at a cafe table, warm afternoon light
a quiet mountain lake at sunrise, mist over the water"""

METRICS = ["auto", "ArcFace (faces)", "CLIP Vision (stylised)"]

_LORA_CACHE = {}
_RENDER_CACHE = OrderedDict()
_RENDER_CACHE_MAX = 3000
_SCORERS = {}


# ---------------------------------------------------------------------- helpers


def _load_lora(lora_name):
    path = folder_paths.get_full_path_or_raise("loras", lora_name)
    mtime = os.path.getmtime(path)
    hit = _LORA_CACHE.get(path)
    if hit and hit[0] == mtime:
        return path, mtime, hit[1], hit[2]
    sd, meta = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
    _LORA_CACHE.clear()
    _LORA_CACHE[path] = (mtime, sd, meta)
    return path, mtime, sd, meta


def _lora_header(path):
    """(tensor names, metadata) without loading the tensors when the file is a .safetensors."""
    if path.lower().endswith(".safetensors"):
        try:
            from safetensors import safe_open

            with safe_open(path, framework="pt", device="cpu") as f:
                return list(f.keys()), dict(f.metadata() or {})
        except Exception:
            pass
    sd, meta = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
    return list(sd.keys()), dict(meta or {})


def _pil_to_image(img: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(img.convert("RGB")).astype(np.float32) / 255.0)[None]


def _image_batch(pils, max_pixels=IMAGE_OUTPUT_PIXELS):
    """PIL images of different sizes -> one IMAGE batch, padded with the background colour. A batch over
    max_pixels is scaled down (float images take 12 bytes a pixel); the saved PNGs keep full size."""
    W, H = max(p.width for p in pils), max(p.height for p in pils)
    scale = min(1.0, math.sqrt(max_pixels / float(W * H * len(pils))))
    out = []
    for p in pils:
        if p.size != (W, H):
            canvas = Image.new("RGB", (W, H), BG)
            canvas.paste(p.convert("RGB"), (0, 0))
            p = canvas
        if scale < 1.0:
            p = p.resize((max(1, round(W * scale)), max(1, round(H * scale))), Image.LANCZOS)
        out.append(_pil_to_image(p))
    return torch.cat(out)


def _save_temp_preview(img: Image.Image, prefix: str):
    out_dir = folder_paths.get_temp_directory()
    os.makedirs(out_dir, exist_ok=True)
    name = f"{prefix}_{int(time.time() * 1000)}.png"
    img.save(os.path.join(out_dir, name))
    return {"filename": name, "subfolder": "", "type": "temp"}


def _jpeg(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _unjpeg(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("._") or "lora"


def _resolve_profile(family, model, sd_keys):
    """Pick the model profile: explicit choice, else the ComfyUI model class, else the LoRA's layer names."""
    keys = list(sd_keys)
    warnings = []
    if family and family != "auto":
        profile = core.profile_by_label(family)
        if profile is None:
            raise ValueError(f"Unknown model family '{family}'. Supported: {MODELS_TEXT}.")
    else:
        cls = type(model.model).__name__ if model is not None else None
        profile = core.profile_for_model_class(cls) if cls else None
        if profile is None:
            profile = core.detect_profile(keys)
            if profile is None:
                raise ValueError(f"Could not tell which model this LoRA is for. Supported: {MODELS_TEXT}.")
            if cls:
                warnings.append(f"The model is {cls}; treating it as {profile.label} from the LoRA's layer names.")
    share = core.recognised_share(profile, keys)
    best = core.detect_profile(keys)
    if share == 0:
        hint = f" It looks like a {best.label} LoRA." if best is not None and best is not profile else ""
        raise ValueError(f"None of this LoRA's layers match {profile.label}.{hint}")
    if best is not None and best is not profile and core.recognised_share(best, keys) > share + 0.2:
        warnings.append(f"This LoRA's layer names fit {best.label} better than {profile.label}.")
    return profile, warnings


def _num_blocks(model, profile):
    try:
        return len(model.get_model_object(f"diffusion_model.{profile.block_attr}"))
    except Exception:
        return profile.default_blocks


def _module_at(root, path):
    obj = root
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def _model_layers(model, profile):
    """Layer shorts that exist in the loaded model's blocks (e.g. MiniMax H3's to_gate_compress only in FastH3),
    or None when the model cannot be inspected."""
    try:
        block = model.get_model_object(f"diffusion_model.{profile.block_attr}")[0]
    except Exception:
        return None
    present = set()
    for l in profile.layers:
        try:
            if getattr(_module_at(block, l.path), "weight", None) is not None:
                present.add(l.short)
        except (AttributeError, IndexError, TypeError):
            pass
    return present or None


def _adapter_shape(sd, parts):
    """(out, in) of the weight change one layer's adapter makes, or None when it cannot tell."""
    def t(*names):
        for n in names:
            if n in parts:
                return sd[parts[n]]
        return None

    a, b = t("lora_A.weight", "lora_down.weight"), t("lora_B.weight", "lora_up.weight")
    if a is not None and b is not None:
        return (int(b.shape[0]), int(a.shape[1])) if a.ndim == 2 and b.ndim == 2 else None
    d = t("diff")
    if d is not None and d.ndim == 2:
        return int(d.shape[0]), int(d.shape[1])

    def factor(full, fa, fb):
        x = t(full)
        if x is not None:
            return tuple(x.shape)
        xa, xb = t(fa), t(fb)
        return (int(xa.shape[0]), int(xb.shape[1])) if xa is not None and xb is not None else None

    w1, w2 = factor("lokr_w1", "lokr_w1_a", "lokr_w1_b"), factor("lokr_w2", "lokr_w2_a", "lokr_w2_b")
    if w1 and w2 and len(w1) == 2 and len(w2) == 2:
        return w1[0] * w2[0], w1[1] * w2[1]
    return None


def _fit_lora(model, profile, sd, info=None):
    """The LoRA without adapters that cannot apply to this model: layers the model does not have (to_gate_compress
    from a FastH3 LoRA on MiniMax H3) or whose shape differs (adaln trained on the full MiniMax H3, on a pruned
    checkpoint). ComfyUI would skip them with errors anyway. Returns (state dict, notes for the report)."""
    if model is None:
        return sd, []
    info = info or core.analyze_lora(profile, sd)
    drop = set()
    missing, mismatch = defaultdict(set), defaultdict(set)
    for (g, b, short, path), parts in info.modules.items():
        layer = profile.layer(short)
        if path != layer.path:  # another trainer's name for it: ComfyUI maps those (e.g. split diffusers MLPs)
            continue
        where = f"{profile.canonical_block.format(b=b)}" if g == "main" else profile.side.blocks[b]
        try:
            w = getattr(model.get_model_object(f"diffusion_model.{where}.{layer.path}"), "weight", None)
        except Exception:
            w = None
        label = f"{b}" if g == "main" else profile.side.labels[b]
        if w is None:
            missing[short].add(label)
            drop.update(parts.values())
            continue
        shape = _adapter_shape(sd, parts)
        if shape is not None and tuple(int(x) for x in w.shape[:2]) != shape:
            mismatch[short].add(label)
            drop.update(parts.values())
    if not drop:
        return sd, []

    def where_text(labels):
        nums = sorted(int(x) for x in labels if str(x).isdigit())
        named = sorted(x for x in labels if not str(x).isdigit())
        parts = ([f"block{'s' if len(nums) > 1 else ''} {core.compact_ranges(nums)}"] if nums else []) + named
        return ", ".join(parts)

    notes = []
    for short, labels in sorted(missing.items()):
        notes.append(f"Left out {short} in {where_text(labels)}: this checkpoint has no such layer"
                     + (" (a LoRA trained on FastH3?)" if short == "gate_compress" else "") + ".")
    for short, labels in sorted(mismatch.items()):
        notes.append(f"Left out {short} in {where_text(labels)}: the LoRA's shape does not fit this checkpoint"
                     + (" (adaln trained on the full MiniMax H3 does not fit the pruned checkpoints, which store "
                        "adaln in a smaller form)" if short == "adaln" else "") + ".")
    return {k: v for k, v in sd.items() if k not in drop}, notes


def _encode(profile, clip, text):
    if profile.encode == "qwen21":  # same call as ComfyUI's Text Encode Qwen Image 2.1 node
        try:
            tokens = clip.tokenize(text, images=[], keep_vision=True, prevent_empty_text=True)
        except TypeError:
            tokens = clip.tokenize(text)
    else:  # same call as CLIPTextEncode, which ComfyUI's Krea 2 template uses
        tokens = clip.tokenize(text)
    return clip.encode_from_tokens_scheduled(tokens)


def _negative(profile, clip, text, any_positive):
    if profile.encode == "plain" and not (text or "").strip():
        # the Krea 2 template feeds ConditioningZeroOut of the prompt as the negative
        return comfy_nodes.ConditioningZeroOut().zero_out(any_positive)[0]
    return _encode(profile, clip, text or "")


def _empty_latent(model, width, height, profile=None, length=5):
    """Exactly what Empty Latent Image + KSampler produce for this model (MiniMax H3: Empty MiniMax H3 AV Latent
    with `length` frames)."""
    if profile is not None and profile.video:
        return media.h3_empty_latent(width, height, length)[0]
    device = comfy.model_management.intermediate_device()
    dtype = comfy.model_management.intermediate_dtype() if hasattr(comfy.model_management, "intermediate_dtype") else None
    kw = {"device": device, **({"dtype": dtype} if dtype is not None else {})}
    try:
        return comfy.sample.fix_empty_latent_channels(model, torch.zeros([1, 4, height // 8, width // 8], **kw), 8)
    except TypeError:  # older ComfyUI: build the model's latent directly
        lf = model.get_model_object("latent_format")
        ratio = getattr(lf, "spacial_downscale_ratio", 8)
        lat = torch.zeros([1, lf.latent_channels, height // ratio, width // ratio], **kw)
        return lat.unsqueeze(2) if getattr(lf, "latent_dimensions", 2) == 3 else lat


def _prompt_lines(text, trigger):
    out = []
    for ln in (text or "").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        if "[trigger]" in ln:
            if not trigger.strip():
                raise ValueError("The prompts use [trigger]; set trigger_word.")
            ln = ln.replace("[trigger]", trigger.strip())
        out.append(ln)
    return out


def _make_scorer(metric, clip_vision):
    if metric in ("auto", "ArcFace (faces)"):
        try:
            if "arcface" not in _SCORERS:
                _SCORERS["arcface"] = ArcFaceScorer(os.path.join(folder_paths.models_dir, "insightface"))
            return _SCORERS["arcface"]
        except Exception as e:
            if metric != "auto" or clip_vision is None:
                raise RuntimeError(
                    "ArcFace needs the insightface and onnxruntime packages "
                    f"({type(e).__name__}: {e}). Install them, or connect a CLIP Vision model and pick "
                    "'CLIP Vision (stylised)'."
                ) from e
            log.warning("ArcFace unavailable (%s); using CLIP Vision", e)
    if clip_vision is None:
        raise RuntimeError("'CLIP Vision (stylised)' needs a CLIP Vision model connected to clip_vision.")
    return ClipVisionScorer(clip_vision)


def _cache_get(key):
    entry = _RENDER_CACHE.get(key)
    if entry is not None:
        _RENDER_CACHE.move_to_end(key)
    return entry


def _cache_put(key, entry):
    _RENDER_CACHE[key] = entry
    while len(_RENDER_CACHE) > _RENDER_CACHE_MAX:
        _RENDER_CACHE.popitem(last=False)


def _heat_layers(profile, info, present=None):
    """Heatmap rows: the layer types the LoRA has, plus the ones ai-toolkit trains by default that this model has
    (an empty row shows a layer type the LoRA left out)."""
    has = {s for (_, _, s) in info.sq}
    out = []
    for m in profile.order:
        layer = profile.layer(m)
        exists = (m in present) if present is not None else not layer.optional
        if m in has or (layer.default and exists):
            out.append(m)
    return out


def _heat(profile, info, nb, show_side, present=None):
    total = info.total_energy()
    cols = [("main", b) for b in range(nb)]
    labels = [str(b) for b in range(nb)]
    side_start = None
    if profile.side is not None and show_side:
        side_start = nb
        cols += [(profile.side.key, i) for i in range(len(profile.side.blocks))]
        labels += list(profile.side.labels)

    def cell(g, b, m):
        v = info.sq.get((g, b, m))
        return math.sqrt(v) if v is not None and core.finite(v) else None

    rows = _heat_layers(profile, info, present)
    return dict(
        modules=rows,
        values=[[cell(g, b, m) for g, b in cols] for m in rows],
        block_share=[info.energy(g, b) / total for g, b in cols],
        labels=labels,
        side_start=side_start,
        side_label=profile.side.label if side_start is not None else "",
    )


def _highlight_cols(profile, nb, blocks, side, show_side):
    cols = set(blocks)
    if profile.side is not None and show_side and side:
        cols |= set(range(nb, nb + len(profile.side.blocks)))
    return cols


def _init_label(patcher):
    """File name of a model loaded by a core loader (from its cached_patcher_init), else None."""
    init = getattr(patcher, "cached_patcher_init", None)
    for a in (init[1] if init else ()):
        if isinstance(a, str) and os.path.isfile(a):
            return os.path.basename(a)
    return None


_SPEED_LORA = {}


def _apply_speed_lora(model, name, strength, warnings):
    """Patch the base model with the speed LoRA. Returns (model, cache id, label)."""
    if not name or name == "none" or strength == 0:
        return model, None, None
    path = folder_paths.get_full_path_or_raise("loras", name)
    mtime = os.path.getmtime(path)
    hit = _SPEED_LORA.get(path)
    if not hit or hit[0] != mtime:
        _SPEED_LORA.clear()
        _SPEED_LORA[path] = hit = (mtime, comfy.utils.load_torch_file(path, safe_load=True))
    before = sum(len(v) for v in model.patches.values())
    patched = comfy.sd.load_lora_for_models(model, None, hit[1], strength, 0)[0]
    if sum(len(v) for v in patched.patches.values()) == before:
        warnings.append(f"The speed LoRA {os.path.basename(name)} matched no layers of this model, so it does nothing.")
    return patched, [rc.file_id(path), round(float(strength), 6)], f"{os.path.basename(name)} @ {strength:g}"


def _check_text(ctx):
    if ctx.check is None:
        return ""
    if ctx.fresh:
        return " (the cached renders did not match a fresh one, so all were rendered again)"
    return " (cache checked against 1 fresh render)"


def _has_side_keys(profile, keys):
    if profile.side is None:
        return False
    for k in keys:
        pk = core.parse_key(profile, k)
        if pk.group not in (None, "main") and pk.short is not None:
            return True
    return False


# ------------------------------------------------------------------- the sweep


class _Progress:
    """One progress bar across every checkpoint's renders (each render = `steps` ticks)."""

    def __init__(self, steps, node_id):
        self.steps = max(1, int(steps))
        self.node_id = node_id
        self.planned = 0
        self.done = 0
        self.pbar = None

    def _total(self):
        return max(1, self.planned * self.steps)

    def _show(self, value, preview=None):
        if self.pbar is None:
            self.pbar = comfy.utils.ProgressBar(self._total(), node_id=self.node_id)
        self.pbar.update_absolute(min(value, self._total()), self._total(), preview)

    def add(self, renders):
        self.planned += renders
        self._show(self.done * self.steps)

    def release(self, renders):
        """Drop planned renders that will not happen (unused test rounds)."""
        self.planned = max(self.done, self.planned - max(0, renders))
        self._show(self.done * self.steps)

    def in_render(self, step, preview=None):
        self._show(self.done * self.steps + step, preview)

    def render_done(self):
        self.done += 1
        self.planned = max(self.planned, self.done)
        self._show(self.done * self.steps)

    def finish(self):
        self.planned = self.done
        self._show(self.done * self.steps)


class _Runner:
    """Renders prompt x seed for variants of one LoRA file, with previews and cancel. Each image comes from
    the memory cache, else the disk cache, else a fresh render (which is then stored in both)."""

    def __init__(self, ctx, sd, lora_path=None, lora_name=None):
        self.ctx = ctx
        self.sd = sd
        self.lora_name = lora_name
        self.lora_fid = rc.file_id(lora_path) if lora_path else None
        self.rendered = 0
        self.from_mem = 0
        self.from_disk = 0

    def key(self, variant, prompt, seed):
        ctx = self.ctx
        lora = None
        if variant.kind != "base":
            lora = [self.lora_fid, round(float(ctx.strength), 6),
                    hashlib.sha1("\n".join(sorted(variant.keys)).encode()).hexdigest()]
        parts = [ctx.model_fp, ctx.speed_id, ctx.vae_fp, lora, ctx.cond_fp[prompt], ctx.neg_fp, int(seed),
                 int(ctx.steps), round(float(ctx.cfg), 6), ctx.sampler_name, ctx.scheduler, int(ctx.width),
                 int(ctx.height)]
        if ctx.profile.video:  # one frame of a short clip (image models keep their old keys)
            parts.append(["clip", int(ctx.frames), CLIP_FRAME])
        return rc.key_of(parts)

    def _callback(self):
        ctx = self.ctx

        def cb(step, x0, x, total_steps):
            preview = None
            if ctx.previewer is not None:
                if getattr(x0, "is_nested", False):
                    x0 = x0.tensors[0]
                preview = ctx.previewer.decode_latent_to_preview_image("JPEG", x0)
            ctx.progress.in_render(step + 1, preview)

        return cb

    def sample(self, patched, prompt, seed):
        """One render as the uint8 pixels that are stored and scored."""
        ctx = self.ctx
        noise = comfy.sample.prepare_noise(ctx.latent, seed)
        samples = comfy.sample.sample(patched, noise, ctx.steps, ctx.cfg, ctx.sampler_name, ctx.scheduler,
                                      ctx.conds[prompt], ctx.neg, ctx.latent, denoise=1.0,
                                      callback=self._callback(), disable_pbar=True, seed=seed)
        frames = media.decode_frames(ctx.vae, samples)  # an image is one frame; an H3 clip is scored on CLIP_FRAME
        return rc.to_u8(frames[min(CLIP_FRAME, frames.shape[0] - 1)])

    def make_entry(self, u8, key, kind):
        img = rc.from_u8(u8)
        return {"key": key, "thumb": _jpeg(thumbnail(img, 168)), "gray": gray_small(img) if kind == "ctrl" else None,
                "lay": layout_thumb(img), "embeds": {self.ctx.scorer.key: self.ctx.scorer.embed_render(img)},
                "path": None}

    def meta(self, variant, kind, idx, prompt, seed):
        ctx = self.ctx
        return {"model": ctx.model_label, "speed_lora": ctx.speed_label, "vae": ctx.vae_label,
                "lora": None if variant.kind == "base" else self.lora_name,
                "lora_strength": None if variant.kind == "base" else ctx.strength,
                "layers": variant.describe(ctx.profile, ctx.nb, True), "prompt": prompt, "seed": seed,
                "steps": ctx.steps, "cfg": ctx.cfg, "sampler": ctx.sampler_name, "scheduler": ctx.scheduler,
                "width": ctx.width, "height": ctx.height,
                **({"clip_frames": ctx.frames, "frame": CLIP_FRAME} if ctx.profile.video else {})}

    def store(self, key, u8, entry, meta):
        ctx = self.ctx
        if ctx.disk is not None:
            entry["path"] = ctx.disk.save(key, u8, entry, meta, ctx.scorer.key)
        _cache_put(key, entry)
        ctx.run_keys.add(key)

    def lookup(self, key):
        """(entry, where) from memory or disk, with this scorer's embeddings, or (None, None)."""
        ctx = self.ctx
        if ctx.fresh and key not in ctx.run_keys:
            return None, None
        entry, where = _cache_get(key), "mem"
        if entry is None and ctx.disk is not None:
            entry, where = ctx.disk.load(key), "disk"
        if entry is None:
            return None, None
        if rc.DiskCache.embeds_for(entry, ctx.scorer.key) is None:  # new scorer: score the stored image
            u8 = ctx.disk.load_image(key) if ctx.disk is not None and entry.get("path") else None
            if u8 is None:
                return None, None
            emb = ctx.scorer.embed_render(rc.from_u8(u8))
            entry["embeds"][ctx.scorer.key] = emb
            ctx.disk.save_embeds(key, ctx.scorer.key, emb)
        _cache_put(key, entry)
        return entry, where

    def render(self, variant):
        ctx = self.ctx
        out, patched = [], None
        for kind, idx, prompt in ctx.prompts:
            for seed in ctx.seeds:
                comfy.model_management.throw_exception_if_processing_interrupted()
                key = self.key(variant, prompt, seed)
                entry, where = self.lookup(key)
                if entry is not None and kind == "ctrl" and entry.get("gray") is None:
                    # stored as an identity render (same prompt in both lists): add the drift thumbnail
                    u8 = ctx.disk.load_image(key) if ctx.disk is not None and entry.get("path") else None
                    if u8 is None:
                        entry = None
                    else:
                        entry["gray"] = gray_small(rc.from_u8(u8))
                if entry is None:
                    if patched is None:
                        if variant.kind == "base":
                            patched = ctx.model
                        else:
                            sub = {k: self.sd[k] for k in variant.keys}
                            patched = comfy.sd.load_lora_for_models(ctx.model, None, sub, ctx.strength, 0)[0]
                    u8 = self.sample(patched, prompt, seed)
                    entry = self.make_entry(u8, key, kind)
                    self.store(key, u8, entry, self.meta(variant, kind, idx, prompt, seed))
                    self.rendered += 1
                elif where == "disk":
                    self.from_disk += 1
                elif key == ctx.check_fresh:  # the cache check's own render, made fresh for this run
                    ctx.check_fresh = None
                    self.rendered += 1
                else:
                    self.from_mem += 1
                ctx.progress.render_done()
                out.append(dict(variant=variant.name, kind=kind, prompt=idx, seed=seed, entry=entry))
        del patched
        return out

    def counts_text(self):
        return _counts_text(self.rendered, self.from_disk, self.from_mem)


def _counts_text(rendered, from_disk, from_mem):
    parts = [f"{rendered} rendered"]
    if from_disk:
        parts.append(f"{from_disk} from the disk cache")
    if from_mem:
        parts.append(f"{from_mem} from memory")
    return ", ".join(parts)


def _records(results, scorer, centroid):
    base_gray = {(r["prompt"], r["seed"]): r["entry"]["gray"]
                 for r in results if r["variant"] == "base" and r["kind"] == "ctrl"}
    recs = []
    for r in results:
        sim, nf = scorer.similarity(rc.DiskCache.embeds_for(r["entry"], scorer.key), centroid)
        d = NAN
        if r["kind"] == "ctrl":
            if r["variant"] == "base":
                d = 0.0
            elif "drift" in r["entry"]:  # same base twin every time for these settings
                d = r["entry"]["drift"]
            else:
                g0, g1 = base_gray.get((r["prompt"], r["seed"])), r["entry"]["gray"]
                if g0 is not None and g1 is not None:
                    d = r["entry"]["drift"] = drift(g1, g0)
        recs.append(dict(variant=r["variant"], kind=r["kind"], prompt=r["prompt"], seed=r["seed"],
                         sim=sim, n_faces=nf, drift=d))
    return recs


def _summarize(ctx, results, names):
    recs = _records(results, ctx.scorer, ctx.centroid)
    summary = core.summarize(recs, names)
    core.add_variety(summary, [dict(variant=r["variant"], kind=r["kind"], prompt=r["prompt"], seed=r["seed"],
                                    lay=r["entry"].get("lay")) for r in results])
    return recs, summary


# ------------------------------------------------------------------------ nodes


class LoRAProber:
    @classmethod
    def INPUT_TYPES(cls):
        loras = folder_paths.get_filename_list("loras")
        compare_tip = ("Another LoRA file to probe with exactly the same settings and compare with lora_name, e.g. "
                       "an earlier or later checkpoint of the same training run (saves from one ai-toolkit run are put "
                       "in step order). Each one costs a full probe; the no-LoRA renders are shared.")
        return {
            "required": {
                "model": ("MODEL", {"tooltip": f"The base model WITHOUT the LoRA under test ({MODELS_TEXT})."}),
                "clip": ("CLIP", {"tooltip": CLIP_TIP}),
                "vae": ("VAE", {"tooltip": VAE_TIP}),
                "lora_name": (loras, {"tooltip": "The character LoRA to probe."}),
                "reference_images": ("IMAGE", {"tooltip": "A batch of photos of the character, used to measure identity."}),
                "trigger_word": ("STRING", {"default": "", "tooltip": "Replaces [trigger] in the identity prompts."}),
                "identity_prompts": ("STRING", {"multiline": True, "default": DEFAULT_ID_PROMPTS,
                                                "tooltip": "One prompt per line, each with the trigger. Mostly photographic: face recognition is unreliable on stylised renders."}),
                "control_prompts": ("STRING", {"multiline": True, "default": DEFAULT_CTRL_PROMPTS,
                                               "tooltip": "One prompt per line, WITHOUT the trigger: other people and scenes the LoRA should leave alone."}),
                "base_seed": ("INT", {"default": 1, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                                      "tooltip": "Seeds used are base_seed, base_seed+1, ... for every prompt and variant."}),
                "seeds_per_prompt": ("INT", {"default": 2, "min": 1, "max": 8}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100,
                                  "tooltip": "Use what you normally use: Qwen-Image 2.1 about 20-25, Krea 2 Turbo 8, "
                                             "MiniMax H3 20 with res_multistep (8 with FastH3 or the turbo LoRA)."}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1, "round": 0.01,
                                  "tooltip": "Qwen-Image 2.1, Krea 2 Turbo and MiniMax H3 (guidance-distilled) are "
                                             "normally sampled at cfg 1 (no negative prompt)."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "width": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 32}),
                "height": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 32}),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05}),
                "sweep": (core.SWEEPS, {"default": "drop + keep groups",
                                        "tooltip": "Which block-group ablations to render. drop + keep is the most informative."}),
                "group_size": ("INT", {"default": 4, "min": 1, "max": 16,
                                       "tooltip": "Consecutive blocks per group (4 -> 8 groups for 32 blocks, 7 for "
                                                  "Krea 2's 28, 13 for MiniMax H3's 50)."}),
                "layer_variants": ("BOOLEAN", {"default": True,
                                               "tooltip": "Also test attention-only, MLP-only and k/v-only (not H3), and "
                                                          "the text-side blocks alone and removed (Krea 2 text fusion, "
                                                          "MiniMax H3 token refiner)."}),
                "identity_metric": (METRICS, {"default": "auto",
                                              "tooltip": "auto: ArcFace if insightface is installed, else CLIP Vision (connect one)."}),
                "target_identity": ("FLOAT", {"default": 0.9, "min": 0.5, "max": 1.0, "step": 0.01,
                                              "tooltip": "Share of the full LoRA's identity gain a subset must keep to count as good enough."}),
                "test_rounds": ("INT", {"default": 3, "min": 0, "max": 8,
                                        "tooltip": "How many suggested subsets to render and check; each miss widens the next one."}),
            },
            "optional": {
                "clip_vision": ("CLIP_VISION", {"tooltip": "For stylised / non-human characters, or when insightface is missing."}),
                "negative_prompt": ("STRING", {"default": "", "tooltip": "Only matters when cfg > 1."}),
                "custom_variants": ("STRING", {"multiline": True, "default": "",
                                               "tooltip": "Extra variants, one per line: name=selection, e.g. "
                                                          "b12_mlp=12:mlp, no_b12_mlp=-12:mlp, mid_attn=8-23:attn. "
                                                          + SELECTION_TIP}),
                "model_family": (core.FAMILY_CHOICES, {"default": "auto",
                                                       "tooltip": "auto detects the model from the loaded checkpoint."}),
                "compare_lora_2": (["none"] + loras, {"default": "none", "tooltip": compare_tip}),
                "compare_lora_3": (["none"] + loras, {"default": "none", "tooltip": compare_tip}),
                "compare_lora_4": (["none"] + loras, {"default": "none", "tooltip": compare_tip}),
                "speed_lora": (["none"] + loras, {"default": "none",
                                                  "tooltip": "A LoRA applied to the base model for every render, "
                                                             "including the no-LoRA baseline: e.g. a turbo / lightning "
                                                             "LoRA so you can sample in a few steps. Part of the cache "
                                                             "key, so its renders are kept apart."}),
                "speed_lora_strength": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05}),
                "disk_cache": ("BOOLEAN", {"default": True,
                                           "tooltip": "Keep every render on disk (output/lora_prober/cache) and reuse it "
                                                      "whenever the same image is asked for again, even after a "
                                                      "restart. Each run folder also gets the images by variant in "
                                                      "renders/."}),
                "video_length": ("INT", {"default": 5, "min": 5, "max": 3600, "step": 17,
                                         "tooltip": "MiniMax H3 only: frames per probe render, scored on the first "
                                                    "frame. 5 (the minimum) keeps the sweep fast; use the LoRA Prober Compare "
                                                    "node for full-length videos."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("diagnostic", "contact_sheet", "report", "ai_toolkit_yaml", "comparison")
    OUTPUT_TOOLTIPS = (
        "One diagnostic per LoRA file.",
        "Contact sheets: one per seed (with several LoRA files, the side-by-side comparison sheets).",
        "Text report (comparison first when several LoRA files are probed).",
        "Suggested ai-toolkit network settings for lora_name.",
        "Checkpoint comparison (the diagnostic itself when only one LoRA file is probed).",
    )
    FUNCTION = "run"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = ("Renders the same prompts and seeds with the LoRA split into block groups and layer types, "
                   "scores identity, leakage, drift and prompt variety, and suggests which layers to train in "
                   f"ai-toolkit. Can compare up to four checkpoints. Supports {MODELS_TEXT} (short clips, scored on "
                   "their first frame).")

    def run(self, model, clip, vae, lora_name, reference_images, trigger_word, identity_prompts, control_prompts,
            base_seed, seeds_per_prompt, steps, cfg, sampler_name, scheduler, width, height, lora_strength, sweep,
            group_size, layer_variants, identity_metric, target_identity, test_rounds,
            clip_vision=None, negative_prompt="", custom_variants="", model_family="auto",
            compare_lora_2="none", compare_lora_3="none", compare_lora_4="none", speed_lora="none",
            speed_lora_strength=1.0, disk_cache=True, video_length=5, unique_id=None):
        t0 = time.time()

        # ---- the LoRA files, read without loading their tensors
        names = []
        for n in (lora_name, compare_lora_2, compare_lora_3, compare_lora_4):
            if n and n != "none" and n not in names:
                names.append(n)
        cks = []
        for n in names:
            path = folder_paths.get_full_path_or_raise("loras", n)
            keys, meta = _lora_header(path)
            cks.append(dict(name=n, path=path, keys=keys, meta=meta, primary=(n == lora_name),
                            step=core.checkpoint_step(n, meta), run=core.checkpoint_run(n, meta)))
        profile, warnings = _resolve_profile(model_family, model, cks[0]["keys"])
        for ck in cks[1:]:
            if core.recognised_share(profile, ck["keys"]) == 0:
                best = core.detect_profile(ck["keys"])
                hint = f" (it looks like a {best.label} LoRA)" if best is not None else ""
                raise ValueError(f"{ck['name']} has no {profile.label} layers{hint}, so it cannot be compared "
                                 f"with {lora_name}.")
        same_run = len({ck["run"] for ck in cks}) == 1
        cks = core.order_checkpoints(cks)
        core.label_checkpoints(cks)
        multi = len(cks) > 1
        nb = _num_blocks(model, profile)
        present = _model_layers(model, profile)
        frames = media.h3_frame_count(video_length) if profile.video else 1
        if profile.video and (width % 32 or height % 32):
            raise ValueError(f"{profile.label} needs width and height in multiples of 32 (got {width}x{height}).")

        # ---- prompts, seeds, identity references
        trigger = trigger_word.strip()
        id_prompts = _prompt_lines(identity_prompts, trigger)
        ctrl_prompts = _prompt_lines(control_prompts, trigger)
        if not id_prompts:
            raise ValueError("Add at least one identity prompt.")
        if trigger and any(trigger.lower() not in p.lower() for p in id_prompts):
            warnings.append("Some identity prompts do not contain the trigger word.")
        if trigger and any(trigger.lower() in p.lower() for p in ctrl_prompts):
            warnings.append("Some control prompts contain the trigger word; leak numbers will be inflated.")
        prompts = [("id", i, p) for i, p in enumerate(id_prompts)] + [("ctrl", i, p) for i, p in enumerate(ctrl_prompts)]
        seeds = [base_seed + j for j in range(seeds_per_prompt)]
        scorer = _make_scorer(identity_metric, clip_vision)
        centroid, n_ref, ref_self = reference_centroid(scorer, reference_images)
        if centroid is None:
            raise ValueError(f"{scorer.name} found nothing usable in the reference images "
                             "(no face detected). Use clearer face photos, or switch to CLIP Vision.")

        # ---- plan every checkpoint's variants up front, so bad settings fail before any rendering
        n_renders = len(prompts) * len(seeds)
        planned = 0
        for ck in cks:
            ck["has_side"] = _has_side_keys(profile, ck["keys"])
            blocks = [pk.block for pk in (core.parse_key(profile, k) for k in ck["keys"]) if pk.group == "main"]
            if blocks and max(blocks) >= nb:
                raise ValueError(f"{ck['name']} has block {max(blocks)} but the model has {nb} blocks. "
                                 f"Is this a {profile.label} LoRA and model?")
            variants = core.build_variants(profile, nb, sweep, group_size, layer_variants, custom_variants,
                                           ck["has_side"])
            ck["n_variants"] = sum(1 for v in variants if v.kind == "base" or core.select_keys(profile, ck["keys"], v))
            planned += (ck["n_variants"] + test_rounds) * n_renders

        # ---- what decides the pixels, for the render cache (the model before the speed LoRA is applied)
        model_fp, vae_fp = rc.model_fingerprint(model), rc.vae_fingerprint(vae)
        model, speed_id, speed_label = _apply_speed_lora(model, speed_lora, speed_lora_strength, warnings)
        conds = {p: _encode(profile, clip, p) for _, _, p in prompts}
        neg = _negative(profile, clip, negative_prompt, next(iter(conds.values())))
        uses_neg = abs(cfg - 1.0) > 1e-9 or bool(model.model_options.get("disable_cfg1_optimization"))
        cache_dir = os.path.join(folder_paths.get_output_directory(), "lora_prober", "cache")
        ctx = types.SimpleNamespace(
            model=model, vae=vae, profile=profile, nb=nb, present=present, frames=frames, prompts=prompts,
            seeds=seeds, scorer=scorer, centroid=centroid, n_ref=n_ref, ref_self=ref_self, conds=conds, neg=neg,
            latent=_empty_latent(model, width, height, profile, frames), steps=steps, cfg=cfg, sampler_name=sampler_name,
            scheduler=scheduler, width=width, height=height, strength=lora_strength, sweep=sweep,
            group_size=group_size, layer_variants=layer_variants, custom_variants=custom_variants,
            target=target_identity, test_rounds=test_rounds, warnings=warnings, multi=multi, same_run=same_run,
            model_fp=model_fp, vae_fp=vae_fp, speed_id=speed_id, speed_label=speed_label,
            cond_fp={p: rc.cond_fingerprint(c) for p, c in conds.items()},
            neg_fp=rc.cond_fingerprint(neg) if uses_neg else None,
            model_label=_init_label(model) or type(model.model).__name__,
            vae_label=_init_label(getattr(vae, "patcher", None)) or type(vae).__name__,
            disk=rc.DiskCache(cache_dir) if disk_cache else None, fresh=False, run_keys=set(), check=None,
            check_fresh=None,
            previewer=latent_preview.get_previewer(model.load_device, model.model.latent_format),
            progress=_Progress(steps, unique_id),
            settings=(f"{profile.label} · {len(id_prompts)} identity + {len(ctrl_prompts)} control prompts x "
                      f"{len(seeds)} seeds · {width}x{height}"
                      + (f" · {frames}-frame clips (first frame scored)" if profile.video else "")
                      + f" · {steps} steps · {sampler_name}/{scheduler} · cfg {cfg:g} · strength {lora_strength:g}"
                      + (f" · speed LoRA {speed_label}" if speed_label else "") + f" · {scorer.name}"),
        )
        ctx.progress.add(planned)
        self._check_cache(ctx)
        log.info("LoRA Prober (%s): %d LoRA file(s), up to %d renders", profile.label, len(cks), planned)

        # ---- probe each checkpoint (no-LoRA renders come from the cache after the first)
        results = [self._probe_checkpoint(ctx, ck) for ck in cks]
        ctx.progress.finish()
        for res in results:
            self._render_checkpoint(ctx, res)
        primary = next(r for r in results if r["primary"])

        if not multi:
            res = results[0]
            ui_images = self._save_single(ctx, res)
            return {"ui": {"images": ui_images},
                    "result": (_pil_to_image(res["diag"]), _image_batch(res["sheets"]), res["report"], res["yaml"],
                               _pil_to_image(res["diag"]))}

        cmp = self._compare(ctx, results, time.time() - t0)
        report = cmp["text"] + "\n\n" + "\n\n".join(
            "=" * 100 + f"\n{r['label']} · {os.path.basename(r['name'])}\n" + "=" * 100 + "\n" + r["report"]
            for r in results)
        yaml = f"# suggested from {os.path.basename(primary['name'])} ({primary['label']})\n" + primary["yaml"]
        ui_images = self._save_multi(ctx, results, cmp, report)
        return {"ui": {"images": ui_images},
                "result": (_image_batch([r["diag"] for r in results]), _image_batch(cmp["sheets"]), report, yaml,
                           _pil_to_image(cmp["image"]))}

    # ------------------------------------------------------------------ cache check

    @staticmethod
    def _check_cache(ctx):
        """Render the first no-LoRA image fresh and compare it with the cached copy, if there is one. The cache
        key covers the model file, patches and options, but a mismatch means something changed that it cannot
        see; then this run renders everything again and the new renders replace the old ones."""
        runner = _Runner(ctx, None)
        kind, idx, prompt = ctx.prompts[0]
        seed = ctx.seeds[0]
        base = core.Variant("base", "base", [], [])
        key = runner.key(base, prompt, seed)
        cached, _ = runner.lookup(key)
        if cached is None:
            return
        u8 = runner.sample(ctx.model, prompt, seed)
        lay = layout_thumb(rc.from_u8(u8)).astype(np.float32)
        ctx.check = diff = float(np.abs(lay - np.asarray(cached["lay"], dtype=np.float32)).mean())
        if diff <= CACHE_TOLERANCE:
            log.info("LoRA Prober: cached renders match this model (difference %.2f/255)", diff)
            return
        ctx.fresh = True
        ctx.warnings.append(f"Cached renders did not match this model's output (difference {diff:.1f}/255), so "
                            "everything was rendered again and the old renders were replaced. Something about the "
                            "model changed that the cache key cannot see.")
        entry = runner.make_entry(u8, key, kind)
        runner.store(key, u8, entry, runner.meta(base, kind, idx, prompt, seed))
        ctx.check_fresh = key

    # ------------------------------------------------------------------ one checkpoint

    def _probe_checkpoint(self, ctx, ck):
        profile, nb = ctx.profile, ctx.nb
        t0 = time.time()
        path, mtime, sd, _meta = _load_lora(ck["name"])
        sd, fit_notes = _fit_lora(ctx.model, profile, sd)
        info = core.analyze_lora(profile, sd)
        if not info.sq:
            raise ValueError(f"{ck['name']} has no {profile.label} block layers"
                             + (" that fit this checkpoint" if fit_notes else "") + ".")
        has_side = info.has_side
        warnings = list(ctx.warnings) + fit_notes

        variants = core.build_variants(profile, nb, ctx.sweep, ctx.group_size, ctx.layer_variants,
                                       ctx.custom_variants, has_side)
        core.assign_keys(profile, sd, variants)
        skipped = [v.name for v in variants if v.kind != "base" and not v.keys]
        variants = [v for v in variants if v.kind == "base" or v.keys]
        if skipped:
            warnings.append(f"Skipped variants with no LoRA layers: {', '.join(skipped)}.")

        runner = _Runner(ctx, sd, path, ck["name"])
        results = []
        for v in variants:
            results += runner.render(v)
        names = [v.name for v in variants]
        recs, summary = _summarize(ctx, results, names)
        rows = core.group_rows(summary, variants, recs)
        side_info = core.side_row(profile, summary, variants, recs) if has_side else None
        sug = core.suggest(profile, summary, rows, nb, ctx.target, side_info, has_side)
        reasons = list(sug.reasons)
        gain = summary["full"]["id_sim"] - summary["base"]["id_sim"]
        reliable = core.finite(gain) and gain >= (0.1 if ctx.scorer.name == "ArcFace" else 0.03)

        # render the suggested subset; widen and retry while it misses the target
        tested, final, rounds = [], None, 0
        sel, remaining = core.Selection(sug.blocks, sug.modules, sug.side), list(sug.remaining)
        everything = core.is_everything(profile, sel, nb, has_side)
        for i in range(ctx.test_rounds if (reliable and not everything) else 0):
            v = core.Variant(f"suggest_{i + 1}", "suggest", list(sel.blocks), list(sel.modules), side=sel.side)
            core.assign_keys(profile, sd, [v])
            if not v.keys:
                break
            variants.append(v)
            names.append(v.name)
            results += runner.render(v)
            rounds += 1
            recs, summary = _summarize(ctx, results, names)
            kept = summary[v.name]["id_kept"]
            tested.append(v)
            if core.finite(kept) and kept >= ctx.target:
                final = v
                break
            nxt = core.widen(profile, sel, remaining, nb, sug.module_kept, sug.predicted, has_side, sug.side_effect)
            if nxt is None:
                break
            sel, remaining, why = nxt
            reasons.append(f"{v.name} kept {core.fmt(100 * kept, '.0f')}%, so the next test {why}")
            if core.is_everything(profile, sel, nb, has_side):
                break
        ctx.progress.release((ctx.test_rounds - rounds) * len(ctx.prompts) * len(ctx.seeds))
        rows = core.group_rows(summary, variants, recs)
        side_info = core.side_row(profile, summary, variants, recs) if has_side else None

        # final pick
        status = {}
        for v in tested:
            k = summary[v.name]["id_kept"]
            status[v.name] = "pass" if core.finite(k) and k >= ctx.target else "fail"
        if final is None and tested:
            final = max(tested, key=lambda v: summary[v.name]["id_kept"] if core.finite(summary[v.name]["id_kept"]) else -1)
        if not reliable:
            final = None
            f_sel = core.Selection(list(range(nb)), list(profile.order), has_side)
        elif final is not None:
            f_sel = core.Selection(final.blocks, final.modules, final.side)
        else:
            f_sel = sel
        f_side = f_sel.side and has_side
        restricted = not core.is_everything(profile, f_sel, nb, has_side)
        if restricted:
            yaml = core.ai_toolkit_yaml(profile, f_sel.blocks, f_sel.modules, f_side, nb, show_side=has_side,
                                        present=ctx.present)
        else:
            yaml = core.ai_toolkit_yaml(profile, list(range(nb)), list(profile.order), profile.side is not None, nb,
                                        show_side=has_side, present=ctx.present)
        if not reliable:
            what = ""
        elif restricted:
            what = core.describe_selection(profile, nb, f_sel.blocks, f_side, f_sel.modules, has_side)
        else:
            what = "all layers"
        tested_kept = summary[final.name]["id_kept"] if final is not None else NAN
        tested_pass = None if final is None else (core.finite(tested_kept) and tested_kept >= ctx.target)
        return dict(
            name=ck["name"], label=ck["label"], short=ck["short"], step=ck["step"], primary=ck["primary"], info=info,
            has_side=has_side, warnings=warnings, variants=variants, results=results, recs=recs, summary=summary,
            rows=rows, side_info=side_info, reasons=reasons, reliable=reliable, tested=tested, final=final,
            status=status, f_sel=f_sel, f_side=f_side, restricted=restricted, yaml=yaml, what=what,
            tested_kept=tested_kept, tested_pass=tested_pass,
            size_frac=core.lora_param_fraction(profile, f_sel.blocks, f_sel.modules, f_side, nb, ctx.present),
            rendered=runner.rendered, from_disk=runner.from_disk, from_mem=runner.from_mem,
            elapsed=time.time() - t0,
        )

    def _render_checkpoint(self, ctx, res):
        """Diagnostic image, contact sheets and text report for one checkpoint (stored in res)."""
        profile, nb, scorer = ctx.profile, ctx.nb, ctx.scorer
        summary, variants, rows, side_info = res["summary"], res["variants"], res["rows"], res["side_info"]
        has_side, f_sel, f_side = res["has_side"], res["f_sel"], res["f_side"]
        verdict, headline = self._verdict(profile, summary, rows, side_info, res["final"], f_sel, f_side, nb,
                                          res["restricted"], ctx.target, scorer, ctx.ref_self, res["warnings"],
                                          res["reliable"], has_side, ctx.present)
        base = os.path.basename(res["name"])
        title = base + (f" · {res['label']}" if ctx.multi and res["label"] != os.path.splitext(base)[0] else "")
        subtitle = [
            ctx.settings,
            f"{len(variants)} variants · {_counts_text(res['rendered'], res['from_disk'], res['from_mem'])}"
            f"{_check_text(ctx)} · {res['elapsed'] / 60:.1f} min · {ctx.n_ref} reference images "
            f"(self-similarity {ctx.ref_self:.2f})",
        ]
        chart_rows = list(rows) + ([side_info] if side_info else [])
        chosen_tags = []
        if res["restricted"]:
            chosen_tags = [r["tag"] for r in rows if set(r["group"]) <= set(f_sel.blocks)]
            if side_info and f_side:
                chosen_tags.append(side_info["tag"])
        vrows = []
        for v in variants:
            s = summary[v.name]
            vrows.append(dict(name=v.name, contents=v.describe(profile, nb, has_side), params=v.param_share,
                              id_kept=s["id_kept"], id_kept_se=s["id_kept_se"], leak=s["leak_kept"],
                              drift=s["drift_rel"], variety=s.get("variety_rel"),
                              faces=f"{s['id_faces']}/{s['id_total']}", highlight=(v is res["final"]),
                              status=res["status"].get(v.name)))
        highlight = None
        if res["restricted"]:
            highlight = (_highlight_cols(profile, nb, f_sel.blocks, f_side, has_side), set(f_sel.modules))
        res["diag"] = render_diagnostic(dict(
            title=title, subtitle=subtitle, verdict=verdict, yaml=res["yaml"],
            yaml_title="ai-toolkit config (suggested)" if res["restricted"] else "ai-toolkit config (no restriction)",
            heat=_heat(profile, res["info"], nb, has_side, ctx.present), highlight=highlight, groups=chart_rows,
            chosen_tags=chosen_tags, target=ctx.target, variants=vrows,
            footer=[
                "Inference-time ablation: each variant is the same trained LoRA with some layers removed. It shows "
                "where this LoRA keeps the identity. Retrain with the config above and probe the new LoRA to confirm: "
                "a LoRA restricted at training time relocates what it learns.",
                f"Identity: {scorer.name} cosine similarity to the reference centroid"
                + (", best-matching face per render. " if scorer.name == "ArcFace" else ", whole-image embedding. ")
                + "Leak: the same on control prompts. Drift: 1 - SSIM of control renders against the base model, "
                "same seed. Variety: how different the identity renders are from each other (small colour "
                "thumbnails), relative to the base model; lower means the images look more alike whatever the prompt.",
            ]))
        stem = os.path.splitext(base)[0]
        res["sheets"], res["sheet_files"] = self._contact_sheets(
            profile, variants, res["results"], res["recs"], summary, ctx.prompts, ctx.seeds, scorer, nb, stem, has_side)
        final_info = {"headline": headline, "reasons": res["reasons"], "yaml": res["yaml"]}
        report = core.text_report(profile, base, res["info"], nb, variants, summary, rows, side_info, final_info,
                                  " · ".join(subtitle), show_side=has_side)
        if res["warnings"]:
            report += "\n\nNotes:\n" + "\n".join(f"  ! {w}" for w in res["warnings"])
        res["report"] = report + "\n\nPrompts:\n" + "\n".join(f"  {k}{i + 1}: {p}" for k, i, p in ctx.prompts)
        res["subtitle"] = subtitle

    # ------------------------------------------------------------------ comparison

    def _compare(self, ctx, results, elapsed):
        profile, nb = ctx.profile, ctx.nb
        probes = [core.CheckpointProbe(
            name=r["name"], label=r["label"], short=r["short"], step=r["step"], info=r["info"], nb=nb,
            variants=r["variants"], summary=r["summary"], rows=r["rows"], side_info=r["side_info"], what=r["what"],
            tested_kept=r["tested_kept"], tested_pass=r["tested_pass"], size_frac=r["size_frac"],
            restricted=r["restricted"], reliable=r["reliable"]) for r in results]
        seq = ctx.same_run
        cmp = core.compare_checkpoints(profile, probes, ctx.target, ctx.group_size, ctx.scorer.name, sequence=seq)
        counts = _counts_text(sum(r["rendered"] for r in results), sum(r["from_disk"] for r in results),
                              sum(r["from_mem"] for r in results))
        subtitle = [ctx.settings,
                    f"{len(results)} LoRA files · {counts}{_check_text(ctx)} · {elapsed / 60:.1f} min · "
                    f"{ctx.n_ref} reference images"]
        g = cmp["grids"]
        grids = [
            dict(title="Necessity", sub="identity lost when only this block group is removed",
                 cols=g["necessity"]["cols"], values=g["necessity"]["values"]),
            dict(title="Sufficiency", sub="identity kept with only this block group",
                 cols=g["sufficiency"]["cols"], values=g["sufficiency"]["values"]),
            dict(title="Weight change", sub="share of the LoRA's weight change in each block group (from the file)",
                 cols=g["weights"]["cols"], values=g["weights"]["values"], vmax=g["weights"]["vmax"], files=True),
            dict(title="Layer types and subsets",
                 sub="identity kept by these variants" + (
                     f" ({g['subsets']['omitted']} custom variants are in each checkpoint's own diagnostic)"
                     if g["subsets"].get("omitted") else " (custom ones included)"),
                 cols=g["subsets"]["cols"], values=g["subsets"]["values"]),
        ]
        table = [dict(row, highlight=(i == cmp["pick"])) for i, row in enumerate(cmp["table"])]
        stems = [os.path.splitext(os.path.basename(r["name"]))[0] for r in results]
        title = os.path.commonprefix(stems).rstrip("_- ") if seq else " vs ".join(stems)
        if seq:
            words = dict(head="Checkpoint Comparison", first_col="checkpoint",
                         section1="1  The full LoRA at each checkpoint",
                         section2="2  What carries the identity at each checkpoint", section3="3  Each checkpoint",
                         footer=[
                             "Each LoRA file was probed with the same prompts, seeds and settings; the no-LoRA renders "
                             "are shared. Green bar in the table: the earliest checkpoint within 5% of the best "
                             "likeness. Rising leak or drift, or falling prompt variety, after likeness has levelled "
                             "off are typical signs of overfitting.",
                             "Every checkpoint also has its own full diagnostic and contact sheets."])
        else:
            words = dict(head="LoRA Comparison", first_col="LoRA file", section1="1  The full LoRA of each file",
                         section2="2  What carries the identity in each file", section3="3  Each LoRA file",
                         footer=[
                             "Each LoRA file was probed with the same prompts, seeds and settings; the no-LoRA renders "
                             "are shared. These files are not saves of one training run, so they are kept in the "
                             "order given and not read as a trend. Green bar in the table: the highest likeness.",
                             "Every file also has its own full diagnostic and contact sheets."])
        if g["subsets"].get("omitted"):
            words["footer"].append(f"Your {g['subsets']['omitted']} custom variants are too many for this view; each "
                                   "checkpoint's own diagnostic and report list them.")
        cmp["image"] = render_comparison(dict(
            title=title or ", ".join(r["short"] for r in results), subtitle=subtitle, verdict=cmp["verdict"],
            shorts=cmp["shorts"], labels=cmp["labels"], weak=cmp["weak"], charts=cmp["charts"], grids=grids,
            table=table, target=ctx.target, **words))
        cmp["sheets"] = self._comparison_sheets(ctx, results)
        cmp["text"] = core.comparison_text(cmp, " · ".join(subtitle))
        cmp["subtitle"] = subtitle
        return cmp

    @staticmethod
    def _comparison_sheets(ctx, results):
        """Per seed: the no-LoRA render and every checkpoint's full LoRA, side by side."""
        scorer = ctx.scorer
        b = results[0]["summary"].get("base") or {}
        lo = b.get("id_sim") if core.finite(b.get("id_sim")) else 0.0
        fulls = [r["summary"]["full"]["id_sim"] for r in results if core.finite(r["summary"]["full"].get("id_sim"))]
        hi = max(fulls) if fulls else 1.0
        span = (hi - lo) if hi - lo > 1e-6 else 1.0
        rows = [("base", "no LoRA" + (f" · likeness {lo:.2f}" if core.finite(b.get("id_sim")) else ""))]
        for r in results:
            sim = r["summary"]["full"].get("id_sim")
            stem = os.path.splitext(os.path.basename(r["name"]))[0]
            rows.append((r["label"], "full LoRA" + (f" · likeness {sim:.2f}" if core.finite(sim) else "")
                         + ("" if r["label"] == stem else f" · {stem}")))
        sources = [("base", results[0])] + [("full", r) for r in results]
        columns = [(f"{k}{i + 1}", k) for k, i, _ in ctx.prompts]
        legend = [f"{k}{i + 1}: {p}" for k, i, p in ctx.prompts]
        legend.append(f"Scores: {scorer.name} similarity to the references. Identity columns are green when close to "
                      "the best full LoRA's likeness; control columns turn red when a render resembles the character.")
        sheets = []
        for n, seed in enumerate(ctx.seeds, start=1):
            cells = []
            for var, r in sources:
                by = {(x["variant"], x["kind"], x["prompt"], x["seed"]): x for x in r["results"]}
                sims = {(x["variant"], x["kind"], x["prompt"], x["seed"]): x["sim"] for x in r["recs"]}
                line = []
                for k, i, _ in ctx.prompts:
                    hit = by.get((var, k, i, seed))
                    sim = sims.get((var, k, i, seed), NAN)
                    if k == "id":
                        t = (sim - lo) / span if core.finite(sim) else -1
                        col = GOOD if t >= 0.8 else WARN if t >= 0.4 else BAD
                    else:
                        col = LEAK if core.finite(sim) and sim >= lo + 0.5 * span else MUTED
                    line.append((_unjpeg(hit["entry"]["thumb"]) if hit else None, sim, col))
                cells.append(line)
            title = (("Checkpoints" if ctx.same_run else "LoRA files") + f" side by side · seed {seed}"
                     + (f" ({n} of {len(ctx.seeds)})" if len(ctx.seeds) > 1 else ""))
            sheets.append(render_contact_sheet(rows, columns, cells, title=title, legend=legend))
        return sheets

    # ------------------------------------------------------------------ pieces

    @staticmethod
    def _verdict(profile, summary, rows, side_info, final, f_sel, f_side, nb, restricted, target, scorer, ref_self,
                 warnings, reliable, has_side, present=None):
        out = []
        b, f = summary.get("base"), summary.get("full")
        if not reliable:
            out.append(("bad", f"The full LoRA barely moves {scorer.name} similarity "
                               f"({core.fmt(b['id_sim'] if b else NAN)} -> {core.fmt(f['id_sim'] if f else NAN)}). "
                               "Check the trigger word, strength and references; the rest of this report is unreliable."))
        else:
            out.append(("good", f"The full LoRA lifts {scorer.name} similarity from {b['id_sim']:.2f} to "
                                f"{f['id_sim']:.2f} (references agree with each other at {ref_self:.2f})."))

        # the suggestion goes second, it is what the reader came for
        what = core.describe_selection(profile, nb, f_sel.blocks, f_side, f_sel.modules, has_side)
        if not reliable:
            out.append(("bad", "No suggestion: the identity signal is too weak to rank blocks or layers. "
                               "The config shows the default (all layers)."))
        elif not restricted:
            out.append(("warn", "Suggestion: keep training all layers. Identity is spread across blocks and layer "
                                "types in this LoRA, so restricting layers would cost identity; use differential "
                                "output preservation or masked loss to limit bleed instead."))
        elif final is not None:
            s = summary[final.name]
            ok = core.finite(s["id_kept"]) and s["id_kept"] >= target
            extra = []
            if core.finite(s["leak_kept"]):
                extra.append(f"{100 * s['leak_kept']:.0f}% of the leak")
            if core.finite(s["drift_rel"]):
                extra.append(f"{100 * s['drift_rel']:.0f}% of the drift")
            frac = core.lora_param_fraction(profile, f_sel.blocks, f_sel.modules, f_side, nb, present)
            out.append(("good" if ok else "warn",
                        f"Suggestion: train {what}. "
                        f"In its test render it kept {core.fmt(100 * s['id_kept'], '.0f')}% of the identity"
                        + (" with " + " and ".join(extra) if extra else "")
                        + f" of the full LoRA, at {100 * frac:.0f}% of its size."
                        + ("" if ok else " It missed the target; consider a wider subset.")))
        else:
            out.append(("info", f"Suggestion (untested): train {what}. Set test_rounds above 0 to check it."))
        headline = out[-1][1]

        def loss(x):
            return "all of it" if x >= 0.995 else f"{100 * max(x, 0.0):.0f}%"

        if rows and reliable:
            rel = [r for r in rows if core.group_effect(r) >= 0.05]
            if not rel:
                out.append(("info", "No single block group moves identity by 5% or more: it is stored redundantly "
                                    "across blocks."))
            elif len(rel) <= len(rows) // 2:
                out.append(("info", f"Identity sits in {len(rel)} of {len(rows)} block groups: "
                                    f"{', '.join(r['tag'] for r in rel)}."))
            else:
                out.append(("info", f"Identity is spread over {len(rel)} of {len(rows)} block groups."))
            suff = [r for r in rows if core.finite(r["sufficiency"])]
            nec = [r for r in rows if core.finite(r["necessity"])]
            if suff:
                top = max(suff, key=lambda r: r["sufficiency"])
                out.append(("info", f"Best group on its own: {top['tag']} keeps {100 * top['sufficiency']:.0f}% "
                                    "of the identity."))
            if nec:
                worst = max(nec, key=lambda r: r["necessity"])
                rest = max([r["necessity"] for r in nec if r is not worst] or [0.0])
                out.append(("info", f"Removing {worst['tag']} hurts most (loses {loss(worst['necessity'])}); "
                                    f"any other single group loses at most {loss(rest)}."))
        if side_info and reliable:
            parts = []
            if core.finite(side_info["sufficiency"]):
                parts.append(f"alone they keep {100 * side_info['sufficiency']:.0f}% of the identity")
            if core.finite(side_info["necessity"]):
                parts.append(f"removing them loses {loss(side_info['necessity'])}")
            if parts:
                out.append(("info", f"Text side ({profile.side.label} layers, which only see the prompt): "
                                    + "; ".join(parts) + "."))
        parts = []
        for label, _ in profile.named_sets():
            s = summary.get(f"keep_{label}")
            if s and core.finite(s["id_kept"]):
                parts.append(f"{core.describe_modules(profile, profile.sets[label])} alone {100 * s['id_kept']:.0f}%")
        if parts:
            out.append(("info", "By layer type: " + ", ".join(parts) + "."))
        fv = (f or {}).get("variety_rel", NAN)
        if reliable and core.finite(fv):
            out.append(("info", f"Prompt variety of the full LoRA: {100 * fv:.0f}% of the base model's (lower means "
                                "its images look more alike whatever the prompt)."))
        for w in warnings:
            out.append(("warn", w))
        out.append(("info", "Confirm by retraining with this config and probing the new LoRA."))
        return out, headline

    @staticmethod
    def _contact_sheets(profile, variants, results, recs, summary, prompts, seeds, scorer, nb, stem, has_side):
        """One contact sheet per seed, all with the same layout."""
        b, f = summary.get("base"), summary.get("full")
        lo = b["id_sim"] if b and core.finite(b["id_sim"]) else 0.0
        hi = f["id_sim"] if f and core.finite(f["id_sim"]) else 1.0
        span = (hi - lo) if hi - lo > 1e-6 else 1.0
        by = {(r["variant"], r["kind"], r["prompt"], r["seed"]): r for r in results}
        sims = {(r["variant"], r["kind"], r["prompt"], r["seed"]): r["sim"] for r in recs}
        columns = [(f"{k}{i + 1}", k) for k, i, _ in prompts]
        rows = {}
        for v in variants:
            s = summary[v.name]
            sub = v.describe(profile, nb, has_side) + (f" · id kept {round(100 * s['id_kept'])}%" if core.finite(s["id_kept"]) else "")
            rows[v.name] = (v.name, sub)
        legend = [f"{k}{i + 1}: {p}" for k, i, p in prompts]
        legend.append(f"Scores: {scorer.name} similarity to the references; identity columns are green when close "
                      "to the full LoRA, control columns turn red when a render resembles the character. "
                      "'id kept' on the left is averaged over all seeds.")
        names = [v.name for v in variants]
        if len(names) <= SHEET_ROWS:
            pages = [names]
        else:  # long sweeps: several parts, each starting with base and full for reference
            ref = [n for n in ("base", "full") if n in rows]
            rest = [n for n in names if n not in ref]
            per = SHEET_ROWS - len(ref)
            pages = [ref + rest[i:i + per] for i in range(0, len(rest), per)]
        sheets, files = [], []
        for n, seed in enumerate(seeds, start=1):
            for part, page in enumerate(pages, start=1):
                cells = []
                for name in page:
                    line = []
                    for k, i, _ in prompts:
                        r = by.get((name, k, i, seed))
                        sim = sims.get((name, k, i, seed), NAN)
                        if k == "id":
                            t = (sim - lo) / span if core.finite(sim) else -1
                            col = GOOD if t >= 0.8 else WARN if t >= 0.4 else BAD
                        else:
                            col = LEAK if core.finite(sim) and sim >= lo + 0.5 * span else MUTED
                        line.append((_unjpeg(r["entry"]["thumb"]) if r else None, sim, col))
                    cells.append(line)
                title = (f"Contact sheet · {stem} · seed {seed}" + (f" ({n} of {len(seeds)})" if len(seeds) > 1 else "")
                         + (f" · part {part} of {len(pages)}" if len(pages) > 1 else ""))
                sheets.append(render_contact_sheet([rows[x] for x in page], columns, cells, title=title, legend=legend))
                files.append(f"contact_sheet_seed{seed}.png" if len(pages) == 1 else f"contact_sheet_seed{seed}_part{part}.png")
        return sheets, files

    # ------------------------------------------------------------------ saving

    @staticmethod
    def _clean(x):
        if isinstance(x, float) and not math.isfinite(x):
            return None
        if isinstance(x, dict):
            return {k: LoRAProber._clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [LoRAProber._clean(v) for v in x]
        return x

    def _results_json(self, ctx, res):
        profile = ctx.profile
        return self._clean({
            "model": profile.label,
            "lora": os.path.basename(res["name"]),
            "step": res["step"],
            "label": res["label"],
            "settings": res["subtitle"],
            "prompts": [{"kind": k, "index": i, "text": p} for k, i, p in ctx.prompts],
            "variants": [{"name": v.name, "kind": v.kind, "blocks": v.blocks, "layers": v.modules,
                          "text_fusion": v.side if profile.side is not None else None,
                          "selection": v.spec or None,
                          "contents": v.describe(profile, ctx.nb, res["has_side"]),
                          "param_share": v.param_share} for v in res["variants"]],
            "summary": res["summary"],
            "renders": [dict(rec, image=res.get("images", {}).get((rec["variant"], rec["kind"], rec["prompt"],
                                                                   rec["seed"]))) for rec in res["recs"]],
        })

    @staticmethod
    def _link_renders(res, run_dir):
        """renders/<variant>/<id1|ctrl1>_seed<N>.png: hard links to the cached images (no extra disk space)."""
        images, copied = {}, 0
        for r in res["results"]:
            src = r["entry"].get("path")
            if not src or not os.path.exists(src):
                continue
            rel = f"renders/{r['variant']}/{r['kind']}{r['prompt'] + 1}_seed{r['seed']}.png"
            try:
                if rc.link_or_copy(src, os.path.join(run_dir, *rel.split("/"))) == "copy":
                    copied += 1
            except OSError as e:
                log.warning("LoRA Prober: could not add %s to the run folder (%s)", rel, e)
                continue
            images[(r["variant"], r["kind"], r["prompt"], r["seed"])] = rel
        if copied:
            log.info("LoRA Prober: %d renders were copied into the run folder (hard links not supported here)", copied)
        res["images"] = images

    def _write_checkpoint_files(self, ctx, res, run_dir):
        os.makedirs(run_dir, exist_ok=True)
        if ctx.disk is not None:
            self._link_renders(res, run_dir)
        res["diag"].save(os.path.join(run_dir, "diagnostic.png"))
        sheet_names = res["sheet_files"]
        for sheet, name in zip(res["sheets"], sheet_names):
            sheet.save(os.path.join(run_dir, name))
        with open(os.path.join(run_dir, "report.txt"), "w", encoding="utf-8") as fh:
            fh.write(res["report"])
        with open(os.path.join(run_dir, "ai_toolkit_network.yaml"), "w", encoding="utf-8") as fh:
            fh.write(res["yaml"] + "\n")
        with open(os.path.join(run_dir, "results.json"), "w", encoding="utf-8") as fh:
            json.dump(self._results_json(ctx, res), fh, indent=1)
        return ["diagnostic.png"] + sheet_names

    def _save_single(self, ctx, res):
        out_root = folder_paths.get_output_directory()
        stem = _safe_name(os.path.splitext(os.path.basename(res["name"]))[0])
        run_dir = os.path.join(out_root, "lora_prober", f"{stem}_{time.strftime('%Y%m%d-%H%M%S')}")
        files = self._write_checkpoint_files(ctx, res, run_dir)
        sub = os.path.relpath(run_dir, out_root).replace(os.sep, "/")
        log.info("LoRA Prober results saved to %s", run_dir)
        return [{"filename": f, "subfolder": sub, "type": "output"} for f in files]

    def _save_multi(self, ctx, results, cmp, report):
        out_root = folder_paths.get_output_directory()
        primary = next(r for r in results if r["primary"])
        stem = _safe_name(os.path.splitext(os.path.basename(primary["name"]))[0])
        run_dir = os.path.join(out_root, "lora_prober", f"{stem}_compare_{time.strftime('%Y%m%d-%H%M%S')}")
        os.makedirs(run_dir, exist_ok=True)
        cmp["image"].save(os.path.join(run_dir, "comparison.png"))
        sheet_names = [f"comparison_sheet_seed{seed}.png" for seed in ctx.seeds]
        for sheet, name in zip(cmp["sheets"], sheet_names):
            sheet.save(os.path.join(run_dir, name))
        with open(os.path.join(run_dir, "report.txt"), "w", encoding="utf-8") as fh:
            fh.write(report)
        data = {
            "settings": cmp["subtitle"],
            "checkpoints": [{"lora": os.path.basename(r["name"]), "label": r["label"], "step": r["step"],
                             "folder": None} for r in results],
            "verdict": [{"kind": k, "text": t} for k, t in cmp["verdict"]],
            "charts": cmp["charts"], "grids": cmp["grids"], "table": cmp["table"],
            "earliest_within_5pct_of_best": cmp["labels"][cmp["pick"]] if cmp["pick"] is not None else None,
        }
        sub = os.path.relpath(run_dir, out_root).replace(os.sep, "/")
        ui = [{"filename": "comparison.png", "subfolder": sub, "type": "output"}]
        ui += [{"filename": n, "subfolder": sub, "type": "output"} for n in sheet_names]
        used = set()
        for i, r in enumerate(results):
            folder = _safe_name(f"{i + 1}_{os.path.splitext(os.path.basename(r['name']))[0]}")
            while folder in used:
                folder += "_"
            used.add(folder)
            data["checkpoints"][i]["folder"] = folder
            files = self._write_checkpoint_files(ctx, r, os.path.join(run_dir, folder))
            ui += [{"filename": f, "subfolder": f"{sub}/{folder}", "type": "output"} for f in files]
        with open(os.path.join(run_dir, "comparison.json"), "w", encoding="utf-8") as fh:
            json.dump(self._clean(data), fh, indent=1)
        log.info("LoRA Prober comparison saved to %s", run_dir)
        return ui


class LoRAProberWeightReport:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "lora_name": (folder_paths.get_filename_list("loras"),),
            "group_size": ("INT", {"default": 4, "min": 1, "max": 16}),
        }, "optional": {
            "model_family": (core.FAMILY_CHOICES, {"default": "auto",
                                                   "tooltip": "auto detects the model from the LoRA's layer names."}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("report_image", "report")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = f"Instant view of where a LoRA changed weights, per block and layer type ({MODELS_TEXT}). No rendering."

    def run(self, lora_name, group_size, model_family="auto"):
        _path, _mtime, sd, _meta = _load_lora(lora_name)
        profile, _warn = _resolve_profile(model_family, None, sd.keys())
        info = core.analyze_lora(profile, sd)
        if not info.sq:
            raise ValueError(f"{lora_name} has no {profile.label} block layers.")
        nb = max(profile.default_blocks, (max(info.blocks) + 1) if info.blocks else 0)
        total = info.total_energy()
        energy = info.block_energy(nb)
        heat = _heat(profile, info, nb, info.has_side)
        module_share = [(m, info.module_energy(m) / total) for m in heat["modules"]]
        group_share = []
        for start in range(0, nb, group_size):
            grp = list(range(start, min(start + group_size, nb)))
            group_share.append((core.group_tag(grp), sum(energy[b] for b in grp) / total))
        if profile.side is not None and info.has_side:
            group_share.append((profile.side.label, sum(info.side_energy()) / total))
        n_layers = len(info.sq)
        sub = [f"{profile.label} · {'/'.join(sorted(info.kinds))} · rank {core.compact_ranges(info.ranks) or '-'} · "
               f"{n_layers} layers in {len(info.blocks)} blocks"
               + (f" + {profile.side.label}" if profile.side is not None and info.has_side else "")]
        if info.other:
            sub.append(f"{len(info.other)} tensors outside the transformer blocks are not shown")
        sub += info.notes
        img = render_weight_report(os.path.basename(lora_name), sub, heat, module_share, group_share)
        lines = [f"{os.path.basename(lora_name)}: " + sub[0], "share of the weight change by block:"]
        lines.append("  " + "  ".join(f"{b}:{100 * energy[b] / total:.1f}%" for b in range(nb)))
        if profile.side is not None and info.has_side:
            lines.append(f"  {profile.side.label}: " + "  ".join(
                f"{lab}:{100 * e / total:.1f}%" for lab, e in zip(profile.side.labels, info.side_energy())))
        lines.append("by layer type: " + "  ".join(f"{m} {100 * s:.0f}%" for m, s in module_share))
        lines.append("by group: " + "  ".join(f"{t} {100 * s:.0f}%" for t, s in group_share))
        return {"ui": {"images": [_save_temp_preview(img, "lora_weight_report")]},
                "result": (_pil_to_image(img), "\n".join(lines))}


class LoRAProberBlockSelectLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "lora_name": (folder_paths.get_filename_list("loras"),),
            "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
            "blocks": ("STRING", {"default": "all", "tooltip": "e.g. all, 8-23, 0-3,12,16-19; add 'txt' for the "
                                                              "text-side blocks (Krea 2 text fusion, MiniMax H3 token "
                                                              "refiner; all includes them)."}),
            "layers": ("STRING", {"default": "all", "tooltip": "all, attn, mlp, kv (not H3), or a list of layer names "
                                                              "(Qwen-Image 2.1: q,k,v,o,mlp_in,mlp_out; Krea 2: "
                                                              "q,k,v,gate,o,mlp_gate,mlp_up,mlp_down; MiniMax H3: "
                                                              "qkv,o,gate_compress,mlp_in,mlp_out,adaln)."}),
        }, "optional": {
            "model_family": (core.FAMILY_CHOICES, {"default": "auto"}),
            "selection": ("STRING", {"default": "", "tooltip": SELECTION_TIP + " When set, it replaces blocks and "
                                                                "layers. Same text as a LoRA Prober custom variant."}),
        }}

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "applied")
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = f"Applies a LoRA to only the chosen blocks and layer types ({MODELS_TEXT})."

    def load(self, model, lora_name, strength_model, blocks, layers, model_family="auto", selection=""):
        _path, _mtime, sd, meta = _load_lora(lora_name)
        profile, _warn = _resolve_profile(model_family, model, sd.keys())
        nb = _num_blocks(model, profile)
        if (selection or "").strip():
            v = core.custom_variant(profile, "select", selection, nb)
        else:
            sel_blocks, sel_side = core.parse_selection(profile, blocks, nb)
            v = core.Variant("select", "custom", sel_blocks, core.parse_modules(profile, layers), side=sel_side)
        n_file = len(sd)
        total = sum(core.numel(t) for t in sd.values()) or 1
        sd, notes = _fit_lora(model, profile, sd)
        core.assign_keys(profile, sd, [v])
        share = sum(core.numel(sd[k]) for k in v.keys) / total
        has_side = core.analyze_lora(profile, sd).has_side
        applied = (f"{profile.label}: {len(v.keys)} of {n_file} tensors: {v.describe(profile, nb, has_side)} "
                   f"({100 * share:.0f}% of the LoRA)" + "".join(" " + n for n in notes))
        if not v.keys or strength_model == 0:
            return (model, applied)
        sub = {k: sd[k] for k in v.keys}
        patched = comfy.sd.load_lora_for_models(model, None, sub, strength_model, 0, lora_metadata=meta)[0]
        return (patched, applied)


NODE_CLASS_MAPPINGS = {
    "LoRAProber": LoRAProber,
    "LoRAProberWeightReport": LoRAProberWeightReport,
    "LoRAProberBlockSelectLoader": LoRAProberBlockSelectLoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoRAProber": "LoRA Prober",
    "LoRAProberWeightReport": "LoRA Prober Weight Report",
    "LoRAProberBlockSelectLoader": "LoRA Prober Loader (Block Select)",
}
