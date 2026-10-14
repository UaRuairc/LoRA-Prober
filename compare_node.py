"""LoRA Prober Compare: the same prompts and seeds rendered with any number of LoRA files - checkpoints of one training run
or files from different runs - scored for likeness and laid side by side.

MiniMax H3 renders full videos with sound; each prompt and seed gets one comparison video in which every clip plays
at once (no sound), then each clip plays in turn with its own sound. Image models get comparison sheets and a
flip-book video that steps through the files.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
import time
import types

import numpy as np
import torch
from PIL import Image, ImageDraw

import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.sd
import comfy.utils
import folder_paths
import latent_preview

from . import media
from . import probe_core as core
from . import render_cache as rc
from .identity import drift, gray_small, layout_thumb, reference_centroid, thumbnail
from .probe_nodes import (CATEGORY, CLIP_TIP, METRICS, MODELS_TEXT, VAE_TIP, _Progress,
                          _apply_speed_lora, _cache_get, _cache_put, _counts_text, _empty_latent, _encode, _fit_lora,
                          _image_batch, _init_label, _jpeg, _load_lora, _lora_header, _make_scorer, _negative,
                          _num_blocks, _pil_to_image, _prompt_lines, _resolve_profile, _safe_name, _unjpeg)
from .render import (BAD, BG, GOOD, LEAK, MUTED, PANEL, RUN_COLORS, TEXT, WARN, ellipsize, font, render_contact_sheet,
                     render_lora_compare)

log = logging.getLogger("lora_prober")
NAN = float("nan")
SCORED_FRAMES = 4  # frames of each video scored for likeness (evenly spaced, the first included)

DEFAULT_PROMPTS = """[trigger] looks into the camera and smiles, close-up, soft window light
[trigger] walks down a city street at night, neon reflections, medium shot
[trigger] sits at a cafe table talking with a friend, warm afternoon light"""
DEFAULT_CTRL = "a person walks down a city street at night, neon reflections, medium shot"
LORAS_TIP = ("One entry per line; each line can match several files:\n"
             "  my_char/my_char_000002000.safetensors   one file (the extension is optional)\n"
             "  my_char_v2                              every file whose path contains this (all saves of a run)\n"
             "  runs/*_000003000                        a pattern with * or ?\n"
             "  my_char.safetensors | 0.8               a different strength\n"
             "  my_char.safetensors | 8-23:mlp          only some layers (same text as a LoRA Prober custom variant)\n"
             "Saves of one run are put in step order; runs keep the order of the lines. # starts a comment. "
             f"Up to {core.MAX_COMPARE_FILES} files.")


def _video_type():
    try:
        from comfy_api.latest import InputImpl

        return InputImpl.VideoFromFile
    except Exception:
        return None


def _frame_indices(n, k=SCORED_FRAMES):
    if n <= 1:
        return [0]
    return sorted({int(round(i * (n - 1) / max(1, k - 1))) for i in range(min(k, n))})


def _video_similarity(scorer, per_frame, centroid):
    sims = []
    faces = 0
    for e in per_frame or []:
        s, nf = scorer.similarity(e, centroid)
        if core.finite(s):
            sims.append(s)
            faces += 1
    return (sum(sims) / len(sims) if sims else NAN), faces


class LoRAProberCompare:
    @classmethod
    def INPUT_TYPES(cls):
        loras = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model": ("MODEL", {"tooltip": f"The base model WITHOUT a LoRA ({MODELS_TEXT})."}),
                "clip": ("CLIP", {"tooltip": CLIP_TIP}),
                "vae": ("VAE", {"tooltip": VAE_TIP}),
                "loras": ("STRING", {"multiline": True, "default": "", "tooltip": LORAS_TIP}),
                "trigger_word": ("STRING", {"default": "", "tooltip": "Replaces [trigger] in the prompts."}),
                "prompts": ("STRING", {"multiline": True, "default": DEFAULT_PROMPTS,
                                       "tooltip": "One prompt per line, each with the trigger; every LoRA renders all "
                                                  "of them."}),
                "base_seed": ("INT", {"default": 1, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "seeds_per_prompt": ("INT", {"default": 1, "min": 1, "max": 8}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100,
                                  "tooltip": "What you normally use: Qwen-Image 2.1 about 20-25, Krea 2 Turbo 8, "
                                             "MiniMax H3 20 with res_multistep (8 with FastH3 or the turbo LoRA)."}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "width": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 32}),
                "height": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 32}),
                "length": ("INT", {"default": 107, "min": 5, "max": 3600, "step": 17,
                                   "tooltip": "MiniMax H3 only: frames per video at 24 fps (107 = about 4.5 s, as "
                                              "ai-toolkit's samples). Image models ignore it."}),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05,
                                            "tooltip": "For every line without its own strength."}),
                "include_base": ("BOOLEAN", {"default": True,
                                             "tooltip": "Also render the base model without a LoRA, as the reference."}),
            },
            "optional": {
                "reference_images": ("IMAGE", {"tooltip": "Photos of the character, for likeness scores. Without "
                                                          "them you get the renders and videos only."}),
                "clip_vision": ("CLIP_VISION", {"tooltip": "For stylised characters, or when insightface is missing."}),
                "identity_metric": (METRICS, {"default": "auto"}),
                "control_prompts": ("STRING", {"multiline": True, "default": DEFAULT_CTRL,
                                               "tooltip": "Prompts WITHOUT the trigger, to see leak and drift. Each "
                                                          "one costs a render per LoRA; leave empty to skip."}),
                "negative_prompt": ("STRING", {"default": "", "tooltip": "Only matters when cfg > 1."}),
                "audio_vae": ("VAE", {"tooltip": "MiniMax H3: the audio VAE (minimax_h3_audio_vae), so the clips "
                                                 "and comparison videos have sound."}),
                "model_family": (core.FAMILY_CHOICES, {"default": "auto"}),
                "speed_lora": (["none"] + loras, {"default": "none",
                                                  "tooltip": "A LoRA applied to the base model for every render (e.g. "
                                                             "a turbo LoRA), as in LoRA Prober."}),
                "speed_lora_strength": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05}),
                "disk_cache": ("BOOLEAN", {"default": True,
                                           "tooltip": "Reuse renders from output/lora_prober/cache, shared with "
                                                      "LoRA Prober, and keep new ones there."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING", "VIDEO")
    RETURN_NAMES = ("comparison", "sheets", "report", "video_files", "video")
    OUTPUT_TOOLTIPS = (
        "Likeness, leak, drift and variety of every LoRA, with the verdict.",
        "One sheet per seed: every LoRA (rows) on every prompt (columns); for videos, their middle frames.",
        "Text report.",
        "Paths of the comparison videos, one per line.",
        "The first comparison video (MiniMax H3) or the flip-book (image models), e.g. for Save Video.",
    )
    FUNCTION = "run"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = ("Renders the same prompts and seeds with any number of LoRA files (checkpoints of one run or "
                   "different runs), scores likeness against reference photos, and lays them side by side. MiniMax "
                   "H3: comparison videos where every clip plays at once, then each in turn with its sound. "
                   f"Supports {MODELS_TEXT}.")

    # ------------------------------------------------------------------ main

    def run(self, model, clip, vae, loras, trigger_word, prompts, base_seed, seeds_per_prompt, steps, cfg,
            sampler_name, scheduler, width, height, length, lora_strength, include_base, reference_images=None,
            clip_vision=None, identity_metric="auto", control_prompts=DEFAULT_CTRL, negative_prompt="",
            audio_vae=None, model_family="auto", speed_lora="none", speed_lora_strength=1.0, disk_cache=True,
            unique_id=None):
        t0 = time.time()
        items = core.parse_lora_list(loras, folder_paths.get_filename_list("loras"))
        for it in items:
            path = folder_paths.get_full_path_or_raise("loras", it["name"])
            keys, meta = _lora_header(path)
            it.update(path=path, keys=keys, step=core.checkpoint_step(it["name"], meta),
                      run=core.checkpoint_run(it["name"], meta))
        profile, warnings = _resolve_profile(model_family, model,
                                             next((it["keys"] for it in items if it["exact"]), items[0]["keys"]))
        skipped = []
        for it in items:
            share, best = core.recognised_share(profile, it["keys"]), core.detect_profile(it["keys"])
            foreign = share == 0 or (best is not None and best is not profile
                                     and core.recognised_share(best, it["keys"]) > share + 0.2)
            if foreign:
                if it["exact"]:
                    raise ValueError(f"{it['name']} is not a {profile.label} LoRA"
                                     + (f" (it looks like a {best.label} LoRA)" if best is not None else "") + ".")
                skipped.append(it["name"])  # a pattern also matched another model's LoRA: leave it out
        if skipped:
            items = [it for it in items if it["name"] not in skipped]
            if not items:
                raise ValueError(f"None of the matching files is a {profile.label} LoRA: {', '.join(skipped[:6])}.")
            warnings.append(f"Left out {len(skipped)} matching file{'s' if len(skipped) > 1 else ''} that "
                            f"{'are not' if len(skipped) > 1 else 'is not a'} {profile.label} LoRA"
                            f"{'s' if len(skipped) > 1 else ''}: {', '.join(os.path.basename(n) for n in skipped[:6])}"
                            + (" ..." if len(skipped) > 6 else "") + ".")
        items = core.order_runs(items)
        core.label_compare_items(items, lora_strength)
        nb = _num_blocks(model, profile)
        for it in items:  # a bad selection fails now, not after the other files have rendered
            if it["spec"]:
                try:
                    core.parse_variant_selection(profile, it["spec"], nb)
                except ValueError as e:
                    raise ValueError(f"{it['name']} | {it['spec']}: {e}") from None
        video = profile.video
        frames = media.h3_frame_count(length) if video else 1
        if video and (width % 32 or height % 32):
            raise ValueError(f"{profile.label} needs width and height in multiples of 32 (got {width}x{height}).")
        if video and audio_vae is None:
            warnings.append("No audio VAE connected: the videos are silent (connect minimax_h3_audio_vae to "
                            "audio_vae for sound).")

        trigger = trigger_word.strip()
        id_prompts = _prompt_lines(prompts, trigger)
        ctrl_prompts = _prompt_lines(control_prompts, trigger)
        if not id_prompts:
            raise ValueError("Add at least one prompt.")
        if trigger and any(trigger.lower() not in p.lower() for p in id_prompts):
            warnings.append("Some prompts do not contain the trigger word.")
        plist = [("id", i, p) for i, p in enumerate(id_prompts)] + [("ctrl", i, p) for i, p in enumerate(ctrl_prompts)]
        seeds = [base_seed + j for j in range(seeds_per_prompt)]
        scorer, centroid, n_ref, ref_self = None, None, 0, NAN
        if reference_images is not None:
            scorer = _make_scorer(identity_metric, clip_vision)
            centroid, n_ref, ref_self = reference_centroid(scorer, reference_images)
            if centroid is None:
                raise ValueError(f"{scorer.name} found nothing usable in the reference images (no face detected). "
                                 "Use clearer face photos, or switch to CLIP Vision.")

        model_fp, vae_fp = rc.model_fingerprint(model), rc.vae_fingerprint(vae)
        audio_fp = rc.vae_fingerprint(audio_vae) if (video and audio_vae is not None) else ""
        model, speed_id, speed_label = _apply_speed_lora(model, speed_lora, speed_lora_strength, warnings)
        conds = {p: _encode(profile, clip, p) for _, _, p in plist}
        neg = _negative(profile, clip, negative_prompt, next(iter(conds.values())))
        uses_neg = abs(cfg - 1.0) > 1e-9 or bool(model.model_options.get("disable_cfg1_optimization"))
        out_root = folder_paths.get_output_directory()
        first = _safe_name(os.path.splitext(os.path.basename(items[0]["name"]))[0])
        run_dir = os.path.join(out_root, "lora_prober", f"compare_{first}_{time.strftime('%Y%m%d-%H%M%S')}")
        os.makedirs(run_dir, exist_ok=True)
        cells = ([dict(name="base", item=None, label="base (no LoRA)", short="base", run=None)] if include_base else [])
        for i, it in enumerate(items):
            it["key"] = f"lora{i + 1}"
            cells.append(dict(name=it["key"], item=it, label=it["label"], short=it["short"], run=it["run"]))
        ctx = types.SimpleNamespace(
            model=model, vae=vae, audio_vae=audio_vae, profile=profile, nb=nb, video=video, frames=frames,
            prompts=plist, seeds=seeds, scorer=scorer, centroid=centroid, conds=conds, neg=neg,
            latent=_empty_latent(model, width, height, profile, frames), steps=steps, cfg=cfg,
            sampler_name=sampler_name, scheduler=scheduler, width=width, height=height, strength=lora_strength,
            model_fp=model_fp, vae_fp=vae_fp, audio_fp=audio_fp, speed_id=speed_id, speed_label=speed_label,
            cond_fp={p: rc.cond_fingerprint(c) for p, c in conds.items()},
            neg_fp=rc.cond_fingerprint(neg) if uses_neg else None,
            model_label=_init_label(model) or type(model.model).__name__,
            vae_label=_init_label(getattr(vae, "patcher", None)) or type(vae).__name__,
            disk=rc.DiskCache(os.path.join(out_root, "lora_prober", "cache")) if disk_cache else None,
            run_dir=run_dir, warnings=warnings, rendered=0, from_disk=0, from_mem=0,
            previewer=latent_preview.get_previewer(model.load_device, model.model.latent_format),
            progress=_Progress(steps, unique_id),
        )
        ctx.progress.add(len(cells) * len(plist) * len(seeds))
        log.info("LoRA compare (%s): %d LoRA files%s, %d renders", profile.label, len(items),
                 " + base" if include_base else "", len(cells) * len(plist) * len(seeds))

        results = []
        for cell in cells:
            results += self._render_cell(ctx, cell)
        ctx.progress.finish()

        # ---- scores
        recs, base_gray = [], {}
        for r in results:
            if r["cell"]["name"] == "base" and r["kind"] == "ctrl":
                base_gray[(r["prompt"], r["seed"])] = r["entry"].get("gray")
        for r in results:
            e = r["entry"]
            sim, nf = NAN, 0
            if scorer is not None:
                if e.get("video"):
                    sim, nf = _video_similarity(scorer, rc.DiskCache.frame_embeds_for(e, scorer.key), centroid)
                else:
                    sim, nf = scorer.similarity(rc.DiskCache.embeds_for(e, scorer.key), centroid)
            r["sim"] = sim
            d = NAN
            if r["kind"] == "ctrl" and include_base:
                if r["cell"]["name"] == "base":
                    d = 0.0
                else:
                    g0, g1 = base_gray.get((r["prompt"], r["seed"])), e.get("gray")
                    if g0 is not None and g1 is not None:
                        d = drift(g1, g0)
            recs.append(dict(variant=r["cell"]["name"], kind=r["kind"], prompt=r["prompt"], seed=r["seed"], sim=sim,
                             n_faces=nf, drift=d))
        names = [c["name"] for c in cells]
        summary = core.summarize(recs, names, base="base", full=None)
        core.add_variety(summary, [dict(variant=r["cell"]["name"], kind=r["kind"], prompt=r["prompt"], seed=r["seed"],
                                        lay=r["entry"].get("lay")) for r in results])

        settings = (f"{profile.label} · {len(id_prompts)} prompts" + (f" + {len(ctrl_prompts)} control" if ctrl_prompts
                                                                       else "")
                    + f" x {len(seeds)} seed{'s' if len(seeds) > 1 else ''} · {width}x{height}"
                    + (f" · {frames} frames ({frames / media.H3_FPS:.1f} s)" if video else "")
                    + f" · {steps} steps · {sampler_name}/{scheduler} · cfg {cfg:g} · strength {lora_strength:g}"
                    + (f" · speed LoRA {speed_label}" if speed_label else "")
                    + (f" · {scorer.name}" if scorer is not None else " · no likeness scores"))
        counts = _counts_text(ctx.rendered, ctx.from_disk, ctx.from_mem)
        runs = []
        for it in items:
            if it["run"] not in runs:
                runs.append(it["run"])
        subtitle = [settings, f"{len(items)} LoRA file{'s' if len(items) > 1 else ''} from {len(runs)} run"
                    f"{'s' if len(runs) > 1 else ''} · {counts} · {(time.time() - t0) / 60:.1f} min"
                    + (f" · {n_ref} reference images (self-similarity {ref_self:.2f})" if scorer is not None else "")]
        verdict = core.compare_verdict(items, summary, scorer is not None, bool(ctrl_prompts),
                                       scorer.name if scorer else "")
        noted = {}  # the same note for several files (e.g. adaln on a pruned model) is listed once
        for c in cells:
            for n in c.get("notes", []):
                noted.setdefault(n, []).append(c["label"])
        for n, labels in noted.items():
            verdict.append(("warn", (f"{labels[0]}: " if len(labels) == 1 else f"{len(labels)} files: ") + n))
        for w in warnings:
            verdict.append(("warn", w))

        # ---- pictures, videos, files
        table = self._table(profile, nb, items, cells, summary)
        image = render_lora_compare(dict(
            title=" vs ".join(runs) if len(runs) > 1 else runs[0], subtitle=subtitle, verdict=verdict,
            charts=self._charts(summary, items, include_base, bool(ctrl_prompts), len(id_prompts), scorer),
            shorts=[it["run_short"] if len(runs) > 1 else it["short"] for it in items],
            groups=[runs.index(it["run"]) for it in items], runs=runs,
            table=table,
            footer=["Every LoRA renders the same prompts, seeds and settings; the base renders are shared. Renders are "
                    "cached (output/lora_prober/cache), so adding a file to the list only renders that file."
                    + (" Video likeness is the mean over evenly spaced frames." if video else "")]))
        sheets = self._sheets(ctx, cells, results, summary, include_base)
        videos = self._videos(ctx, cells, results, summary, items, include_base)
        report = self._report(profile, nb, subtitle, verdict, table, plist, ctx)
        ui = self._save(ctx, image, sheets, videos, report, cells, results, summary, verdict, table, subtitle)
        Video = _video_type()
        video_out = Video(videos[0]) if (Video is not None and videos) else None
        return {"ui": ui, "result": (_pil_to_image(image), _image_batch(sheets), report, "\n".join(videos), video_out)}

    # ------------------------------------------------------------------ rendering

    @staticmethod
    def _key(ctx, lora_part, prompt, seed):
        parts = [ctx.model_fp, ctx.speed_id, ctx.vae_fp, lora_part, ctx.cond_fp[prompt], ctx.neg_fp, int(seed),
                 int(ctx.steps), round(float(ctx.cfg), 6), ctx.sampler_name, ctx.scheduler, int(ctx.width),
                 int(ctx.height)]
        if ctx.video:
            parts.append(["video", int(ctx.frames)])
        return rc.key_of(parts)

    def _render_cell(self, ctx, cell):
        profile, item = ctx.profile, cell["item"]
        lora_part, sub, strength = None, None, 0.0
        cell["change"] = None
        if item is not None:
            path, _mtime, sd, _meta = _load_lora(item["name"])
            sd, notes = _fit_lora(ctx.model, profile, sd)
            cell["notes"] = notes
            if item["spec"]:
                v = core.custom_variant(profile, "selection", item["spec"], ctx.nb)
                core.assign_keys(profile, sd, [v])
                keys = v.keys
                cell["how_layers"] = v.describe(profile, ctx.nb, True)
            else:
                keys = list(sd.keys())
                cell["how_layers"] = "all layers"
            if not keys:
                raise ValueError(f"{item['name']}" + (f" | {item['spec']}" if item["spec"] else "")
                                 + ": no LoRA layers selected" + (" that fit this checkpoint" if notes else "") + ".")
            strength = item["strength"] if item["strength"] is not None else ctx.strength
            cell["strength"] = strength
            lora_part = [rc.file_id(path), round(float(strength), 6),
                         hashlib.sha1("\n".join(sorted(keys)).encode()).hexdigest()]
            sub = {k: sd[k] for k in keys}
            info = core.analyze_lora(profile, sub)
            cell["change"] = math.sqrt(info.total_energy()) * abs(strength) if info.sq else NAN
        patched, out = None, []
        for kind, idx, prompt in ctx.prompts:
            for seed in ctx.seeds:
                comfy.model_management.throw_exception_if_processing_interrupted()
                key = self._key(ctx, lora_part, prompt, seed)
                entry, where = self._lookup(ctx, key, kind)
                if entry is None:
                    if patched is None:
                        patched = ctx.model if item is None else \
                            comfy.sd.load_lora_for_models(ctx.model, None, sub, strength, 0)[0]
                    entry = self._fresh(ctx, patched, key, cell, kind, idx, prompt, seed)
                    ctx.rendered += 1
                elif where == "disk":
                    ctx.from_disk += 1
                else:
                    ctx.from_mem += 1
                ctx.progress.render_done()
                out.append(dict(cell=cell, kind=kind, prompt=idx, seed=seed, entry=entry))
        del patched
        return out

    def _callback(self, ctx):
        def cb(step, x0, x, total_steps):
            preview = None
            if ctx.previewer is not None:
                if getattr(x0, "is_nested", False):
                    x0 = x0.tensors[0]
                try:
                    preview = ctx.previewer.decode_latent_to_preview_image("JPEG", x0)
                except Exception:
                    preview = None
            ctx.progress.in_render(step + 1, preview)

        return cb

    def _meta(self, ctx, cell, kind, prompt, seed):
        it = cell["item"]
        return {"model": ctx.model_label, "speed_lora": ctx.speed_label, "vae": ctx.vae_label,
                "lora": None if it is None else it["name"], "lora_strength": None if it is None else cell["strength"],
                "layers": None if it is None else cell["how_layers"], "prompt": prompt, "seed": seed,
                "steps": ctx.steps, "cfg": ctx.cfg, "sampler": ctx.sampler_name, "scheduler": ctx.scheduler,
                "width": ctx.width, "height": ctx.height, **({"frames": ctx.frames} if ctx.video else {})}

    def _fresh(self, ctx, patched, key, cell, kind, idx, prompt, seed):
        noise = comfy.sample.prepare_noise(ctx.latent, seed)
        samples = comfy.sample.sample(patched, noise, ctx.steps, ctx.cfg, ctx.sampler_name, ctx.scheduler,
                                      ctx.conds[prompt], ctx.neg, ctx.latent, denoise=1.0, callback=self._callback(ctx),
                                      disable_pbar=True, seed=seed)
        frames = media.decode_frames(ctx.vae, samples)
        meta = self._meta(ctx, cell, kind, prompt, seed)
        if not ctx.video:
            u8 = rc.to_u8(frames[0])
            img = rc.from_u8(u8)
            entry = {"key": key, "thumb": _jpeg(thumbnail(img, 168)), "gray": gray_small(img) if kind == "ctrl" else None,
                     "lay": layout_thumb(img), "embeds": {}, "path": None}
            if ctx.scorer is not None:
                entry["embeds"][ctx.scorer.key] = ctx.scorer.embed_render(img)
            if ctx.disk is not None:
                entry["path"] = ctx.disk.save(key, u8, entry, meta, ctx.scorer.key if ctx.scorer else None)
            if entry["path"] is None:  # no cache: keep the image in the run folder until it is filed by LoRA
                p = os.path.join(ctx.run_dir, ".raw", f"{key}.png")
                os.makedirs(os.path.dirname(p), exist_ok=True)
                Image.fromarray(u8).save(p)
                entry["path"] = p
            _cache_put(key, entry)
            return entry

        audio = media.decode_audio(ctx.audio_vae, samples)
        alat = samples.unbind()[-1].detach().float().cpu().numpy() if media.is_nested(samples) else None
        n = frames.shape[0]
        mid = frames[n // 2]
        fidx = _frame_indices(n)
        entry = {"key": key, "thumb": _jpeg(thumbnail(mid, 168)), "gray": gray_small(mid), "lay": layout_thumb(mid),
                 "frames": n, "fps": media.H3_FPS, "fidx": fidx, "frame_embeds": {}, "path": None, "wav": None,
                 "afp": ctx.audio_fp if audio is not None else "", "rate": audio[1] if audio else 0, "video": True,
                 "alat": alat}
        if ctx.scorer is not None:
            entry["frame_embeds"][ctx.scorer.key] = [ctx.scorer.embed_render(frames[i]) for i in fidx]
        u8 = media.to_u8(frames)
        del frames
        wav, rate = audio if audio is not None else (None, 0)

        def writer(p):
            media.write_clip(p, u8, media.H3_FPS, wav, rate)

        if ctx.disk is not None:
            entry["path"] = ctx.disk.save_video(key, writer, entry, meta, ctx.scorer.key if ctx.scorer else None,
                                                wav=wav, rate=rate, audio_latent=alat, audio_fp=entry["afp"])
            if entry["path"] is not None and wav is not None:
                entry["wav"] = ctx.disk.video_paths(key)[1]
        if entry["path"] is None:  # no cache: the clip lives in the run folder
            p = os.path.join(ctx.run_dir, ".raw", f"{key}.mp4")
            os.makedirs(os.path.dirname(p), exist_ok=True)
            writer(p)
            entry["path"] = p
            if wav is not None:
                entry["wav"] = p[:-4] + ".wav"
                media.write_wav(entry["wav"], wav, rate)
        _cache_put(key, entry)
        return entry

    def _lookup(self, ctx, key, kind):
        entry, where = _cache_get(key), "mem"
        if entry is None and ctx.disk is not None:
            entry = ctx.disk.load_video(key) if ctx.video else ctx.disk.load(key)
            where = "disk"
        if entry is None:
            return None, None
        if not entry.get("path") or not os.path.exists(entry["path"]):  # sheets and videos need the file itself
            return None, None
        if ctx.video:
            entry = self._refresh_video(ctx, key, entry)
            if entry is None:
                return None, None
        else:
            if ctx.scorer is not None and rc.DiskCache.embeds_for(entry, ctx.scorer.key) is None:
                u8 = self._pixels(ctx, key, entry)
                if u8 is None:
                    return None, None
                emb = ctx.scorer.embed_render(rc.from_u8(u8))
                entry["embeds"][ctx.scorer.key] = emb
                if ctx.disk is not None:
                    ctx.disk.save_embeds(key, ctx.scorer.key, emb)
            if kind == "ctrl" and entry.get("gray") is None:
                u8 = self._pixels(ctx, key, entry)
                if u8 is None:
                    return None, None
                entry["gray"] = gray_small(rc.from_u8(u8))
        _cache_put(key, entry)
        return entry, where

    @staticmethod
    def _pixels(ctx, key, entry):
        """A stored render's pixels: from the disk cache, else from the file the entry points at."""
        u8 = ctx.disk.load_image(key) if ctx.disk is not None else None
        if u8 is None:
            try:
                u8 = np.asarray(Image.open(entry["path"]).convert("RGB"))
            except Exception:
                return None
        return u8

    def _refresh_video(self, ctx, key, entry):
        """A cached clip scored by this scorer and, when an audio VAE is connected, with its sound."""
        if ctx.scorer is not None and rc.DiskCache.frame_embeds_for(entry, ctx.scorer.key) is None:
            want = set(entry["fidx"])
            per = {}
            for i, fr in enumerate(media.iter_frames(entry["path"])):
                if i in want:
                    per[i] = ctx.scorer.embed_render(torch.from_numpy(fr.astype(np.float32) / 255.0))
            if len(per) != len(want):
                return None
            embeds = [per[i] for i in entry["fidx"]]
            entry["frame_embeds"][ctx.scorer.key] = embeds
            if ctx.disk is not None:
                ctx.disk.update_video(key, frame_embeds=embeds, scorer_key=ctx.scorer.key)
        if ctx.audio_vae is not None and entry.get("afp") != ctx.audio_fp and entry.get("alat") is not None:
            lat = torch.from_numpy(np.asarray(entry["alat"], dtype=np.float32))
            audio = media.decode_audio_latent(ctx.audio_vae, lat)
            if audio is not None:
                wav, rate = audio
                if ctx.disk is not None:
                    ctx.disk.update_video(key, wav=wav, rate=rate, audio_fp=ctx.audio_fp)
                    entry["wav"] = ctx.disk.video_paths(key)[1]
                else:  # no disk cache: a copy of the clip in this run, with the new sound
                    p = os.path.join(ctx.run_dir, ".raw", f"{key}.mp4")
                    os.makedirs(os.path.dirname(p), exist_ok=True)
                    if os.path.abspath(p) != os.path.abspath(entry["path"]):
                        shutil.copyfile(entry["path"], p)
                    media.remux_audio(p, wav, rate)
                    entry["path"], entry["wav"] = p, p[:-4] + ".wav"
                    media.write_wav(entry["wav"], wav, rate)
                entry["afp"], entry["rate"] = ctx.audio_fp, rate
        return entry

    @staticmethod
    def _clip_audio(entry):
        """A clip's sound as (waveform, rate): the stored WAV, else the clip's own track; None when silent."""
        wavp = entry.get("wav")
        if wavp and os.path.exists(wavp):
            return media.read_wav(wavp)
        if entry.get("rate"):  # rendered with sound but the WAV is gone: read it back from the MP4
            try:
                return media.read_audio(entry["path"])
            except Exception as e:
                log.warning("LoRA compare: could not read the sound of %s (%s)", entry["path"], e)
        return None

    # ------------------------------------------------------------------ outputs

    @staticmethod
    def _charts(summary, items, include_base, has_ctrl, n_id, scorer):
        def series(key, se=None):
            return [(summary[it["key"]].get(key, NAN), summary[it["key"]].get(se, NAN) if se else NAN) for it in items]

        b = summary.get("base") or {}
        out = []
        if scorer is not None:
            out.append(dict(title="Likeness", sub=f"{scorer.name} similarity to the references (identity prompts)",
                            values=series("id_sim", "id_se"), ref=(b.get("id_sim", NAN), "base"), fmt="num"))
            if has_ctrl:
                out.append(dict(title="Leak", sub="the same similarity on the control prompts",
                                values=series("leak", "leak_se"), ref=(b.get("leak", NAN), "base"), fmt="num"))
        if has_ctrl and include_base:
            out.append(dict(title="Drift", sub="how much control renders change against the base model",
                            values=series("drift"), ref=(0.0, None), fmt="num", ymin=0.0))
        if include_base and n_id > 1:
            out.append(dict(title="Prompt variety", sub="identity renders vs each other; base = 100%",
                            values=series("variety_rel", "variety_rel_se"), ref=(1.0, "base"), fmt="pct", ymin=0.0))
        return out

    @staticmethod
    def _table(profile, nb, items, cells, summary):
        by = {c["name"]: c for c in cells}
        changes = [by[it["key"]].get("change") for it in items]
        cmax = max([c for c in changes if c is not None and core.finite(c)] or [0.0])
        runs = []
        for it in items:
            if it["run"] not in runs:
                runs.append(it["run"])
        rows = []
        if "base" in by:
            s = summary["base"]
            rows.append(dict(label="base (no LoRA)", name="", how="", id_sim=s["id_sim"], id_se=s["id_se"],
                             leak=s["leak"], drift=s["drift"], variety=s.get("variety_rel", NAN), change=None, color=None))
        for it in items:
            c, s = by[it["key"]], summary[it["key"]]
            ch = c.get("change")
            rows.append(dict(label=it["label"], name=it["name"], how=f"{c['strength']:g} · {c['how_layers']}",
                             id_sim=s["id_sim"], id_se=s["id_se"], leak=s["leak"], drift=s["drift"],
                             variety=s.get("variety_rel", NAN),
                             change=(ch / cmax) if ch is not None and core.finite(ch) and cmax > 0 else None,
                             color=RUN_COLORS[runs.index(it["run"]) % len(RUN_COLORS)] if len(runs) > 1 else None))
        return rows

    @staticmethod
    def _score_scale(summary, cells):
        """(low, span) for colouring scores: the base model's likeness up to the best LoRA's."""
        b = summary.get("base") or {}
        sims = [summary[c["name"]]["id_sim"] for c in cells if core.finite(summary[c["name"]]["id_sim"])]
        lo = b.get("id_sim") if core.finite(b.get("id_sim")) else (min(sims) if sims else 0.0)
        hi = max(sims) if sims else 1.0
        return lo, (hi - lo) if hi - lo > 1e-6 else 1.0

    @staticmethod
    def _score_color(kind, sim, lo, span):
        if kind == "id":
            t = (sim - lo) / span if core.finite(sim) else -1
            return GOOD if t >= 0.8 else WARN if t >= 0.4 else BAD
        return LEAK if core.finite(sim) and sim >= lo + 0.5 * span else MUTED

    def _sheets(self, ctx, cells, results, summary, include_base):
        lo, span = self._score_scale(summary, cells)
        rows = []
        for c in cells:
            s = summary[c["name"]]["id_sim"]
            sub = ("no LoRA" if c["item"] is None else os.path.basename(c["item"]["name"])) + \
                  (f" · likeness {s:.2f}" if core.finite(s) else "")
            rows.append((c["label"], sub))
        columns = [(f"{k}{i + 1}", k) for k, i, _ in ctx.prompts]
        legend = [f"{k}{i + 1}: {p}" for k, i, p in ctx.prompts]
        if ctx.scorer is not None:
            legend.append(f"Scores: {ctx.scorer.name} similarity to the references"
                          + (" (mean over frames)" if ctx.video else "") + ".")
        by = {(r["cell"]["name"], r["kind"], r["prompt"], r["seed"]): r for r in results}
        sheets = []
        for n, seed in enumerate(ctx.seeds, start=1):
            grid = []
            for c in cells:
                line = []
                for k, i, _ in ctx.prompts:
                    r = by.get((c["name"], k, i, seed))
                    sim = r["sim"] if r else NAN
                    line.append((_unjpeg(r["entry"]["thumb"]) if r else None, sim, self._score_color(k, sim, lo, span)))
                grid.append(line)
            title = (f"LoRA Compare · seed {seed}" + (f" ({n} of {len(ctx.seeds)})" if len(ctx.seeds) > 1 else "")
                     + (" · middle frames" if ctx.video else ""))
            sheets.append(render_contact_sheet(rows, columns, grid, title=title, legend=legend))
        return sheets

    def _layout(self, ctx, cells, items, include_base):
        lay = core.compare_layout(items, include_base)
        if lay is not None:
            pos, rows, cols = lay
            placed = []
            for it, r, col in pos:
                c = cells[0] if it is None else next(x for x in cells if x["item"] is it)
                placed.append((c, r, col))
            return placed, rows, cols
        w, h = ctx.width, ctx.height
        _, cols, rows = media.best_grid(len(cells), w, h, media.VideoGrid.HEAD + media.VideoGrid.FOOT,
                                        *media.MAX_CANVAS)
        return [(c, i // cols, i % cols) for i, c in enumerate(cells)], rows, cols

    def _videos(self, ctx, cells, results, summary, items, include_base):
        vdir = os.path.join(ctx.run_dir, "videos")
        by = {(r["cell"]["name"], r["kind"], r["prompt"], r["seed"]): r for r in results}
        out = []
        try:
            if ctx.video:
                placed, rows, cols = self._layout(ctx, cells, items, include_base)
                matrix = core.compare_layout(items, include_base) is not None
                runs = []
                for it in items:
                    if it["run"] not in runs:
                        runs.append(it["run"])
                lo, span = self._score_scale(summary, cells)
                for kind, idx, prompt in ctx.prompts:
                    for seed in ctx.seeds:
                        vcells, clips, audio = [], [], []
                        for c, r, col in placed:
                            res = by[(c["name"], kind, idx, seed)]
                            sim = res["sim"]
                            it = c["item"]
                            label = "base" if it is None else (it["run_label"] if matrix else it["label"])
                            vcells.append(dict(label=label, row=r, col=col, name=c["label"],
                                               badge=f"{sim:.2f}" if core.finite(sim) else "",
                                               badge_color=self._score_color(kind, sim, lo, span)))
                            clips.append(res["entry"]["path"])
                            audio.append(self._clip_audio(res["entry"]))
                        path = os.path.join(vdir, f"compare_{kind}{idx + 1}_seed{seed}.mp4")
                        title = ellipsize(ImageDraw.Draw(Image.new("RGB", (4, 4))), prompt, font(22, "bold"), 1800)
                        sub = (f"{'identity' if kind == 'id' else 'control'} prompt {idx + 1} · seed {seed} · "
                               f"{ctx.frames} frames"
                               + (" · rows: runs" if matrix else "")
                               + (" · numbers: likeness" if ctx.scorer is not None else ""))
                        media.comparison_video(path, vcells, clips, rows, cols, title, sub, fps=media.H3_FPS,
                                               audio=audio, row_labels=runs if matrix else None)
                        out.append(path)
            else:
                for seed in ctx.seeds:
                    pages = [self._flip_page(ctx, c, by, seed) for c in cells]
                    path = os.path.join(vdir, f"flipbook_seed{seed}.mp4")
                    media.flipbook_video(path, pages)
                    out.append(path)
        except ImportError as e:  # PyAV missing: ComfyUI normally has it
            ctx.warnings.append(f"Could not write videos ({e}).")
        return out

    @staticmethod
    def _flip_page(ctx, cell, by, seed, size=384):
        """One flip-book page: a LoRA's renders for every prompt of a seed, labelled."""
        imgs = []
        for k, i, p in ctx.prompts:
            r = by.get((cell["name"], k, i, seed))
            im = None
            if r and r["entry"].get("path") and os.path.exists(r["entry"]["path"]):
                im = Image.open(r["entry"]["path"]).convert("RGB")
                im.thumbnail((size, size), Image.LANCZOS)
            imgs.append((f"{k}{i + 1}", im, r["sim"] if r else NAN))
        n = len(imgs)
        per = min(n, max(1, 1880 // (size + 12)))
        nrows = math.ceil(n / per)
        W = 24 + per * (size + 12)
        H = 70 + nrows * (size + 40) + 10
        page = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(page)
        d.text((16, 12), ellipsize(d, cell["label"], font(26, "bold"), W - 32), font=font(26, "bold"), fill=TEXT)
        sub = ("no LoRA" if cell["item"] is None else os.path.basename(cell["item"]["name"])) + f" · seed {seed}"
        d.text((16, 44), ellipsize(d, sub, font(15), W - 32), font=font(15), fill=MUTED)
        for j, (lab, im, sim) in enumerate(imgs):
            x = 16 + (j % per) * (size + 12)
            y = 70 + (j // per) * (size + 40)
            d.rectangle([x, y, x + size, y + size], fill=PANEL)
            if im is not None:
                page.paste(im, (x + (size - im.width) // 2, y + (size - im.height) // 2))
            d.text((x, y + size + 6), lab + (f"  ·  {sim:.2f}" if core.finite(sim) else ""), font=font(15), fill=MUTED)
        return page

    @staticmethod
    def _report(profile, nb, subtitle, verdict, table, plist, ctx):
        out = ["LoRA compare", *subtitle, ""]
        for kind, text in verdict:
            out.append(f"{'!' if kind in ('warn', 'bad') else '-'} {text}")
        out.append("")
        out.append(f"{'LoRA':<40} {'strength · layers':<28} {'likeness':>9} {'±':>6} {'leak':>6} {'drift':>6} "
                   f"{'variety':>8} {'change':>7}")
        for r in table:
            ch = r["change"]
            out.append(f"{(r['label'] + (' · ' + os.path.basename(r['name']) if r['name'] else ''))[:40]:<40} "
                       f"{r['how'][:28]:<28} {core.fmt(r['id_sim'], '.3f'):>9} {core.fmt(r['id_se'], '.3f'):>6} "
                       f"{core.fmt(r['leak'], '.3f'):>6} {core.fmt(r['drift'], '.3f'):>6} "
                       f"{(core.fmt(100 * r['variety'], '.0f') + '%') if core.finite(r['variety']) else '-':>8} "
                       f"{(core.fmt(100 * ch, '.0f') + '%') if ch is not None and core.finite(ch) else '-':>7}")
        out.append("")
        out.append("Prompts:")
        out += [f"  {k}{i + 1}: {p}" for k, i, p in plist]
        return "\n".join(out)

    def _save(self, ctx, image, sheets, videos, report, cells, results, summary, verdict, table, subtitle):
        out_root = folder_paths.get_output_directory()
        run_dir = ctx.run_dir
        image.save(os.path.join(run_dir, "compare.png"))
        sheet_names = [f"sheet_seed{seed}.png" for seed in ctx.seeds]
        for s, n in zip(sheets, sheet_names):
            s.save(os.path.join(run_dir, n))
        with open(os.path.join(run_dir, "report.txt"), "w", encoding="utf-8") as fh:
            fh.write(report)
        # every render, by LoRA: hard links to the cache (no extra space)
        folders, used = {}, set()
        for i, c in enumerate(cells):
            stem = "base" if c["item"] is None else os.path.splitext(os.path.basename(c["item"]["name"]))[0]
            f = _safe_name(f"{i:02d}_{stem}" + ("" if c["item"] is None or not (c["item"]["spec"] or c["item"]["strength"]
                                                                                 is not None) else f"_{c['short']}"))
            while f in used:
                f += "_"
            used.add(f)
            folders[c["name"]] = f
        renders = {}
        raw = os.path.join(run_dir, ".raw")  # renders made without the disk cache: filed by LoRA, then removed
        for r in results:
            e = r["entry"]
            src = e.get("path")
            if not src or not os.path.exists(src):
                continue
            ext = os.path.splitext(src)[1]
            rel = f"renders/{folders[r['cell']['name']]}/{r['kind']}{r['prompt'] + 1}_seed{r['seed']}{ext}"
            dst = os.path.join(run_dir, *rel.split("/"))
            if os.path.abspath(src) != os.path.abspath(dst):
                try:
                    rc.link_or_copy(src, dst)
                except OSError as err:
                    log.warning("LoRA compare: could not add %s to the run folder (%s)", rel, err)
                    continue
            renders[(r["cell"]["name"], r["kind"], r["prompt"], r["seed"])] = rel
            if os.path.dirname(os.path.abspath(src)) == os.path.abspath(raw):
                e["path"] = dst  # the memory cache follows the filed copy (an MP4 carries its own sound)
                if e.get("wav"):
                    e["wav"] = None
        if os.path.isdir(raw):
            shutil.rmtree(raw, ignore_errors=True)
        data = {
            "settings": subtitle,
            "verdict": [{"kind": k, "text": t} for k, t in verdict],
            "loras": [{"label": c["label"], "file": None if c["item"] is None else c["item"]["name"],
                       "run": c["run"], "step": None if c["item"] is None else c["item"]["step"],
                       "strength": c.get("strength"), "layers": c.get("how_layers"), "folder": folders[c["name"]],
                       "summary": summary[c["name"]]} for c in cells],
            "renders": [{"lora": r["cell"]["label"], "kind": r["kind"], "prompt": r["prompt"], "seed": r["seed"],
                         "likeness": r["sim"], "file": renders.get((r["cell"]["name"], r["kind"], r["prompt"],
                                                                    r["seed"]))} for r in results],
            "prompts": [{"kind": k, "index": i, "text": p} for k, i, p in ctx.prompts],
            "videos": [os.path.relpath(v, run_dir).replace(os.sep, "/") for v in videos],
        }
        from .probe_nodes import LoRAProber

        with open(os.path.join(run_dir, "compare.json"), "w", encoding="utf-8") as fh:
            json.dump(LoRAProber._clean(data), fh, indent=1)
        sub = os.path.relpath(run_dir, out_root).replace(os.sep, "/")
        log.info("LoRA compare saved to %s", run_dir)
        if ctx.video and videos:  # the node previews the comparison videos (the images are on its outputs)
            return {"images": [{"filename": os.path.basename(v), "subfolder": f"{sub}/videos", "type": "output"}
                               for v in videos], "animated": [True]}
        return {"images": [{"filename": n, "subfolder": sub, "type": "output"} for n in ["compare.png"] + sheet_names]}


NODE_CLASS_MAPPINGS = {"LoRAProberCompare": LoRAProberCompare}
NODE_DISPLAY_NAME_MAPPINGS = {"LoRAProberCompare": "LoRA Prober Compare"}
