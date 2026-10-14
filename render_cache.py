"""Disk cache of LoRA Prober renders, and the fingerprints that key it.

Every render is stored once, under a key built from everything that decides its pixels: the base model
(file, extra patches, sampling options), the speed LoRA, the VAE, the probed LoRA file and the layers the
variant keeps, the prompt's conditioning, the seed and the sampler settings. The identity scorer is not
part of the key: embeddings are stored next to the image, one set per scorer, so changing the scorer or
the reference photos never re-renders anything.

Layout (sharded by the first two characters of the key):
    <root>/ab/<key>.png   the image as decoded, 8-bit, with the settings in a "lora_prober" text chunk
    <root>/ab/<key>.npz   thumbnail (JPEG bytes), grey and colour thumbnails, embeddings per scorer

Video renders (MiniMax H3, LoRA Prober Compare node):
    <root>/ab/<key>.mp4   the clip with its sound (H.264 + AAC)
    <root>/ab/<key>.wav   the sound, lossless (used for the comparison videos)
    <root>/ab/<key>.npz   thumbnails of the middle frame, per-frame embeddings per scorer, and the audio latent
                          (so the sound can be decoded later with another audio VAE without rendering again)
"""

from __future__ import annotations

import functools
import hashlib
import io
import json
import logging
import os
import types
import uuid

import numpy as np
import torch
from PIL import Image, PngImagePlugin

log = logging.getLogger("lora_prober")

CACHE_VERSION = 1  # bump when anything changes what a cached render or its derived data means


# ---------------------------------------------------------------- fingerprints


_FILE_IDS = {}


def file_id(path: str) -> str:
    """Content identity of a file without reading all of it: size plus hashes of its start, middle and end.
    Renaming or copying a file keeps its id; a different checkpoint gets a different one."""
    st = os.stat(path)
    memo = (os.path.abspath(path), st.st_size, st.st_mtime_ns)
    hit = _FILE_IDS.get(memo)
    if hit:
        return hit
    h = hashlib.sha1(str(st.st_size).encode())
    chunk = 1 << 20
    with open(path, "rb") as f:
        for pos in sorted({0, max(0, st.st_size // 2 - chunk // 2), max(0, st.st_size - chunk)}):
            f.seek(pos)
            h.update(f.read(chunk))
    out = h.hexdigest()[:24]
    _FILE_IDS[memo] = out
    return out


def _tensor_bytes(t: torch.Tensor, full: bool) -> bytes:
    """All values (full) or an evenly spaced sample of 1024, as float32 bytes."""
    flat = t.detach().reshape(-1)
    n = flat.numel()
    if not full and n > 1024:
        idx = torch.arange(1024, dtype=torch.int64) * (n - 1) // 1023  # integer maths: same on every device
        flat = flat[idx.to(flat.device)]
    return flat.float().cpu().numpy().astype(np.float32).tobytes()


class _Hasher:
    """Feeds an arbitrary object graph (tensors, containers, functions and their closures, modules,
    adapters) into a hash, deterministically across sessions. Best effort: what it cannot read becomes
    its type name, and the node's validation render catches what that misses."""

    def __init__(self, full_tensors=False, budget=200_000):
        self.h = hashlib.sha1()
        self.full = full_tensors
        self.budget = budget
        self.seen = set()

    def text(self, s):
        self.h.update(str(s).encode("utf-8", "replace"))
        self.h.update(b"\x00")

    def tensor(self, t):
        self.text(f"T{tuple(t.shape)}{t.dtype}")
        try:
            self.h.update(_tensor_bytes(t, self.full))
        except Exception:  # meta tensors, exotic quantised types: shape and dtype only
            self.text("?")

    def feed(self, obj, depth=0):
        self.budget -= 1
        if self.budget < 0:
            return
        if obj is None or isinstance(obj, (bool, int, float, str, bytes)):
            self.text(repr(obj))
            return
        if isinstance(obj, uuid.UUID):  # random per session
            self.text("uuid")
            return
        if isinstance(obj, torch.Tensor):
            self.tensor(obj)
            return
        if isinstance(obj, np.ndarray):
            self.text(f"A{obj.shape}{obj.dtype}")
            self.h.update(np.ascontiguousarray(obj).tobytes()[: 1 << 16])
            return
        if isinstance(obj, (torch.device, torch.dtype)):
            self.text(str(obj))
            return
        oid = id(obj)
        if oid in self.seen:
            self.text("cycle")
            return
        self.seen.add(oid)
        if isinstance(obj, (list, tuple)):
            self.text(f"[{len(obj)}")
            for x in obj:
                self.feed(x, depth + 1)
            self.text("]")
        elif isinstance(obj, (set, frozenset)):
            self.text(f"set{len(obj)}")
            for r in sorted(repr(x) for x in obj if isinstance(x, (str, int, float, bool, tuple))):
                self.text(r)
        elif isinstance(obj, dict):
            self.text(f"{{{len(obj)}")
            for k in sorted(obj, key=lambda k: repr(k) if isinstance(k, (str, int, float, tuple)) else type(k).__name__):
                if isinstance(k, str) and "uuid" in k.lower():
                    continue
                self.feed(k, depth + 1)
                self.feed(obj[k], depth + 1)
            self.text("}")
        elif isinstance(obj, torch.nn.Module):
            self.text(f"M{type(obj).__module__}.{type(obj).__qualname__}")
            items = list(obj.state_dict().items())
            step = max(1, len(items) // 64)
            for name, t in items[::step]:
                self.text(name)
                self.tensor(t)
        elif isinstance(obj, functools.partial):
            self.text("partial")
            self.feed(obj.func, depth + 1)
            self.feed(obj.args, depth + 1)
            self.feed(obj.keywords, depth + 1)
        elif isinstance(obj, types.MethodType):
            self.text("method")
            self.feed(obj.__func__, depth + 1)
            if depth < 6:
                self._attrs(obj.__self__, depth + 1)
        elif isinstance(obj, types.FunctionType):
            self.text(f"F{obj.__module__}.{obj.__qualname__}")
            self.feed(obj.__defaults__, depth + 1)
            for cell in obj.__closure__ or ():
                try:
                    self.feed(cell.cell_contents, depth + 1)
                except ValueError:  # empty cell
                    self.text("empty")
        elif type(obj).__name__ == "ModelPatcher" or hasattr(obj, "patches_uuid"):
            self.text("ModelPatcher")  # another model referenced from options: its class only
            self.text(type(getattr(obj, "model", None)).__name__)
        elif depth < 6:
            self._attrs(obj, depth)
        else:
            self.text(type(obj).__qualname__)

    def _attrs(self, obj, depth):
        self.text(f"O{type(obj).__module__}.{type(obj).__qualname__}")
        try:
            d = vars(obj)
        except TypeError:
            slots = getattr(type(obj), "__slots__", ())
            d = {s: getattr(obj, s, None) for s in slots} if slots else {}
        self.feed({k: v for k, v in d.items() if not k.startswith("__")}, depth + 1)

    def hexdigest(self):
        return self.h.hexdigest()[:24]


def fingerprint(obj, full_tensors=False) -> str:
    hs = _Hasher(full_tensors=full_tensors)
    hs.feed(obj)
    return hs.hexdigest()


def _init_id(init):
    """A core loader's cached_patcher_init: the loader name with file paths replaced by their content id."""
    fn, args = init[0], init[1]

    def conv(a):
        if isinstance(a, str) and os.path.isfile(a):
            try:
                return {"file": file_id(a)}
            except OSError:
                return a
        if isinstance(a, (list, tuple)):
            return [conv(x) for x in a]
        return a

    return [getattr(fn, "__qualname__", str(fn)), conv(list(args)), init[2] if len(init) > 2 else None]


def model_fingerprint(m) -> str:
    """The MODEL input: weights (by file when ComfyUI knows it, else a sample of them), extra patches such
    as LoRAs applied before the node, object patches (e.g. a sampling shift) and model options."""
    hs = _Hasher()
    base = getattr(m, "model", None)
    hs.text(f"{type(base).__module__}.{type(base).__qualname__}")
    cfg = getattr(getattr(base, "model_config", None), "unet_config", None)
    hs.feed(cfg)
    init = getattr(m, "cached_patcher_init", None)
    if init:
        hs.text("init")
        hs.feed(_init_id(init))
    else:
        dm = getattr(base, "diffusion_model", base)
        try:
            items = list(dm.state_dict().items())
            step = max(1, len(items) // 48)
            for name, t in items[::step]:
                hs.text(name)
                hs.tensor(t)
        except Exception:
            hs.text("weights?")
    hs.text("patches")
    hs.feed(getattr(m, "patches", {}))
    hs.text("objects")
    hs.feed(getattr(m, "object_patches", {}))
    hs.text("options")
    hs.feed(getattr(m, "model_options", {}))
    hs.text("wrappers")
    hs.feed(getattr(m, "wrappers", {}))
    hs.feed(len(getattr(m, "hook_patches", {}) or {}))
    return hs.hexdigest()


def vae_fingerprint(vae) -> str:
    hs = _Hasher()
    hs.text(type(vae).__qualname__)
    init = getattr(getattr(vae, "patcher", None), "cached_patcher_init", None)
    fsm = getattr(vae, "first_stage_model", None)
    if init:
        hs.feed(_init_id(init))
        hs.text(type(fsm).__qualname__)
    elif fsm is not None:
        hs.feed(fsm)
    else:
        hs._attrs(vae, 0)
    for attr in ("vae_dtype", "downscale_ratio", "upscale_ratio", "latent_channels", "output_channels"):
        hs.feed(repr(getattr(vae, attr, None)))
    return hs.hexdigest()


def cond_fingerprint(cond) -> str:
    """Conditioning from the text encoder, hashed exactly (so it also identifies the text encoder)."""
    return fingerprint(cond, full_tensors=True)


def key_of(parts) -> str:
    return hashlib.sha1(json.dumps([CACHE_VERSION, parts], sort_keys=True, default=str).encode()).hexdigest()


# ---------------------------------------------------------------- the disk cache


def _scorer_slot(scorer_key: str) -> str:
    return "e_" + hashlib.sha1(scorer_key.encode()).hexdigest()[:16]


class DiskCache:
    def __init__(self, root: str):
        self.root = root
        self.writable = True
        os.makedirs(root, exist_ok=True)

    def paths(self, key):
        d = os.path.join(self.root, key[:2])
        return os.path.join(d, key + ".png"), os.path.join(d, key + ".npz")

    def load(self, key):
        """Entry dict (thumb, gray, lay, embeds by scorer key, path) or None. Embeddings are read lazily by
        slot name, so callers ask for one scorer with embeds_for()."""
        png, npz = self.paths(key)
        if not (os.path.exists(png) and os.path.exists(npz)):
            return None
        try:
            with np.load(npz, allow_pickle=False) as z:
                if int(z["v"]) != CACHE_VERSION:
                    return None
                gray = z["gray"]
                entry = {"thumb": z["thumb"].tobytes(), "gray": gray if gray.size else None, "lay": z["lay"],
                         "embeds": {}, "slots": {k: z[k] for k in z.files if k.startswith("e_")},
                         "path": png, "key": key}
            return entry
        except Exception as e:  # half-written or damaged: treat as missing
            log.debug("render cache: unreadable %s (%s)", npz, e)
            return None

    @staticmethod
    def embeds_for(entry, scorer_key):
        if scorer_key in entry["embeds"]:
            return entry["embeds"][scorer_key]
        slot = entry.get("slots", {}).get(_scorer_slot(scorer_key))
        if slot is not None:
            entry["embeds"][scorer_key] = slot
        return slot

    def load_image(self, key):
        png, _ = self.paths(key)
        try:
            with Image.open(png) as im:
                return np.asarray(im.convert("RGB")).copy()
        except Exception:
            return None

    def _write(self, path, writer):
        tmp = f"{path}.tmp{os.getpid()}"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        writer(tmp)
        os.replace(tmp, path)

    def _write_npz(self, path, arrays):
        def w(tmp):
            with open(tmp, "wb") as f:
                np.savez_compressed(f, **arrays)
        self._write(path, w)

    def save(self, key, image_u8, entry, meta, scorer_key):
        """Store a fresh render. Returns the PNG path, or None when the disk cannot be written."""
        if not self.writable:
            return None
        png, npz = self.paths(key)
        try:
            info = PngImagePlugin.PngInfo()
            info.add_text("lora_prober", json.dumps(meta, default=str))
            self._write(png, lambda tmp: Image.fromarray(image_u8).save(tmp, format="PNG", compress_level=4,
                                                                        pnginfo=info))
            arrays = {"v": np.array(CACHE_VERSION), "thumb": np.frombuffer(entry["thumb"], dtype=np.uint8),
                      "gray": entry["gray"] if entry["gray"] is not None else np.zeros((0, 0), np.uint8),
                      "lay": entry["lay"]}
            if scorer_key is not None:  # the LoRA Prober Compare node can run without references (no scorer)
                arrays[_scorer_slot(scorer_key)] = np.asarray(entry["embeds"][scorer_key], dtype=np.float32)
            self._write_npz(npz, arrays)
            return png
        except OSError as e:
            self.writable = False
            log.warning("render cache: cannot write to %s (%s); renders from now on are not cached on disk",
                        self.root, e)
            return None
        except Exception as e:
            log.warning("render cache: could not store a render (%s: %s)", type(e).__name__, e)
            return None

    def save_embeds(self, key, scorer_key, embeds):
        if not self.writable:
            return
        _, npz = self.paths(key)
        try:
            with np.load(npz, allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files}
            arrays[_scorer_slot(scorer_key)] = np.asarray(embeds, dtype=np.float32)
            self._write_npz(npz, arrays)
        except Exception as e:
            log.debug("render cache: could not add embeddings to %s (%s)", npz, e)

    # ------------------------------------------------------------------ video renders

    def video_paths(self, key):
        d = os.path.join(self.root, key[:2])
        return os.path.join(d, key + ".mp4"), os.path.join(d, key + ".wav"), os.path.join(d, key + ".npz")

    def load_video(self, key):
        """Video entry (thumb, gray, lay, frames, fps, fidx, per-frame embedding slots, audio latent, paths) or None."""
        mp4, wav, npz = self.video_paths(key)
        if not (os.path.exists(mp4) and os.path.exists(npz)):
            return None
        try:
            with np.load(npz, allow_pickle=False) as z:
                if int(z["v"]) != CACHE_VERSION or str(z["kind"]) != "video":
                    return None
                gray = z["gray"]
                alat = z["alat"]
                return {"thumb": z["thumb"].tobytes(), "gray": gray if gray.size else None, "lay": z["lay"],
                        "frames": int(z["frames"]), "fps": float(z["fps"]), "fidx": z["fidx"].tolist(),
                        "frame_embeds": {}, "slots": {k: z[k] for k in z.files if k.startswith("e_")},
                        "alat": alat if alat.size else None, "afp": str(z["afp"]), "rate": int(z["rate"]),
                        "path": mp4, "wav": wav if os.path.exists(wav) else None, "key": key, "video": True}
        except Exception as e:
            log.debug("render cache: unreadable %s (%s)", npz, e)
            return None

    @staticmethod
    def frame_embeds_for(entry, scorer_key):
        """Per scored frame, the embeddings [n, d] a scorer gave (list), or None."""
        if scorer_key in entry["frame_embeds"]:
            return entry["frame_embeds"][scorer_key]
        slot = _scorer_slot(scorer_key)
        flat, counts = entry.get("slots", {}).get(slot), entry.get("slots", {}).get(slot + "_n")
        if flat is None or counts is None:
            return None
        out, pos = [], 0
        for n in counts.tolist():
            out.append(flat[pos : pos + n])
            pos += n
        entry["frame_embeds"][scorer_key] = out
        return out

    @staticmethod
    def _embed_arrays(scorer_key, per_frame):
        dim = next((e.shape[1] for e in per_frame if e.ndim == 2 and e.shape[1]), 1)
        flat = np.concatenate([np.asarray(e, np.float32).reshape(-1, dim) for e in per_frame]) if per_frame else \
            np.zeros((0, dim), np.float32)
        counts = np.array([int(np.asarray(e).reshape(-1, dim).shape[0]) for e in per_frame], dtype=np.int32)
        slot = _scorer_slot(scorer_key)
        return {slot: flat.astype(np.float32), slot + "_n": counts}

    def save_video(self, key, writer, entry, meta, scorer_key, wav=None, rate=0, audio_latent=None, audio_fp=""):
        """Store a fresh video render. writer(path) writes the MP4. Returns the MP4 path, or None."""
        if not self.writable:
            return None
        mp4, wav_path, npz = self.video_paths(key)
        try:
            os.makedirs(os.path.dirname(mp4), exist_ok=True)
            writer(mp4)
            if wav is not None:
                from .media import write_wav

                write_wav(wav_path, wav, rate)
            elif os.path.exists(wav_path):
                os.remove(wav_path)
            arrays = {"v": np.array(CACHE_VERSION), "kind": np.array("video"),
                      "thumb": np.frombuffer(entry["thumb"], dtype=np.uint8),
                      "gray": entry["gray"] if entry["gray"] is not None else np.zeros((0, 0), np.uint8),
                      "lay": entry["lay"], "frames": np.array(int(entry["frames"])), "fps": np.array(float(entry["fps"])),
                      "fidx": np.array(entry["fidx"], dtype=np.int32),
                      "alat": (np.asarray(audio_latent, np.float32) if audio_latent is not None
                               else np.zeros((0,), np.float32)),
                      "afp": np.array(audio_fp or ""), "rate": np.array(int(rate or 0)),
                      "meta": np.array(json.dumps(meta, default=str))}
            if scorer_key is not None:
                arrays.update(self._embed_arrays(scorer_key, entry["frame_embeds"][scorer_key]))
            self._write_npz(npz, arrays)
            return mp4
        except OSError as e:
            self.writable = False
            log.warning("render cache: cannot write to %s (%s); renders from now on are not cached on disk",
                        self.root, e)
            return None
        except Exception as e:
            log.warning("render cache: could not store a video render (%s: %s)", type(e).__name__, e)
            return None

    def update_video(self, key, frame_embeds=None, scorer_key=None, wav=None, rate=0, audio_fp=None):
        """Add a scorer's frame embeddings, or replace the sound (WAV, the MP4's audio track and its fingerprint)."""
        if not self.writable:
            return
        mp4, wav_path, npz = self.video_paths(key)
        try:
            with np.load(npz, allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files}
            if frame_embeds is not None:
                arrays.update(self._embed_arrays(scorer_key, frame_embeds))
            if audio_fp is not None:
                from .media import remux_audio, write_wav

                if wav is not None:
                    write_wav(wav_path, wav, rate)
                    remux_audio(mp4, wav, rate)
                arrays["afp"], arrays["rate"] = np.array(audio_fp), np.array(int(rate or 0))
            self._write_npz(npz, arrays)
        except Exception as e:
            log.debug("render cache: could not update %s (%s)", npz, e)


def to_u8(image: torch.Tensor) -> np.ndarray:
    """[H,W,3] float 0-1 -> uint8, the exact pixels that are stored and scored."""
    return (image.detach().float().clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)


def from_u8(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr.astype(np.float32) / 255.0)


def jpeg_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def link_or_copy(src: str, dst: str) -> str:
    """Hard link (no extra disk space) when possible, else a copy. Returns 'link' or 'copy'."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        import shutil

        shutil.copyfile(src, dst)
        return "copy"
