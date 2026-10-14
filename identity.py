"""Identity similarity and image helpers for the LoRA Prober.

Two scorers:
  ArcFace (insightface, buffalo_l) - face identity, the right choice for real people
  CLIP Vision (any ComfyUI CLIP_VISION model) - whole-image similarity, a fallback for
    stylised or non-human characters where face recognition does not work
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ------------------------------------------------------------------ image utils


def to_rgb(images: torch.Tensor) -> torch.Tensor:
    """[B,H,W,C] float 0-1 -> [B,H,W,3]; alpha (Qwen-Image 2.1 can emit RGBA) is put over white."""
    if images.shape[-1] == 4:
        rgb, a = images[..., :3], images[..., 3:4]
        return rgb * a + (1.0 - a)
    if images.shape[-1] == 1:
        return images.repeat(1, 1, 1, 3)
    return images[..., :3]


def to_uint8(image: torch.Tensor) -> np.ndarray:
    """[H,W,3] float 0-1 -> uint8 RGB array."""
    return (image.detach().float().clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)


def thumbnail(image: torch.Tensor, size: int) -> Image.Image:
    im = Image.fromarray(to_uint8(image))
    im.thumbnail((size, size), Image.LANCZOS)
    return im


def gray_small(image: torch.Tensor, size: int = 256) -> np.ndarray:
    """[H,W,3] float -> [size,size] uint8 grey, for the drift measure."""
    x = image.detach().float().cpu()
    g = (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2])[None, None]
    g = F.interpolate(g, size=(size, size), mode="area")[0, 0]
    return (g.clamp(0, 1).numpy() * 255.0 + 0.5).astype(np.uint8)


def layout_thumb(image: torch.Tensor, size: int = 32) -> np.ndarray:
    """[H,W,3] float -> [size,size,3] uint8 colour thumbnail, for the prompt-variety measure
    (composition and colour, not detail)."""
    x = image.detach().float().cpu().movedim(-1, 0)[None]
    x = F.interpolate(x, size=(size, size), mode="area")[0].movedim(0, -1)
    return (x.clamp(0, 1).numpy() * 255.0 + 0.5).astype(np.uint8)


def _gauss_1d(size=11, sigma=1.5):
    ax = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    k = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    return k / k.sum()


_K1 = _gauss_1d()
_KH = _K1.view(1, 1, 1, -1).repeat(5, 1, 1, 1)  # separable Gaussian, 5 maps at once
_KV = _K1.view(1, 1, -1, 1).repeat(5, 1, 1, 1)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean SSIM of two uint8 grey images (Gaussian window 11, sigma 1.5)."""
    x = torch.from_numpy(a.astype(np.float32))
    y = torch.from_numpy(b.astype(np.float32))
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    maps = torch.stack([x, y, x * x, y * y, x * y])[None]  # [1,5,H,W]
    maps = F.pad(maps, (5, 5, 5, 5), mode="reflect")
    maps = F.conv2d(F.conv2d(maps, _KH, groups=5), _KV, groups=5)[0]
    mx, my, exx, eyy, exy = maps
    sxx, syy, sxy = exx - mx * mx, eyy - my * my, exy - mx * my
    num = (2 * mx * my + c1) * (2 * sxy + c2)
    den = (mx * mx + my * my + c1) * (sxx + syy + c2)
    return float((num / den).mean())


def drift(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - ssim(a, b)


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


# ---------------------------------------------------------------------- scorers


class ArcFaceScorer:
    """Largest face for references, best-matching face for renders (cosine similarity)."""

    name = "ArcFace"

    def __init__(self, models_root: str, det_size: int = 640):
        import onnxruntime as ort  # noqa: F401  (import errors are reported by the caller)
        from insightface.app import FaceAnalysis

        available = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available] or available
        self.app = FaceAnalysis(name="buffalo_l", root=models_root,
                                allowed_modules=["detection", "recognition"], providers=providers)
        self.app.prepare(ctx_id=0 if providers[0].startswith("CUDA") else -1, det_size=(det_size, det_size))
        self.key = f"ArcFace|buffalo_l|{det_size}"  # which embeddings stored renders carry

    def _faces(self, image: torch.Tensor):
        bgr = to_uint8(image)[:, :, ::-1].copy()
        faces = self.app.get(bgr)
        if not faces:  # tight crops defeat the detector; retry once with a margin
            h, w = bgr.shape[:2]
            padded = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
            padded[h // 2 : h // 2 + h, w // 2 : w // 2 + w] = bgr
            faces = self.app.get(padded)
        return faces

    def embed_reference(self, image: torch.Tensor):
        faces = self._faces(image)
        if not faces:
            return None
        f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return np.asarray(f.normed_embedding, dtype=np.float32)

    def embed_render(self, image: torch.Tensor) -> np.ndarray:
        """All face embeddings in a render, shape [n_faces, 512] (n may be 0)."""
        faces = self._faces(image)
        if not faces:
            return np.zeros((0, 512), dtype=np.float32)
        return np.stack([np.asarray(f.normed_embedding, dtype=np.float32) for f in faces])

    @staticmethod
    def similarity(embeds: np.ndarray, centroid: np.ndarray):
        if embeds.shape[0] == 0:
            return math.nan, 0
        return float((embeds @ centroid).max()), int(embeds.shape[0])


def _fingerprint(clip_vision) -> str:
    """Cheap identity of a CLIP Vision model (layer names, shapes and a few weights), so cached render
    embeddings are not reused with a different model."""
    try:
        sd = clip_vision.model.state_dict()
        keys = sorted(sd.keys())
        h = hashlib.sha1(str(len(keys)).encode())
        for k in keys[:2] + keys[-2:]:
            t = sd[k]
            h.update(f"{k}{tuple(t.shape)}{float(t.detach().float().flatten()[:256].sum()):.6g}".encode())
        return h.hexdigest()[:16]
    except Exception:
        return str(id(clip_vision))


class ClipVisionScorer:
    """Whole-image embedding similarity with a ComfyUI CLIP_VISION model."""

    name = "CLIP Vision"

    def __init__(self, clip_vision):
        self.cv = clip_vision
        self.key = f"CLIP Vision|{_fingerprint(clip_vision)}"  # which embeddings stored renders carry

    def _embed(self, image: torch.Tensor) -> np.ndarray:
        out = self.cv.encode_image(image[None])
        emb = getattr(out, "image_embeds", None)
        if emb is None:
            emb = out["last_hidden_state"][:, 0]
        return _normalize(emb[0].detach().float().cpu().numpy().astype(np.float32))

    def embed_reference(self, image: torch.Tensor):
        return self._embed(image)

    def embed_render(self, image: torch.Tensor) -> np.ndarray:
        return self._embed(image)[None]

    @staticmethod
    def similarity(embeds: np.ndarray, centroid: np.ndarray):
        if embeds.shape[0] == 0:
            return math.nan, 0
        return float((embeds @ centroid).max()), 1


def reference_centroid(scorer, images: torch.Tensor):
    """images: [B,H,W,C]. Returns (centroid, n_used, mean self-similarity)."""
    embs = []
    for i in range(images.shape[0]):
        e = scorer.embed_reference(to_rgb(images[i : i + 1])[0])
        if e is not None:
            embs.append(_normalize(e))
    if not embs:
        return None, 0, math.nan
    E = np.stack(embs)
    c = _normalize(E.mean(0))
    return c.astype(np.float32), len(embs), float((E @ c).mean())
