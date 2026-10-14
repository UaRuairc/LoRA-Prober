"""Video and audio for MiniMax H3 renders, and the comparison videos.

Uses PyAV, which ComfyUI itself depends on. Videos are H.264 (yuv420p) + AAC in MP4 with faststart, so they
play in browsers and in ComfyUI's previews. Nothing here touches the model; probe_nodes / compare_node call it.
"""

from __future__ import annotations

import math
import os
import wave

import numpy as np
import torch
from PIL import Image, ImageDraw

from .render import BG, GOOD, MUTED, PANEL, PANEL2, TEXT, ellipsize, font, text_w, wrap

H3_FPS = 24
AAC_RATE = 48000
MAX_CANVAS = (1920, 1080)  # comparison videos are scaled to fit this

# ------------------------------------------------------------------ latents and decoding


def is_nested(x) -> bool:
    return bool(getattr(x, "is_nested", False))


def video_stream(samples):
    """The video part of a latent (MiniMax H3 latents are nested video + audio)."""
    return samples.unbind()[0] if is_nested(samples) else samples


def h3_frame_count(length: int) -> int:
    """MiniMax H3 clip lengths snap up to 17k + 5 frames (5, 22, 39, ...)."""
    n = max(5, int(length))
    while n % 17 != 5:
        n += 1
    return n


def h3_empty_latent(width: int, height: int, length: int):
    """(nested video+audio latent, frame count), exactly what Empty MiniMax H3 AV Latent makes."""
    try:
        from comfy_extras.nodes_minimax_h3 import _empty_av_latent

        lat, frames = _empty_av_latent(width, height, length)
        return lat["samples"], frames
    except ImportError:  # same geometry, written out (ComfyUI without the H3 nodes cannot run H3 anyway)
        import comfy.model_management
        import comfy.nested_tensor

        frames = h3_frame_count(length)
        latent_t = 2 if frames <= 5 else ((frames - 5) // 17) * 5 + 2
        audio_t = round(frames / H3_FPS * 40)
        dev = comfy.model_management.intermediate_device()
        video = torch.zeros([1, 24, latent_t, height // 16, width // 16], device=dev)
        audio = torch.zeros([1, 32, 2, audio_t], device=dev)
        return comfy.nested_tensor.NestedTensor((video, audio)), frames


def to_rgb(images: torch.Tensor) -> torch.Tensor:
    if images.shape[-1] == 4:
        rgb, a = images[..., :3], images[..., 3:4]
        return rgb * a + (1.0 - a)
    if images.shape[-1] == 1:
        return images.repeat(1, 1, 1, 3)
    return images[..., :3]


def decode_frames(vae, samples) -> torch.Tensor:
    """Decoded frames [F, H, W, 3] float 0-1 on the CPU (F = 1 for images)."""
    images = vae.decode(video_stream(samples))
    if images.ndim == 5:
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    return to_rgb(images.float().cpu())


def decode_audio(audio_vae, samples):
    """(waveform [C, L] float32, sample rate) of a nested latent's audio stream, or None."""
    if audio_vae is None or not is_nested(samples):
        return None
    return decode_audio_latent(audio_vae, samples.unbind()[-1])


def decode_audio_latent(audio_vae, latent):
    """(waveform [C, L] float32, sample rate) from an audio latent, as VAE Decode Audio makes it."""
    if audio_vae is None or latent is None:
        return None
    try:
        from comfy_extras.nodes_audio import vae_decode_audio

        out = vae_decode_audio(audio_vae, {"samples": latent})
        wav, rate = out["waveform"], out["sample_rate"]
    except ImportError:  # what VAE Decode Audio does
        wav = audio_vae.decode(latent).movedim(-1, 1)
        std = torch.std(wav, dim=[1, 2], keepdim=True) * 5.0
        std[std < 1.0] = 1.0
        wav = wav / std
        rate = getattr(audio_vae, "audio_sample_rate_output", getattr(audio_vae, "audio_sample_rate", 44100))
    return wav[0].detach().float().cpu().numpy().astype(np.float32), int(rate)


def to_u8(frames: torch.Tensor) -> np.ndarray:
    """[..., 3] float 0-1 -> uint8."""
    return (frames.detach().float().clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)


# ------------------------------------------------------------------------- audio files


def stereo(wav: np.ndarray) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 1:
        wav = wav[None]
    if wav.shape[0] == 1:
        return np.repeat(wav, 2, axis=0)
    return wav[:2]


def fit_seconds(wav: np.ndarray, rate: int, seconds: float) -> np.ndarray:
    """Stereo waveform cut or padded with silence to exactly `seconds`."""
    wav = stereo(wav)
    n = int(round(seconds * rate))
    if wav.shape[1] >= n:
        return np.ascontiguousarray(wav[:, :n])
    return np.concatenate([wav, np.zeros((2, n - wav.shape[1]), np.float32)], axis=1)


def write_wav(path: str, wav: np.ndarray, rate: int) -> None:
    wav = stereo(wav)
    data = (np.clip(wav, -1.0, 1.0) * 32767.0).round().astype("<i2").T.copy()
    tmp = f"{path}.tmp{os.getpid()}"
    with wave.open(tmp, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(int(rate))
        w.writeframes(data.tobytes())
    os.replace(tmp, path)


def read_wav(path: str):
    with wave.open(path, "rb") as w:
        rate, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
        data = np.frombuffer(w.readframes(n), dtype="<i2").astype(np.float32) / 32767.0
    return data.reshape(-1, ch).T.copy(), rate


# ----------------------------------------------------------------------------- MP4


def _even(x: int) -> int:
    return int(x) + (int(x) % 2)


def resample(wav: np.ndarray, rate: int, to_rate: int) -> np.ndarray:
    """Stereo waveform at another sample rate (polyphase when scipy is there, else linear)."""
    wav = stereo(wav)
    if int(rate) == int(to_rate) or wav.shape[1] == 0:
        return wav
    try:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(int(rate), int(to_rate))
        return resample_poly(wav, int(to_rate) // g, int(rate) // g, axis=1).astype(np.float32)
    except ImportError:
        n = int(round(wav.shape[1] * to_rate / rate))
        x = np.linspace(0, wav.shape[1] - 1, n)
        return np.stack([np.interp(x, np.arange(wav.shape[1]), ch) for ch in wav]).astype(np.float32)


class Mp4Writer:
    """Streaming H.264 + AAC writer. Push frames (uint8 RGB, all the same size) and, when the file has
    sound, audio for the same span; the audio goes in at one rate (audio_rate; other rates are converted to
    it) and is encoded at 48 kHz stereo."""

    def __init__(self, path, width, height, fps, with_audio=False, crf=18, preset="medium", audio_rate=None):
        import av

        self.av = av
        self.path = path
        self.w, self.h = _even(width), _even(height)
        self.tmp = f"{path}.tmp{os.getpid()}.mp4"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.c = av.open(self.tmp, mode="w", options={"movflags": "+faststart"})
        self.vs = self.c.add_stream("libx264" if "libx264" in av.codecs_available else "h264", rate=fps)
        self.vs.width, self.vs.height, self.vs.pix_fmt = self.w, self.h, "yuv420p"
        self.vs.options = {"crf": str(crf), "preset": preset}
        self.as_ = self.res = None
        self.audio_rate = int(audio_rate or AAC_RATE)
        if with_audio:
            self.as_ = self.c.add_stream("aac", rate=AAC_RATE, layout="stereo")
            self.res = av.AudioResampler(format="fltp", layout="stereo", rate=AAC_RATE)

    def frame(self, rgb: np.ndarray) -> None:
        if rgb.shape[0] != self.h or rgb.shape[1] != self.w:
            canvas = np.zeros((self.h, self.w, 3), np.uint8)
            canvas[: min(self.h, rgb.shape[0]), : min(self.w, rgb.shape[1])] = rgb[: self.h, : self.w]
            rgb = canvas
        f = self.av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")
        for p in self.vs.encode(f):
            self.c.mux(p)

    def audio(self, wav: np.ndarray, rate: int) -> None:
        if self.as_ is None or wav.shape[-1] == 0:
            return
        wav = resample(wav, rate, self.audio_rate)  # the resampler only takes frames like its first one
        af = self.av.AudioFrame.from_ndarray(np.ascontiguousarray(wav), format="fltp", layout="stereo")
        af.sample_rate = self.audio_rate
        for rf in self.res.resample(af):
            for p in self.as_.encode(rf):
                self.c.mux(p)

    def close(self) -> str:
        if self.as_ is not None:
            for rf in self.res.resample(None):
                for p in self.as_.encode(rf):
                    self.c.mux(p)
        for p in self.vs.encode():
            self.c.mux(p)
        if self.as_ is not None:
            for p in self.as_.encode():
                self.c.mux(p)
        self.c.close()
        os.replace(self.tmp, self.path)
        return self.path

    def abort(self) -> None:
        try:
            self.c.close()
        except Exception:
            pass
        if os.path.exists(self.tmp):
            os.remove(self.tmp)


def write_clip(path, frames, fps, wav=None, rate=None, crf=17) -> str:
    """frames: uint8 [F, H, W, 3]; wav: [C, L] float32 at `rate`, interleaved one frame at a time."""
    frames = np.asarray(frames)
    n, h, w = frames.shape[0], frames.shape[1], frames.shape[2]
    out = Mp4Writer(path, w, h, fps, with_audio=wav is not None, crf=crf, audio_rate=rate)
    try:
        audio = fit_seconds(wav, rate, n / fps) if wav is not None else None
        pos = 0
        for i in range(n):
            out.frame(frames[i])
            if audio is not None:
                end = int(round((i + 1) * rate / fps))
                out.audio(audio[:, pos:end], rate)
                pos = end
        return out.close()
    except BaseException:
        out.abort()
        raise


def remux_audio(path, wav, rate) -> None:
    """Give a clip a new sound track without re-encoding its video."""
    import av

    tmp = f"{path}.tmp{os.getpid()}.mp4"
    with av.open(path) as src:
        vin = src.streams.video[0]
        fps = float(vin.average_rate or vin.base_rate or H3_FPS)
        n = vin.frames or 0
        dur = float(vin.duration * vin.time_base) if vin.duration else (n / fps if n else 0.0)
        dst = av.open(tmp, mode="w", options={"movflags": "+faststart"})
        try:
            vout = dst.add_stream_from_template(vin)
            aout = dst.add_stream("aac", rate=AAC_RATE, layout="stereo")
            res = av.AudioResampler(format="fltp", layout="stereo", rate=AAC_RATE)
            for pkt in src.demux(vin):
                if pkt.dts is None:
                    continue
                pkt.stream = vout
                dst.mux(pkt)
            af = av.AudioFrame.from_ndarray(fit_seconds(wav, rate, dur or wav.shape[-1] / rate), format="fltp",
                                            layout="stereo")
            af.sample_rate = int(rate)
            for rf in list(res.resample(af)) + list(res.resample(None)):
                for p in aout.encode(rf):
                    dst.mux(p)
            for p in aout.encode():
                dst.mux(p)
        finally:
            dst.close()
    os.replace(tmp, path)


def read_audio(path):
    """(stereo waveform [2, L] float32, rate) of a video file's sound track, or None when it has none."""
    import av

    chunks, rate = [], AAC_RATE
    with av.open(path) as c:
        if not c.streams.audio:
            return None
        s = c.streams.audio[0]
        rate = int(s.rate or AAC_RATE)
        res = av.AudioResampler(format="fltp", layout="stereo", rate=rate)
        for f in c.decode(s):
            chunks += [rf.to_ndarray() for rf in res.resample(f)]
        chunks += [rf.to_ndarray() for rf in res.resample(None)]
    if not chunks:
        return None
    return np.concatenate(chunks, axis=1).astype(np.float32), rate


def iter_frames(path, size=None):
    """Decoded frames of a video file as uint8 RGB arrays, resized to size=(w, h) when given."""
    import av

    with av.open(path) as c:
        for f in c.decode(video=0):
            if size is not None and (f.width, f.height) != tuple(size):
                yield np.asarray(f.to_image().resize(tuple(size), Image.LANCZOS))
            else:
                yield f.to_ndarray(format="rgb24")


def frame_count(path) -> int:
    import av

    with av.open(path) as c:
        s = c.streams.video[0]
        if s.frames:
            return int(s.frames)
        return sum(1 for _ in c.decode(video=0))


# ----------------------------------------------------------------------- comparison grid


def best_grid(n, cell_w, cell_h, extra_h, max_w, max_h, gap=10):
    """Columns that give the largest cells for n clips of cell_w x cell_h in the canvas."""
    best = None
    for cols in range(1, n + 1):
        rows = math.ceil(n / cols)
        s = min((max_w - 2 * 16 - (cols - 1) * gap) / (cols * cell_w),
                (max_h - extra_h - rows * 34 - (rows - 1) * gap) / (rows * cell_h), 1.0)
        if best is None or s > best[0] + 1e-9:
            best = (s, cols, rows)
    return best


class VideoGrid:
    """Places clips on one canvas: a title bar, then a grid of labelled cells (an optional row label per row, e.g.
    the training run); one cell can be highlighted (playing with its sound) while the others show a dimmed still.

    cells: dicts with label, row, col (grid position) and optionally badge (a score drawn over the clip's corner)
    and badge_color. rows / cols: grid size."""

    HEAD = 66
    FOOT = 30
    LABEL = 34
    GAP = 10
    MARGIN = 16

    def __init__(self, cells, rows, cols, clip_w, clip_h, title, subtitle, max_size=MAX_CANVAS, row_labels=None):
        self.cells, self.rows, self.cols = cells, rows, cols
        self.row_labels = list(row_labels or [])
        scratch = ImageDraw.Draw(Image.new("RGB", (4, 4)))
        self.gutter = 0
        if self.row_labels:
            fr = font(16, "bold")
            self.gutter = int(min(220, max(text_w(scratch, str(r), fr) for r in self.row_labels) + 24))
        extra = self.HEAD + self.FOOT
        s = min((max_size[0] - 2 * self.MARGIN - self.gutter - (cols - 1) * self.GAP) / (cols * clip_w),
                (max_size[1] - extra - rows * self.LABEL - (rows - 1) * self.GAP) / (rows * clip_h), 1.0)
        self.cw = max(32, int(clip_w * s) // 2 * 2)
        self.ch = max(32, int(clip_h * s) // 2 * 2)
        grid_w = self.gutter + cols * self.cw + (cols - 1) * self.GAP
        self.W = _even(max(2 * self.MARGIN + grid_w, 640))
        self.H = _even(extra + rows * (self.LABEL + self.ch) + (rows - 1) * self.GAP)
        self.ox = (self.W - grid_w) // 2 + self.gutter
        self._base = self._draw_base(title, subtitle)
        self._badges = [self._badge(c) for c in cells]
        self._status = {}

    @property
    def cell_size(self):
        return self.cw, self.ch

    def origin(self, cell):
        x = self.ox + cell["col"] * (self.cw + self.GAP)
        y = self.HEAD + cell["row"] * (self.LABEL + self.ch + self.GAP)
        return x, y

    def _draw_base(self, title, subtitle):
        img = Image.new("RGB", (self.W, self.H), BG)
        d = ImageDraw.Draw(img)
        d.text((self.MARGIN, 10), ellipsize(d, title, font(22, "bold"), self.W - 2 * self.MARGIN), font=font(22, "bold"),
               fill=TEXT)
        d.text((self.MARGIN, 40), ellipsize(d, subtitle, font(15), self.W - 2 * self.MARGIN), font=font(15), fill=MUTED)
        fr = font(16, "bold")
        for r, lab in enumerate(self.row_labels):
            y = self.HEAD + r * (self.LABEL + self.ch + self.GAP) + self.LABEL + self.ch / 2
            lines = wrap(d, str(lab), fr, self.gutter - 16)[:3]
            for k, ln in enumerate(lines):
                d.text((self.ox - self.gutter, y + (k - (len(lines) - 1) / 2) * 20), ln, font=fr, fill=TEXT,
                       anchor="lm")
        for c in self.cells:
            x, y = self.origin(c)
            d.rectangle([x, y, x + self.cw - 1, y + self.LABEL + self.ch - 1], fill=PANEL)
            f = font(15, "bold") if text_w(d, c["label"], font(15, "bold")) <= self.cw - 16 else font(13, "bold")
            d.text((x + 8, y + 8), ellipsize(d, c["label"], f, self.cw - 16), font=f, fill=TEXT)
        return np.asarray(img).copy()

    def _badge(self, c):
        if not c.get("badge"):
            return None
        f = font(14)
        scratch = ImageDraw.Draw(Image.new("RGB", (4, 4)))
        w = int(text_w(scratch, c["badge"], f)) + 10
        img = Image.new("RGB", (w, 21), (0, 0, 0))
        ImageDraw.Draw(img).text((5, 2), c["badge"], font=f, fill=c.get("badge_color", TEXT))
        return np.asarray(img)

    def _status_bar(self, status):
        if status not in self._status:
            img = Image.new("RGB", (self.W, self.FOOT), PANEL2)
            d = ImageDraw.Draw(img)
            d.text((self.MARGIN, 6), ellipsize(d, status, font(15), self.W - 2 * self.MARGIN), font=font(15), fill=TEXT)
            self._status[status] = np.asarray(img)
        return self._status[status]

    def compose(self, frames, active=None, status="", dim=0.35):
        """frames: {cell index: uint8 [ch, cw, 3]}. active: index of the highlighted cell (others dimmed)."""
        out = self._base.copy()
        for i, c in enumerate(self.cells):
            fr = frames.get(i)
            if fr is None:
                continue
            x, y = self.origin(c)
            y += self.LABEL
            if active is not None and i != active:
                fr = (fr.astype(np.float32) * dim).astype(np.uint8)
            out[y : y + self.ch, x : x + self.cw] = fr[: self.ch, : self.cw]
            b = self._badges[i]
            if b is not None and b.shape[1] < self.cw - 8:
                by = y + self.ch - b.shape[0] - 4
                out[by : by + b.shape[0], x + 4 : x + 4 + b.shape[1]] = b
        if active is not None:
            x, y = self.origin(self.cells[active])
            g = np.array(GOOD, np.uint8)
            t = 4
            y1 = y + self.LABEL + self.ch
            out[y : y + t, x : x + self.cw] = g
            out[y1 - t : y1, x : x + self.cw] = g
            out[y : y1, x : x + t] = g
            out[y : y1, x + self.cw - t : x + self.cw] = g
        if status:
            out[self.H - self.FOOT :] = self._status_bar(status)
        return out


def comparison_video(path, cells, clips, rows, cols, title, subtitle, fps=H3_FPS, audio=None, max_size=MAX_CANVAS,
                     row_labels=None):
    """One file: every clip plays at once (no sound), then each clip plays in turn with its own sound while
    the others show a dimmed first frame. Without any sound only the first part is written.

    cells: dicts with label, row, col, optional badge / badge_color. clips: per cell a video path.
    audio: per cell (waveform [C, L], rate) or None. row_labels: one per grid row (e.g. training runs)."""
    first = next(p for p in clips if p)
    probe = next(iter_frames(first))
    grid = VideoGrid(cells, rows, cols, probe.shape[1], probe.shape[0], title, subtitle, max_size, row_labels)
    size = grid.cell_size
    n = max(frame_count(p) for p in clips if p)
    has_audio = audio is not None and any(a is not None for a in audio)
    rate0 = next((a[1] for a in audio if a is not None), AAC_RATE) if has_audio else AAC_RATE
    out = Mp4Writer(path, grid.W, grid.H, fps, with_audio=has_audio, audio_rate=rate0)
    try:
        stills = {}
        gens = {i: iter_frames(p, size) for i, p in enumerate(clips) if p}
        last = {}
        seconds = n / fps
        silence = np.zeros((2, int(round(seconds * rate0))), np.float32)
        pos = 0
        for t in range(n):
            frames = {}
            for i, g in gens.items():
                fr = next(g, None)
                if fr is None:
                    fr = last.get(i)
                last[i] = fr
                frames[i] = fr
                if t == 0:
                    stills[i] = fr
            out.frame(grid.compose(frames, status="All together" + ("  ·  each one with its sound next" if has_audio
                                                                      else "")))
            if has_audio:
                end = int(round((t + 1) * rate0 / fps))
                out.audio(silence[:, pos:end], rate0)
                pos = end
        if has_audio:
            order = [i for i, p in enumerate(clips) if p]
            for k, i in enumerate(order, start=1):
                wav, rate = audio[i] if audio[i] is not None else (np.zeros((2, 1), np.float32), rate0)
                track = fit_seconds(resample(wav, rate, rate0), rate0, seconds)  # whole track: no chunk seams
                rate = rate0
                pos = 0
                frames = dict(stills)
                g = iter_frames(clips[i], size)
                status = f"With sound: {cells[i].get('name') or cells[i]['label']}  ({k} of {len(order)})"
                if audio[i] is None:
                    status += "  ·  no sound for this one"
                for t in range(n):
                    fr = next(g, None)
                    if fr is not None:
                        frames[i] = fr
                    out.frame(grid.compose(frames, active=i, status=status))
                    end = int(round((t + 1) * rate / fps))
                    out.audio(track[:, pos:end], rate)
                    pos = end
        return out.close()
    except BaseException:
        out.abort()
        raise


def flipbook_video(path, pages, fps=12, hold=1.5):
    """Still pages (PIL images, padded to one size) shown one after another, `hold` seconds each."""
    W = _even(max(p.width for p in pages))
    H = _even(max(p.height for p in pages))
    scale = min(1.0, MAX_CANVAS[0] / W, 1440 / H)
    if scale < 1.0:
        W, H = _even(int(W * scale)), _even(int(H * scale))
    out = Mp4Writer(path, W, H, fps, with_audio=False, crf=20)
    try:
        for p in pages:
            canvas = Image.new("RGB", (W, H), BG)
            q = p.convert("RGB")
            if scale < 1.0:
                q = q.resize((max(1, int(q.width * scale)), max(1, int(q.height * scale))), Image.LANCZOS)
            canvas.paste(q, ((W - q.width) // 2, 0))
            arr = np.asarray(canvas)
            for _ in range(max(1, int(round(hold * fps)))):
                out.frame(arr)
        return out.close()
    except BaseException:
        out.abort()
        raise
