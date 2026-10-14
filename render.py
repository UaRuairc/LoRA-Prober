"""PIL-only drawing of the LoRA Prober diagnostic, the contact sheet and the weight report."""

from __future__ import annotations

import math
import os

from PIL import Image, ImageDraw, ImageFont

BG = (17, 19, 24)
PANEL = (26, 29, 36)
PANEL2 = (33, 37, 46)
GRID = (55, 60, 72)
TEXT = (233, 235, 239)
MUTED = (150, 158, 171)
GOOD = (62, 207, 142)
BAD = (255, 107, 107)
WARN = (255, 196, 0)
NEC = (245, 165, 36)
SUF = (76, 154, 255)
LEAK = (255, 107, 107)
DRIFT = (176, 124, 255)
VARIETY = (64, 196, 204)
IDC = (62, 207, 142)

_VIRIDIS = [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)]

# ------------------------------------------------------------------------ fonts

_FONT_CACHE = {}


def _font_candidates(kind):
    names = {
        "regular": ["DejaVuSans.ttf", "segoeui.ttf", "Arial.ttf", "arial.ttf", "Helvetica.ttc", "LiberationSans-Regular.ttf"],
        "bold": ["DejaVuSans-Bold.ttf", "segoeuib.ttf", "Arial Bold.ttf", "arialbd.ttf", "LiberationSans-Bold.ttf"],
        "mono": ["DejaVuSansMono.ttf", "consola.ttf", "Menlo.ttc", "Courier New.ttf", "cour.ttf", "LiberationMono-Regular.ttf"],
    }[kind]
    dirs = ["/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/TTF", "/usr/share/fonts/dejavu",
            "/usr/share/fonts/truetype/liberation", "C:/Windows/Fonts", "/System/Library/Fonts",
            "/System/Library/Fonts/Supplemental", "/Library/Fonts"]
    try:
        import matplotlib

        dirs.insert(0, os.path.join(matplotlib.get_data_path(), "fonts", "ttf"))
    except Exception:
        pass
    for n in names:
        yield n
        for d in dirs:
            yield os.path.join(d, n)


def font(size, kind="regular"):
    key = (size, kind)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    f = None
    for cand in _font_candidates(kind):
        try:
            f = ImageFont.truetype(cand, size)
            break
        except Exception:
            continue
    if f is None:
        try:
            f = ImageFont.load_default(size=size)
        except TypeError:  # Pillow < 10.1
            f = ImageFont.load_default()
    _FONT_CACHE[key] = f
    return f


def text_w(draw, s, f):
    return draw.textlength(s, font=f)


def wrap(draw, s, f, width):
    lines, cur = [], ""
    for w in s.split(" "):
        trial = (cur + " " + w).strip()
        if text_w(draw, trial, f) <= width:
            cur = trial
            continue
        if cur:
            lines.append(cur)
        cur = w
        while len(cur) > 1 and text_w(draw, cur, f) > width:  # a word wider than the line is broken
            n = len(cur) - 1
            while n > 1 and text_w(draw, cur[:n], f) > width:
                n -= 1
            lines.append(cur[:n])
            cur = cur[n:]
    if cur:
        lines.append(cur)
    return lines


def ellipsize(draw, s, f, width):
    if text_w(draw, s, f) <= width:
        return s
    while s and text_w(draw, s + "…", f) > width:
        s = s[:-1]
    return s + "…"


def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def lerp(a, b, t):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def viridis(t):
    t = max(0.0, min(1.0, t))
    seg = t * (len(_VIRIDIS) - 1)
    i = min(int(seg), len(_VIRIDIS) - 2)
    return lerp(_VIRIDIS[i], _VIRIDIS[i + 1], seg - i)


def fmt_pct(x, na="–"):
    return f"{round(100 * x)}%" if finite(x) else na  # round() so tiny negatives print as 0%, not -0%


# ----------------------------------------------------------------- shared panels


def _section_title(draw, x, y, title, sub=None):
    draw.text((x, y), title, font=font(21, "bold"), fill=TEXT)
    if sub:
        draw.text((x, y + 29), sub, font=font(15), fill=MUTED)
        return y + 56
    return y + 34


def _heatmap(draw, x, y, w, heat, highlight=None):
    """Weight-change heatmap: rows = layer types, columns = blocks, plus per-block share bars.

    heat: modules (row labels), values (per row, per column), block_share (per column),
    optional labels (column labels), side_start / side_label (extra columns after a gap).
    highlight: (column indices, row labels) to outline.
    """
    mods = heat["modules"]
    vals = heat["values"]  # list per module of per-column values (norms), may contain None
    share = heat["block_share"]
    n = len(share)
    labels = heat.get("labels") or [str(b) for b in range(n)]
    side_start = heat.get("side_start")
    gap = 26 if side_start is not None else 0
    label_w = 92
    cw = (w - label_w - gap) / n
    ch = 24
    vmax = max((v for row in vals for v in row if finite(v)), default=1.0) or 1.0
    f_lab, f_small = font(15), font(12)
    hb, hm = (set(highlight[0]), set(highlight[1])) if highlight else (set(), set())

    def cx0(c):
        return x + label_w + c * cw + (gap if side_start is not None and c >= side_start else 0)

    for i, m in enumerate(mods):
        yy = y + i * ch
        draw.text((x, yy + 3), m, font=f_lab, fill=MUTED)
        for c in range(n):
            v = vals[i][c]
            x0 = cx0(c)
            col = viridis(math.sqrt(v / vmax)) if finite(v) and v > 0 else PANEL2
            draw.rectangle([x0 + 1, yy + 1, x0 + cw - 1, yy + ch - 1], fill=col)
    if highlight:
        for i, m in enumerate(mods):
            if m not in hm:
                continue
            for c in range(n):
                if c in hb:
                    x0 = cx0(c)
                    yy = y + i * ch
                    draw.rectangle([x0, yy, x0 + cw, yy + ch], outline=GOOD, width=2)
    # share bars
    by = y + len(mods) * ch + 24
    bh = 70
    smax = max((s for s in share if finite(s)), default=1.0) or 1.0
    draw.text((x, by + bh - 34), "share", font=f_lab, fill=MUTED)
    draw.text((x, by + bh - 16), "of change", font=f_small, fill=MUTED)
    for c in range(n):
        s = share[c] if finite(share[c]) else 0.0
        x0 = cx0(c)
        hh = bh * s / smax
        draw.rectangle([x0 + 3, by + bh - hh, x0 + cw - 3, by + bh], fill=SUF if c not in hb else GOOD)
        if s >= 0.005 and cw >= 30:
            t = f"{100 * s:.0f}" if s >= 0.095 else f"{100 * s:.1f}"
            draw.text((x0 + cw / 2, by + bh - hh - 3), t, font=f_small, fill=MUTED, anchor="md")
    ly = by + bh + 6
    for c in range(n):
        x0 = cx0(c)
        draw.text((x0 + cw / 2, ly), labels[c], font=f_small, fill=GOOD if c in hb else MUTED, anchor="ma")
    end = cx0(side_start) - gap if side_start is not None else x + label_w + n * cw
    draw.text((end, ly + 18), "block", font=f_small, fill=MUTED, anchor="ra")
    if side_start is not None:
        draw.text((x + w, ly + 18), heat.get("side_label", ""), font=f_small, fill=MUTED, anchor="ra")
    return ly + 36


def _legend(draw, x, y, items):
    f = font(14)
    for color, label, shape in items:
        if shape == "bar":
            draw.rectangle([x, y + 4, x + 14, y + 16], fill=color)
        else:
            draw.line([x, y + 10, x + 16, y + 10], fill=color, width=3)
            draw.ellipse([x + 5, y + 6, x + 11, y + 14], fill=color)
        draw.text((x + 22, y + 1), label, font=f, fill=MUTED)
        x += 22 + text_w(draw, label, f) + 22
    return y + 24


def _group_chart(draw, x, y, w, rows, chosen_tags, target):
    h = 230
    left = 44
    f_small, f_lab = font(13), font(14)
    top, bottom = y + 10, y + h
    vals = [r.get(k) for r in rows for k in ("necessity", "sufficiency", "leak", "drift")]
    ymax = min(1.5, max([1.05] + [v + 0.05 for v in vals if finite(v)]))

    def Y(v):
        return bottom - (bottom - top) * max(0.0, min(ymax, v)) / ymax

    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        yy = Y(t)
        draw.line([x + left, yy, x + w, yy], fill=GRID, width=1)
        draw.text((x + left - 6, yy), f"{int(t * 100)}%", font=f_small, fill=MUTED, anchor="rm")
    if finite(target):
        yy = Y(target)
        for xx in range(int(x + left), int(x + w), 12):
            draw.line([xx, yy, min(xx + 6, x + w), yy], fill=GOOD, width=1)
        draw.text((x + left + 6, yy - 3), f"target {int(target * 100)}%", font=f_small, fill=GOOD, anchor="ld")
    n = max(1, len(rows))
    slot = (w - left) / n
    bw = min(34, slot / 3.2)
    for i, r in enumerate(rows):
        cx = x + left + slot * (i + 0.5)
        # necessity bar, sufficiency bar, then a narrow lane with the group's solo leak / drift
        for j, (key, se_key, col) in enumerate((("necessity", "nec_se", NEC), ("sufficiency", "suf_se", SUF))):
            v = r.get(key)
            x0 = cx - 1.5 * bw - 4 + j * (bw + 4)
            if finite(v):
                top_y = Y(max(0.0, v))
                draw.rectangle([x0, top_y, x0 + bw, bottom], fill=col)
                se = r.get(se_key)
                if finite(se) and se > 0:
                    mx = x0 + bw / 2
                    hi_y, lo_y = Y(max(0.0, v + se)), Y(max(0.0, v - se))
                    draw.line([mx, hi_y, mx, lo_y], fill=TEXT, width=1)
                    draw.line([mx - 4, hi_y, mx + 4, hi_y], fill=TEXT, width=1)
                    draw.line([mx - 4, lo_y, mx + 4, lo_y], fill=TEXT, width=1)
                    top_y = min(top_y, hi_y)
                if v > ymax:  # off the scale: arrow at the top, value inside the bar
                    ty = Y(ymax)
                    draw.polygon([(x0 + 2, ty + 9), (x0 + bw - 2, ty + 9), (x0 + bw / 2, ty)], fill=TEXT)
                    draw.text((x0 + bw / 2, ty + 12), f"{100 * v:.0f}", font=f_small, fill=BG, anchor="ma")
                else:
                    draw.text((x0 + bw / 2, top_y - 3), f"{100 * v:.0f}", font=f_small, fill=TEXT, anchor="md")
            else:
                draw.text((x0 + bw / 2, bottom - 4), "–", font=f_small, fill=MUTED, anchor="md")
        lx0 = cx + 0.5 * bw + 10  # solo leak / drift ticks beside the sufficiency bar
        marks = [(r.get(k), col) for k, col in (("leak", LEAK), ("drift", DRIFT)) if finite(r.get(k))]
        if marks:
            draw.line([lx0 + 7, Y(max(max(v, 0.0) for v, _ in marks)), lx0 + 7, bottom], fill=GRID, width=1)
        for v, col in marks:
            yy = Y(min(max(v, 0.0), ymax))
            draw.line([lx0, yy, lx0 + 14, yy], fill=col, width=3)
        tag = r["tag"]
        draw.text((cx, bottom + 6), tag, font=f_lab, fill=GOOD if tag in chosen_tags else MUTED, anchor="ma")
    return bottom + 30


def _bar_cell(draw, x, y, w, h, v, vmax, col, ref=None, text=None):
    draw.rectangle([x, y, x + w, y + h], fill=PANEL2)
    if finite(v):
        ww = w * max(0.0, min(vmax, v)) / vmax
        draw.rectangle([x, y, x + ww, y + h], fill=col)
    if ref is not None:
        rx = x + w * ref / vmax
        draw.line([rx, y - 2, rx, y + h + 2], fill=TEXT, width=1)
    if text:
        draw.text((x + w + 6, y + h / 2), text, font=font(14), fill=TEXT, anchor="lm")


def _variant_table(draw, x, y, w, vrows, target):
    cols = [("variant", 150), ("contents", 300), ("params", 70), ("identity kept", 330), ("leak", 160),
            ("drift", 160), ("variety", 160), ("faces", 70)]
    scale = (w) / sum(c[1] for c in cols)
    cols = [(n, cw * scale) for n, cw in cols]
    f_h, f = font(14, "bold"), font(14)
    xx = x
    for name, cw in cols:
        draw.text((xx + 6, y), name, font=f_h, fill=MUTED)
        xx += cw
    y += 26
    rh = 30
    for i, r in enumerate(vrows):
        if i % 2 == 0:
            draw.rectangle([x, y, x + w, y + rh], fill=PANEL2 if not r.get("highlight") else (30, 48, 42))
        elif r.get("highlight"):
            draw.rectangle([x, y, x + w, y + rh], fill=(30, 48, 42))
        if r.get("highlight"):
            draw.rectangle([x, y, x + 4, y + rh], fill=GOOD)
        xx = x
        cy = y + rh / 2
        name = r["name"]
        if r.get("status") == "pass":
            name += "  ✓"
        elif r.get("status") == "fail":
            name += "  ✗"
        draw.text((xx + 10, cy), ellipsize(draw, name, f, cols[0][1] - 12), font=f, fill=TEXT, anchor="lm")
        xx += cols[0][1]
        draw.text((xx + 6, cy), ellipsize(draw, r["contents"], f, cols[1][1] - 12), font=f, fill=MUTED, anchor="lm")
        xx += cols[1][1]
        draw.text((xx + 6, cy), fmt_pct(r["params"]), font=f, fill=MUTED, anchor="lm")
        xx += cols[2][1]
        kept = r["id_kept"]
        se = r.get("id_kept_se")
        txt = "–" if not finite(kept) else fmt_pct(kept) + (f" ±{round(100 * se)}" if finite(se) else "")
        bar_w = cols[3][1] - 110
        col = GOOD if finite(kept) and kept >= target else (WARN if finite(kept) and kept >= 0.6 else BAD)
        if r["name"] in ("base", "full"):
            col = MUTED
        _bar_cell(draw, xx + 6, cy - 7, bar_w, 14, kept, 1.2, col, ref=1.0, text=txt)
        xx += cols[3][1]
        for key, colr, k in (("leak", LEAK, 4), ("drift", DRIFT, 5), ("variety", VARIETY, 6)):
            v = r.get(key)
            _bar_cell(draw, xx + 6, cy - 6, cols[k][1] - 70, 12, v, 1.5, colr, ref=1.0, text=fmt_pct(v))
            xx += cols[k][1]
        draw.text((xx + 6, cy), r.get("faces", ""), font=f, fill=MUTED, anchor="lm")
        y += rh
    return y + 8


# --------------------------------------------------------------------- diagnostic


def render_diagnostic(d: dict) -> Image.Image:
    W, M = 1640, 34
    inner = W - 2 * M
    # measure on a scratch canvas, draw for real on the second pass
    height = 10  # first pass measures, second pass draws at the measured height
    for _pass in range(2):
        img = Image.new("RGB", (W, height), BG)
        draw = ImageDraw.Draw(img)
        y = M
        draw.text((M, y), "LoRA Prober", font=font(34, "bold"), fill=TEXT)
        tw = text_w(draw, "LoRA Prober", font(34, "bold"))
        draw.text((M + tw + 16, y + 9), d["title"], font=font(22), fill=GOOD)
        y += 50
        for line in d["subtitle"]:
            draw.text((M, y), line, font=font(16), fill=MUTED)
            y += 22
        y += 14

        # verdict + config
        left_w = int(inner * 0.52)
        right_w = inner - left_w - 16
        f_v, f_m = font(17), font(14, "mono")
        v_lines = []
        for kind, text in d["verdict"]:
            first = True
            for ln in wrap(draw, text, f_v, left_w - 56):
                v_lines.append((kind if first else None, ln))
                first = False
        yaml_lines = d["yaml"].splitlines()
        vh = 50 + 25 * len(v_lines) + 16
        yh = 50 + 19 * len(yaml_lines) + 16
        ph = max(vh, yh)
        draw.rounded_rectangle([M, y, M + left_w, y + ph], radius=10, fill=PANEL)
        draw.rounded_rectangle([M + left_w + 16, y, M + inner, y + ph], radius=10, fill=PANEL)
        draw.text((M + 20, y + 16), "Verdict", font=font(21, "bold"), fill=TEXT)
        yy = y + 52
        for kind, ln in v_lines:
            if kind:
                col = {"good": GOOD, "warn": WARN, "bad": BAD}.get(kind, SUF)
                draw.ellipse([M + 22, yy + 6, M + 32, yy + 16], fill=col)
            draw.text((M + 42, yy), ln, font=f_v, fill=TEXT)
            yy += 25
        rx = M + left_w + 36
        draw.text((rx, y + 16), d.get("yaml_title", "ai-toolkit config"), font=font(21, "bold"), fill=TEXT)
        yy = y + 52
        for ln in yaml_lines:
            col = MUTED if ln.lstrip().startswith("#") else TEXT
            draw.text((rx, yy), ellipsize(draw, ln, f_m, right_w - 40), font=f_m, fill=col)
            yy += 19
        y += ph + 26

        # 1 heatmap
        draw.rounded_rectangle([M, y, M + inner, y + 10], radius=0, fill=BG)
        y = _section_title(draw, M, y, "1  Where the LoRA changed weights",
                           "Size of each layer's weight change (brighter = larger); outlined cells = suggested subset. "
                           "A hint, not proof: large is not the same as important.")
        y = _heatmap(draw, M, y, inner, d["heat"], d.get("highlight"))
        y += 10

        # 2 groups
        if d["groups"]:
            y = _section_title(draw, M, y, "2  What each block group does",
                               "Necessity: identity lost when only this group is removed. Sufficiency: identity kept "
                               "when only this group is active. Ticks: leak and drift of the group alone (relative to full).")
            y = _legend(draw, M, y, [(NEC, "necessity", "bar"), (SUF, "sufficiency", "bar"),
                                     (LEAK, "leak (alone)", "tick"), (DRIFT, "drift (alone)", "tick")])
            y = _group_chart(draw, M, y, inner, d["groups"], set(d.get("chosen_tags", [])), d.get("target"))
            y += 6

        # 3 variants
        n = 3 if d["groups"] else 2
        y = _section_title(draw, M, y, f"{n}  Every variant",
                           "Identity kept: share of the full LoRA's identity gain that survives (base 0%, full 100%). "
                           "Leak and drift relative to full; lower is cleaner.")
        y = _variant_table(draw, M, y, inner, d["variants"], d.get("target", 0.9))

        # footer
        y += 8
        f_s = font(14)
        for line in d.get("footer", []):
            for ln in wrap(draw, line, f_s, inner):
                draw.text((M, y), ln, font=f_s, fill=MUTED)
                y += 20
        height = y + M
    return img


# -------------------------------------------------------------------- contact sheet


def render_contact_sheet(rows, columns, cells, thumb=168, title="", legend=()) -> Image.Image:
    """rows: [(name, subtitle)], columns: [(label, kind)], cells[r][c] = (PIL image or None, score or nan, color),
    legend: lines printed under the grid (e.g. the prompt behind each column)."""
    head_w, head_h, pad = 190, 64, 6
    W = max(head_w + len(columns) * (thumb + pad) + pad + 20, 900)
    grid_bottom = 56 + head_h + len(rows) * (thumb + pad)
    scratch = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    legend_lines = [ln for item in legend for ln in wrap(scratch, item, font(14), W - 32)]
    H = grid_bottom + (16 + 20 * len(legend_lines) if legend_lines else 0) + 20
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    ly = grid_bottom + 16
    for ln in legend_lines:
        draw.text((16, ly), ln, font=font(14), fill=MUTED)
        ly += 20
    draw.text((16, 14), title or "Contact sheet", font=font(24, "bold"), fill=TEXT)
    y0 = 56
    f, fs = font(14), font(12)
    for c, (label, kind) in enumerate(columns):
        x = head_w + pad + c * (thumb + pad)
        draw.text((x + thumb / 2, y0 + 8), label, font=font(15, "bold"),
                  fill=IDC if kind == "id" else LEAK, anchor="ma")
        draw.text((x + thumb / 2, y0 + 30), "identity" if kind == "id" else "control", font=fs, fill=MUTED, anchor="ma")
    for r, (name, sub) in enumerate(rows):
        y = y0 + head_h + r * (thumb + pad)
        draw.text((14, y + 10), ellipsize(draw, name, font(16, "bold"), head_w - 20), font=font(16, "bold"), fill=TEXT)
        for i, ln in enumerate(wrap(draw, sub, fs, head_w - 24)[:5]):
            draw.text((14, y + 36 + i * 17), ln, font=fs, fill=MUTED)
        for c in range(len(columns)):
            x = head_w + pad + c * (thumb + pad)
            im, score, col = cells[r][c]
            draw.rectangle([x, y, x + thumb, y + thumb], fill=PANEL)
            if im is not None and (im.width > thumb or im.height > thumb):
                im = im.copy()
                im.thumbnail((thumb, thumb), Image.LANCZOS)
            if im is not None:
                ox = x + (thumb - im.width) // 2
                oy = y + (thumb - im.height) // 2
                img.paste(im, (ox, oy))
            txt = f"{score:.2f}" if finite(score) else "no face"
            tw = text_w(draw, txt, f)
            draw.rectangle([x + 3, y + thumb - 22, x + 11 + tw, y + thumb - 3], fill=(0, 0, 0))
            draw.text((x + 7, y + thumb - 20), txt, font=f, fill=col)
    return img


# --------------------------------------------------------------------- weight report


def render_weight_report(title, subtitle_lines, heat, module_share, group_share) -> Image.Image:
    W, M = 1640, 34
    inner = W - 2 * M
    height = 10  # first pass measures, second pass draws at the measured height
    for _pass in range(2):
        img = Image.new("RGB", (W, height), BG)
        draw = ImageDraw.Draw(img)
        y = M
        draw.text((M, y), "LoRA Weight Report", font=font(34, "bold"), fill=TEXT)
        tw = text_w(draw, "LoRA Weight Report", font(34, "bold"))
        draw.text((M + tw + 16, y + 9), title, font=font(22), fill=GOOD)
        y += 50
        for line in subtitle_lines:
            draw.text((M, y), line, font=font(16), fill=MUTED)
            y += 22
        y += 12
        y = _section_title(draw, M, y, "Where the LoRA changed weights",
                           "Brighter = larger weight change. This shows where the LoRA put its change, not what it does; "
                           "use LoRA Prober to test it.")
        y = _heatmap(draw, M, y, inner, heat)
        y += 8
        f = font(16)
        draw.text((M, y), "By layer type:  " + "   ".join(f"{m} {100 * s:.0f}%" for m, s in module_share), font=f, fill=TEXT)
        y += 26
        line, lines = "By block group:  ", []
        for tag, s in group_share:
            item = f"{tag} {100 * s:.0f}%   "
            if text_w(draw, line + item, f) > inner:
                lines.append(line)
                line = "                 "
            line += item
        lines.append(line)
        for ln in lines:
            draw.text((M, y), ln, font=f, fill=TEXT)
            y += 24
        height = y + M
    return img


# ------------------------------------------------------------------ checkpoint comparison


def _nice_step(span, n=4):
    raw = span / max(1, n)
    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1.0
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            return m * mag
    return 10 * mag


def _fmt_axis(v, kind, dec=2):
    if kind == "pct":
        return f"{round(100 * v)}%"
    return f"{v:.{dec}f}" if abs(v) < 10 else f"{v:.0f}"


def _line_panel(draw, x, y, w, h, p, shorts):
    """One small trend chart: a value per checkpoint (with error bars) and an optional reference line."""
    f_t, f_s, f_v, f_x = font(16, "bold"), font(13), font(12), font(13)
    draw.rounded_rectangle([x, y, x + w, y + h], radius=8, fill=PANEL)
    draw.text((x + 14, y + 10), p["title"], font=f_t, fill=TEXT)
    draw.text((x + 14, y + 32), ellipsize(draw, p.get("sub", ""), f_s, w - 28), font=f_s, fill=MUTED)
    top, bottom = y + 70, y + h - 34
    left, right = x + 56, x + w - 18
    vals = p["values"]
    ref_v, ref_lab = p.get("ref") or (None, None)
    ext = []
    for v, se in vals:
        if finite(v):
            ext.append(v)
            if finite(se):
                ext += [v - se, v + se]
    if finite(ref_v):
        ext.append(ref_v)
    if not [v for v, _ in vals if finite(v)]:
        draw.text(((left + right) / 2, (top + bottom) / 2), "no data", font=f_x, fill=MUTED, anchor="mm")
        return
    lo, hi = min(ext), max(ext)
    if p.get("ymin") is not None:
        lo = min(lo, p["ymin"])
    span = hi - lo if hi - lo > 1e-9 else max(abs(hi), 0.05)
    lo_pad = 0.0 if p.get("ymin") is not None and lo == p["ymin"] else 0.12 * span
    lo, hi = lo - lo_pad, hi + 0.15 * span

    def Y(v):
        return bottom - (bottom - top) * (v - lo) / (hi - lo)

    step = _nice_step(hi - lo)
    dec = 2 if step >= 0.01 - 1e-12 else 3  # enough decimals to tell the ticks (and values) apart
    t = math.ceil(lo / step) * step
    while t <= hi + 1e-9:
        yy = Y(t)
        draw.line([left, yy, right, yy], fill=GRID, width=1)
        draw.text((left - 6, yy), _fmt_axis(t, p.get("fmt"), dec), font=f_v, fill=MUTED, anchor="rm")
        t += step
    if finite(ref_v):
        yy = Y(ref_v)
        for xx in range(int(left), int(right), 10):
            draw.line([xx, yy, min(xx + 5, right), yy], fill=MUTED, width=1)
        if ref_lab:
            draw.text((right, yy - 3), ref_lab, font=f_v, fill=MUTED, anchor="rd")
    n = len(vals)
    xs = [left + (right - left) * (i + 0.5) / n for i in range(n)]
    pts = [(xs[i], Y(v)) for i, (v, _) in enumerate(vals) if finite(v)]
    if len(pts) > 1 and p.get("connect", True):
        draw.line(pts, fill=SUF, width=3)
    for i, (v, se) in enumerate(vals):
        cx = xs[i]
        if finite(v):
            if finite(se) and se > 0:
                y0, y1 = Y(v + se), Y(v - se)
                draw.line([cx, y0, cx, y1], fill=TEXT, width=1)
                draw.line([cx - 4, y0, cx + 4, y0], fill=TEXT, width=1)
                draw.line([cx - 4, y1, cx + 4, y1], fill=TEXT, width=1)
            cy = Y(v)
            draw.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=SUF, outline=BG)
            ty = (Y(v + se) if finite(se) else cy) - 6
            draw.text((cx, ty), _fmt_axis(v, p.get("fmt"), dec), font=f_v, fill=TEXT, anchor="md")
        draw.text((cx, bottom + 8), ellipsize(draw, shorts[i], f_x, (right - left) / n - 4), font=f_x,
                  fill=MUTED, anchor="ma")


def _cell_colour(t):
    return lerp(PANEL2, SUF, max(0.0, min(1.0, t)))


def _grid_size(draw, g, row_labels, w_avail, min_cw=40, max_cw=86):
    f_r = font(13)
    label_w = min(170, max([text_w(draw, s, f_r) for s in row_labels] + [60]) + 22)
    n = max(1, len(g["cols"]))
    cw = max(min_cw, min(max_cw, (w_avail - label_w) / n))
    return label_w, cw, label_w + n * cw


def _grid(draw, x, y, w, g, row_labels, weak=()):
    """weak: per row, True when that checkpoint's identity signal is too weak to trust (greyed out).
    The weight-change grid comes from the files, so it passes no weak rows."""
    """Rows = checkpoints, columns = block groups / subsets, cells coloured by value (0..vmax)."""
    f_t, f_s, f_h, f_c, f_r = font(16, "bold"), font(13), font(12), font(13), font(13)
    draw.text((x, y), g["title"], font=f_t, fill=TEXT)
    sub_lines = wrap(draw, g.get("sub", ""), f_s, w)[:2]
    for i, ln in enumerate(sub_lines):
        draw.text((x, y + 22 + 17 * i), ln, font=f_s, fill=MUTED)
    y += 26 + 17 * len(sub_lines) + 6
    cols, vals = g["cols"], g["values"]
    vmax = g.get("vmax") or 1.0
    label_w, cw, _ = _grid_size(draw, g, row_labels, w)
    # header: up to two lines per column label
    for j, c in enumerate(cols):
        cx = x + label_w + j * cw + cw / 2
        lines = wrap(draw, c.replace("_", "_​"), f_h, cw - 4)[:2]
        if len(wrap(draw, c, f_h, cw - 4)) > 2:
            lines = [lines[0], ellipsize(draw, " ".join(wrap(draw, c, f_h, cw - 4)[1:]), f_h, cw - 4)]
        for k, ln in enumerate(lines):
            draw.text((cx, y + k * 15), ln.replace("​", ""), font=f_h, fill=MUTED, anchor="ma")
    y += 36
    ch = 28
    for i, (lab, row) in enumerate(zip(row_labels, vals)):
        yy = y + i * ch
        is_weak = i < len(weak) and weak[i]
        draw.text((x, yy + ch / 2), ellipsize(draw, lab + (" *" if is_weak else ""), f_r, label_w - 10), font=f_r,
                  fill=MUTED if is_weak else TEXT, anchor="lm")
        for j, v in enumerate(row):
            x0 = x + label_w + j * cw
            if finite(v) and is_weak:  # too little identity signal: show the number, not a colour
                draw.rectangle([x0 + 1, yy + 1, x0 + cw - 1, yy + ch - 1], fill=PANEL)
                draw.text((x0 + cw / 2, yy + ch / 2), fmt_pct(v), font=f_c, fill=MUTED, anchor="mm")
            elif finite(v):
                t = v / vmax if vmax > 0 else 0.0
                col = _cell_colour(t)
                draw.rectangle([x0 + 1, yy + 1, x0 + cw - 1, yy + ch - 1], fill=col)
                lum = 0.299 * col[0] + 0.587 * col[1] + 0.114 * col[2]
                draw.text((x0 + cw / 2, yy + ch / 2), fmt_pct(v), font=f_c,
                          fill=BG if lum > 140 else TEXT, anchor="mm")
            else:
                draw.rectangle([x0 + 1, yy + 1, x0 + cw - 1, yy + ch - 1], fill=PANEL)
                draw.text((x0 + cw / 2, yy + ch / 2), "–", font=f_c, fill=MUTED, anchor="mm")
    return y + len(vals) * ch + 8


def _cmp_table(draw, x, y, w, rows, target, first_col="checkpoint"):
    cols = [(first_col, 300), ("likeness", 150), ("leak", 90), ("drift", 90), ("variety", 90),
            ("weight change", 130), ("suggested subset", 560), ("its test", 90), ("size", 70)]
    scale = w / sum(c[1] for c in cols)
    cols = [(n, cw * scale) for n, cw in cols]
    f_h, f, f_s = font(14, "bold"), font(14), font(12)
    xx = x
    for name, cw in cols:
        draw.text((xx + 6, y), name, font=f_h, fill=MUTED)
        xx += cw
    y += 26
    rh = 46
    for i, r in enumerate(rows):
        if i % 2 == 0:
            draw.rectangle([x, y, x + w, y + rh], fill=PANEL2)
        if r.get("highlight"):
            draw.rectangle([x, y, x + 4, y + rh], fill=GOOD)
        cy = y + rh / 2
        xx = x
        draw.text((xx + 10, cy - 10), ellipsize(draw, r["label"], f, cols[0][1] - 16), font=f,
                  fill=MUTED if not r["reliable"] else TEXT, anchor="lm")
        draw.text((xx + 10, cy + 10), ellipsize(draw, r["name"], f_s, cols[0][1] - 16), font=f_s, fill=MUTED,
                  anchor="lm")
        xx += cols[0][1]
        lik = "–" if not finite(r["id_sim"]) else f"{r['id_sim']:.3f}" + (f" ±{r['id_se']:.3f}" if finite(r["id_se"]) else "")
        draw.text((xx + 6, cy), lik, font=f, fill=TEXT, anchor="lm")
        xx += cols[1][1]
        for key, k in (("leak", 2), ("drift", 3)):
            draw.text((xx + 6, cy), f"{r[key]:.3f}" if finite(r[key]) else "–", font=f, fill=TEXT, anchor="lm")
            xx += cols[k][1]
        draw.text((xx + 6, cy), fmt_pct(r["variety"]), font=f, fill=TEXT, anchor="lm")
        xx += cols[4][1]
        _bar_cell(draw, xx + 6, cy - 6, cols[5][1] - 64, 12, r["change"], 1.0, SUF, text=fmt_pct(r["change"]))
        xx += cols[5][1]
        lines = wrap(draw, r["what"] or "–", f, cols[6][1] - 12)[:2]
        for k, ln in enumerate(lines):
            draw.text((xx + 6, cy + (k - (len(lines) - 1) / 2) * 18), ln, font=f, fill=TEXT, anchor="lm")
        xx += cols[6][1]
        kept = r["kept"]
        mark = "" if r["passed"] is None else ("  ✓" if r["passed"] else "  ✗")
        col = GOOD if r["passed"] else (WARN if r["passed"] is False else MUTED)
        draw.text((xx + 6, cy), (fmt_pct(kept) + mark) if finite(kept) else "–", font=f, fill=col, anchor="lm")
        xx += cols[7][1]
        size = "–" if not r.get("reliable", True) else fmt_pct(r["size"]) if r.get("restricted") else "100%"
        draw.text((xx + 6, cy), size, font=f, fill=MUTED, anchor="lm")
        y += rh
    return y + 8


RUN_COLORS = [SUF, NEC, VARIETY, DRIFT, LEAK, GOOD, WARN, (200, 200, 200)]


def _series_panel(draw, x, y, w, h, p, shorts, groups):
    """A value per LoRA (error bars), one colour and line per training run, a dashed base-model line."""
    f_t, f_s, f_v, f_x = font(16, "bold"), font(13), font(12), font(12)
    draw.rounded_rectangle([x, y, x + w, y + h], radius=8, fill=PANEL)
    draw.text((x + 14, y + 10), p["title"], font=f_t, fill=TEXT)
    draw.text((x + 14, y + 32), ellipsize(draw, p.get("sub", ""), f_s, w - 28), font=f_s, fill=MUTED)
    top, bottom = y + 70, y + h - 40
    left, right = x + 56, x + w - 18
    vals = p["values"]
    ref_v, ref_lab = p.get("ref") or (None, None)
    ext = [v for v, _ in vals if finite(v)] + [v + s for v, s in vals if finite(v) and finite(s)] + \
          [v - s for v, s in vals if finite(v) and finite(s)] + ([ref_v] if finite(ref_v) else [])
    if not [v for v, _ in vals if finite(v)]:
        draw.text(((left + right) / 2, (top + bottom) / 2), "no data", font=f_x, fill=MUTED, anchor="mm")
        return
    lo, hi = min(ext), max(ext)
    if p.get("ymin") is not None:
        lo = min(lo, p["ymin"])
    span = hi - lo if hi - lo > 1e-9 else max(abs(hi), 0.05)
    lo_pad = 0.0 if p.get("ymin") is not None and lo == p["ymin"] else 0.12 * span
    lo, hi = lo - lo_pad, hi + 0.15 * span

    def Y(v):
        return bottom - (bottom - top) * (v - lo) / (hi - lo)

    step = _nice_step(hi - lo)
    dec = 2 if step >= 0.01 - 1e-12 else 3
    t = math.ceil(lo / step) * step
    while t <= hi + 1e-9:
        yy = Y(t)
        draw.line([left, yy, right, yy], fill=GRID, width=1)
        draw.text((left - 6, yy), _fmt_axis(t, p.get("fmt"), dec), font=f_v, fill=MUTED, anchor="rm")
        t += step
    if finite(ref_v):
        yy = Y(ref_v)
        for xx in range(int(left), int(right), 10):
            draw.line([xx, yy, min(xx + 5, right), yy], fill=MUTED, width=1)
        if ref_lab:
            draw.text((right, yy - 3), ref_lab, font=f_v, fill=MUTED, anchor="rd")
    n = len(vals)
    xs = [left + (right - left) * (i + 0.5) / max(1, n) for i in range(n)]
    for g in sorted(set(groups)):
        pts = [(xs[i], Y(vals[i][0])) for i in range(n) if groups[i] == g and finite(vals[i][0])]
        if len(pts) > 1:
            draw.line(pts, fill=RUN_COLORS[g % len(RUN_COLORS)], width=3)
    for i, (v, se) in enumerate(vals):
        cx, col = xs[i], RUN_COLORS[groups[i] % len(RUN_COLORS)]
        if finite(v):
            if finite(se) and se > 0:
                y0, y1 = Y(v + se), Y(v - se)
                draw.line([cx, y0, cx, y1], fill=TEXT, width=1)
                draw.line([cx - 4, y0, cx + 4, y0], fill=TEXT, width=1)
                draw.line([cx - 4, y1, cx + 4, y1], fill=TEXT, width=1)
            cy = Y(v)
            draw.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=col, outline=BG)
            if n <= 12:
                ty = (Y(v + se) if finite(se) else cy) - 6
                draw.text((cx, ty), _fmt_axis(v, p.get("fmt"), dec), font=f_v, fill=TEXT, anchor="md")
        lab = ellipsize(draw, shorts[i], f_x, max(12, (right - left) / max(1, n) - 4))
        draw.text((cx, bottom + 8), lab, font=f_x, fill=MUTED, anchor="ma")


def _lora_table(draw, x, y, w, rows):
    cols = [("LoRA", 330), ("strength · layers", 230), ("likeness", 150), ("leak", 90), ("drift", 90),
            ("variety", 90), ("weight change", 150)]
    scale = w / sum(c[1] for c in cols)
    cols = [(n, cw * scale) for n, cw in cols]
    f_h, f, f_s = font(14, "bold"), font(14), font(12)
    xx = x
    for name, cw in cols:
        draw.text((xx + 6, y), name, font=f_h, fill=MUTED)
        xx += cw
    y += 26
    rh = 44
    for i, r in enumerate(rows):
        if i % 2 == 0:
            draw.rectangle([x, y, x + w, y + rh], fill=PANEL2)
        if r.get("highlight"):
            draw.rectangle([x, y, x + 4, y + rh], fill=GOOD)
        if r.get("color") is not None:
            draw.ellipse([x + 10, y + rh / 2 - 5, x + 20, y + rh / 2 + 5], fill=r["color"])
        cy = y + rh / 2
        xx = x + 16
        draw.text((xx + 10, cy - 10), ellipsize(draw, r["label"], f, cols[0][1] - 30), font=f, fill=TEXT, anchor="lm")
        draw.text((xx + 10, cy + 10), ellipsize(draw, r["name"], f_s, cols[0][1] - 30), font=f_s, fill=MUTED,
                  anchor="lm")
        xx = x + cols[0][1]
        draw.text((xx + 6, cy), ellipsize(draw, r["how"], f, cols[1][1] - 12), font=f, fill=MUTED, anchor="lm")
        xx += cols[1][1]
        lik = "–" if not finite(r["id_sim"]) else f"{r['id_sim']:.3f}" + (f" ±{r['id_se']:.3f}" if finite(r["id_se"]) else "")
        draw.text((xx + 6, cy), lik, font=f, fill=TEXT, anchor="lm")
        xx += cols[2][1]
        for key, k in (("leak", 3), ("drift", 4)):
            draw.text((xx + 6, cy), f"{r[key]:.3f}" if finite(r[key]) else "–", font=f, fill=TEXT, anchor="lm")
            xx += cols[k][1]
        draw.text((xx + 6, cy), fmt_pct(r["variety"]), font=f, fill=TEXT, anchor="lm")
        xx += cols[5][1]
        if r.get("change") is None:
            draw.text((xx + 6, cy), "–", font=f, fill=MUTED, anchor="lm")
        else:
            _bar_cell(draw, xx + 6, cy - 6, cols[6][1] - 64, 12, r["change"], 1.0, SUF, text=fmt_pct(r["change"]))
        y += rh
    return y + 8


def render_lora_compare(d: dict) -> Image.Image:
    """d: title, subtitle, verdict, charts (title, sub, values, ref, fmt, ymin), shorts, groups, runs (names),
    table rows, footer."""
    W, M = 1640, 34
    inner = W - 2 * M
    height = 10
    for _pass in range(2):
        img = Image.new("RGB", (W, height), BG)
        draw = ImageDraw.Draw(img)
        y = M
        head = "LoRA Compare"
        draw.text((M, y), head, font=font(34, "bold"), fill=TEXT)
        tw = text_w(draw, head, font(34, "bold"))
        draw.text((M + tw + 16, y + 9), ellipsize(draw, d["title"], font(22), inner - tw - 16), font=font(22), fill=GOOD)
        y += 50
        for line in d["subtitle"]:
            draw.text((M, y), ellipsize(draw, line, font(16), inner), font=font(16), fill=MUTED)
            y += 22
        y += 14
        f_v = font(17)
        v_lines = []
        for kind, text in d["verdict"]:
            first = True
            for ln in wrap(draw, text, f_v, inner - 70):
                v_lines.append((kind if first else None, ln))
                first = False
        ph = 50 + 25 * len(v_lines) + 14
        draw.rounded_rectangle([M, y, M + inner, y + ph], radius=10, fill=PANEL)
        draw.text((M + 20, y + 16), "Verdict", font=font(21, "bold"), fill=TEXT)
        yy = y + 52
        for kind, ln in v_lines:
            if kind:
                col = {"good": GOOD, "warn": WARN, "bad": BAD}.get(kind, SUF)
                draw.ellipse([M + 22, yy + 6, M + 32, yy + 16], fill=col)
            draw.text((M + 42, yy), ln, font=f_v, fill=TEXT)
            yy += 25
        y += ph + 26
        charts = [c for c in d["charts"] if any(finite(v) for v, _ in c["values"])]
        if charts:
            y = _section_title(draw, M, y, "1  Each LoRA",
                               "Mean over all renders of each LoRA; bars are ± one standard error. Dashed: the base "
                               "model without a LoRA. One colour and line per training run.")
            if len(d.get("runs", [])) > 1:
                y = _legend(draw, M, y, [(RUN_COLORS[i % len(RUN_COLORS)], r, "tick") for i, r in enumerate(d["runs"])])
            per_row = 2 if len(d["shorts"]) > 8 else min(4, len(charts))
            gap = 18
            pw = (inner - gap * (per_row - 1)) / per_row
            for i, p in enumerate(charts):
                cx = M + (i % per_row) * (pw + gap)
                _series_panel(draw, cx, y, pw, 250, p, d["shorts"], d["groups"])
                if i % per_row == per_row - 1 or i == len(charts) - 1:
                    y += 250 + 22
            y += 8
        y = _section_title(draw, M, y, ("2" if charts else "1") + "  Table",
                           "Likeness / leak: similarity to the references on identity / control prompts. Drift: control "
                           "renders against the base model. Variety: base = 100%. Weight change: from the files.")
        y = _lora_table(draw, M, y, inner, d["table"])
        y += 8
        f_s = font(14)
        for line in d.get("footer", []):
            for ln in wrap(draw, line, f_s, inner):
                draw.text((M, y), ln, font=f_s, fill=MUTED)
                y += 20
        height = y + M
    return img


def render_comparison(d: dict) -> Image.Image:
    """d: title, subtitle, verdict, shorts, labels, weak, charts, grids, table, footer, target."""
    W, M = 1640, 34
    inner = W - 2 * M
    height = 10
    for _pass in range(2):
        img = Image.new("RGB", (W, height), BG)
        draw = ImageDraw.Draw(img)
        y = M
        head = d.get("head") or "Checkpoint Comparison"
        draw.text((M, y), head, font=font(34, "bold"), fill=TEXT)
        tw = text_w(draw, head, font(34, "bold"))
        draw.text((M + tw + 16, y + 9), ellipsize(draw, d["title"], font(22), inner - tw - 16), font=font(22), fill=GOOD)
        y += 50
        for line in d["subtitle"]:
            draw.text((M, y), line, font=font(16), fill=MUTED)
            y += 22
        y += 14

        # verdict
        f_v = font(17)
        v_lines = []
        for kind, text in d["verdict"]:
            first = True
            for ln in wrap(draw, text, f_v, inner - 70):
                v_lines.append((kind if first else None, ln))
                first = False
        ph = 50 + 25 * len(v_lines) + 14
        draw.rounded_rectangle([M, y, M + inner, y + ph], radius=10, fill=PANEL)
        draw.text((M + 20, y + 16), "Verdict", font=font(21, "bold"), fill=TEXT)
        yy = y + 52
        for kind, ln in v_lines:
            if kind:
                col = {"good": GOOD, "warn": WARN, "bad": BAD}.get(kind, SUF)
                draw.ellipse([M + 22, yy + 6, M + 32, yy + 16], fill=col)
            draw.text((M + 42, yy), ln, font=f_v, fill=TEXT)
            yy += 25
        y += ph + 26

        # 1 trends
        y = _section_title(draw, M, y, d.get("section1") or "1  The full LoRA at each checkpoint",
                           "Mean over all identity (likeness) or control (leak, drift) renders; bars are ± one "
                           "standard error. Dashed: the base model without the LoRA.")
        charts = d["charts"]
        gap = 18
        pw = (inner - gap * (len(charts) - 1)) / max(1, len(charts))
        for i, p in enumerate(charts):
            _line_panel(draw, M + i * (pw + gap), y, pw, 250, p, d["shorts"])
        y += 250 + 30

        # 2 grids
        y = _section_title(draw, M, y, d.get("section2") or "2  What carries the identity at each checkpoint",
                           "Each row is one LoRA file; * = identity signal too weak to trust (greyed out). "
                           "Values are relative to that file's own full LoRA.")
        grids = []
        for g in d["grids"]:
            if not g["cols"] or not any(finite(v) for row in g["values"] for v in row):
                continue
            label_w, _, _ = _grid_size(draw, g, d["labels"], inner)
            fit = max(1, int((inner - label_w) // 40))  # columns that fit at the narrowest cell width
            n = len(g["cols"])
            if n <= fit:
                grids.append(g)
                continue
            chunks = math.ceil(n / fit)
            per = math.ceil(n / chunks)
            for c in range(chunks):  # too many columns for one row: continue the grid below
                a, b = c * per, min(n, (c + 1) * per)
                grids.append(dict(g, title=g["title"] + ("" if c == 0 else " (continued)"),
                                  sub=g.get("sub", "") if c == 0 else "",
                                  cols=g["cols"][a:b], values=[row[a:b] for row in g["values"]]))
        half = (inner - 30) / 2
        i = 0
        while i < len(grids):
            g = grids[i]
            _, _, need = _grid_size(draw, g, d["labels"], half)
            nxt = grids[i + 1] if i + 1 < len(grids) else None
            fits_pair = need <= half and nxt is not None and _grid_size(draw, nxt, d["labels"], half)[2] <= half
            if fits_pair:
                y1 = _grid(draw, M, y, half, g, d["labels"], () if g.get("files") else d["weak"])
                y2 = _grid(draw, M + half + 30, y, half, nxt, d["labels"], () if nxt.get("files") else d["weak"])
                y = max(y1, y2) + 18
                i += 2
            else:
                y = _grid(draw, M, y, inner, g, d["labels"], () if g.get("files") else d["weak"]) + 18
                i += 1
        if not grids:
            draw.text((M, y), "No block-group or layer variants were rendered.", font=font(15), fill=MUTED)
            y += 30

        # 3 table
        y = _section_title(draw, M, y, d.get("section3") or "3  Each checkpoint",
                           "Weight change: size of the LoRA's change relative to the largest file. "
                           "Its test: identity the suggested subset kept in its own test render.")
        y = _cmp_table(draw, M, y, inner, d["table"], d.get("target", 0.9), d.get("first_col") or "checkpoint")

        y += 8
        f_s = font(14)
        for line in d.get("footer", []):
            for ln in wrap(draw, line, f_s, inner):
                draw.text((M, y), ln, font=f_s, fill=MUTED)
                y += 20
        height = y + M
    return img
