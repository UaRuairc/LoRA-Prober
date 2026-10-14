# LoRA Prober

ComfyUI nodes that find which transformer blocks and layer types of a **character LoRA** carry the
identity, score what each part costs in leakage and drift, and write the ai-toolkit config that
trains only that subset. For Qwen-Image 2.1, Krea 2 and MiniMax H3.

It works inside ComfyUI: the same prompts and seeds are rendered with the LoRA split into block
groups and layer types, every render is scored against your reference photos, and the result is a
single diagnostic image plus a contact sheet per seed.

**LoRA Prober Compare** lays any number of LoRA files side by side (the saves of one training run, or
different runs) on the same prompts and seeds, with likeness scores. For MiniMax H3 it renders full
videos with sound and builds a comparison video per prompt: every clip plays at once, then each one
plays in turn with its own sound.

## Supported models

The model is detected from the loaded checkpoint (override with `model_family` if needed).

| | Qwen-Image 2.1 | Krea 2 (Raw or Turbo) | MiniMax H3 (incl. pruned, FastH3) |
|---|---|---|---|
| Blocks | 32 `transformer_blocks` | 28 `blocks` + 4 text-fusion blocks | 50 `blocks` + 2 token-refiner blocks |
| Layers per block | `q k v o` (attention), `mlp_in mlp_out` | `q k v gate o` (attention), `mlp_gate mlp_up mlp_down` | `qkv o` (attention; FastH3 adds `gate_compress`), `mlp_in mlp_out` (`fc1 fc2`), `adaln` |
| Text encoder | Qwen3-VL 8B, Load CLIP type `qwen_image` | Qwen3-VL 4B, Load CLIP type `krea2` | Qwen3-VL 32B, Load CLIP type `minimax` |
| VAE | `qwen_image_2.1_vae` | `qwen_image_vae` | `minimax_h3_video_vae` (+ `minimax_h3_audio_vae` for sound in LoRA Prober Compare) |
| Typical sampling | ~20-25 steps, cfg 1 | Turbo: 8 steps, cfg 1 | 20 steps res_multistep, cfg 1 (8 with the turbo LoRA or FastH3) |
| ai-toolkit arch | `qwen_image_2` | `krea2` | `minimax_h3` |

**Krea 2's text fusion** is a small 4-block transformer that only processes the prompt before it
enters the main blocks, and ai-toolkit LoRAs train it by default. LoRA Prober treats it as its own
part: `keep_txtfusion` (only those layers) and `drop_txtfusion` (everything else) show how much of the
identity comes through the text side, and the suggestion keeps or drops it accordingly. **MiniMax
H3's token refiner** (2 blocks that refine the prompt) is handled the same way
(`keep_token_refiner` / `drop_token_refiner`); `txt` selects it in a selection.

## Nodes

| Node | What it does |
|---|---|
| **LoRA Prober** | Runs the whole test and outputs the diagnostic, contact sheets, a text report and the ai-toolkit YAML. Can run it on up to four checkpoints and compare them. |
| **LoRA Prober Weight Report** | Instant heatmap of where the LoRA changed weights. No rendering. A hint, not a test. |
| **LoRA Prober Compare** | Renders the same prompts and seeds with any number of LoRA files (checkpoints of a run, or several runs) and lays them side by side with likeness scores; comparison videos with sound for MiniMax H3. No block sweep, so it is quick. See [LoRA Prober Compare](#lora-prober-compare). |
| **LoRA Prober Loader (Block Select)** | Applies a LoRA to only chosen blocks / layer types, e.g. to try a subset by hand. Its `selection` field takes the same text as a custom variant (e.g. `-12:mlp`). |

## Install

Clone or download this repository into `ComfyUI/custom_nodes/ComfyUI-LoRA-Prober` and restart
ComfyUI.

Identity scoring for real people uses ArcFace through **insightface**, the same package the
IPAdapter FaceID / InstantID / PuLID nodes use. If you don't have it:

```
pip install insightface onnxruntime        # or onnxruntime-gpu
```

The `buffalo_l` face models download on first use into `ComfyUI/models/insightface/models/`.
For anime, stylised or non-human characters, connect a **CLIP Vision** model instead and set
`identity_metric` to *CLIP Vision (stylised)*.

## Workflow

Six ready-made workflows ship with the pack: in ComfyUI open **Templates → Extensions →
ComfyUI-LoRA-Prober**, or drag the file from `example_workflows/` onto the canvas.

| | Qwen-Image 2.1 | Krea 2 | MiniMax H3 |
|---|---|---|---|
| LoRA Prober | `qwen21_lora_prober` | `krea2_lora_prober` | `minimax_h3_lora_prober` |
| LoRA Prober Compare | `qwen21_lora_compare` | `krea2_lora_compare` | `minimax_h3_lora_compare` |

```
Load Diffusion Model ─────────┐
Load CLIP ────────────────────┼─► LoRA Prober ─► (shows diagnostic + contact sheets itself)
Load VAE ─────────────────────┤                └► Preview Any (ai_toolkit_yaml)
Load Image x N ─► Batch Images┘
```

Connect the base model **without** the LoRA; pick the LoRA in the node. Set `trigger_word`; the
default prompts use `[trigger]` as a placeholder. Edit the control prompts so they describe the same
kind of person as your character (for example "a woman" rather than "a person"). Use the steps and
cfg you normally generate with (the Krea 2 workflow is set for Turbo: 8 steps, cfg 1).

Batch Images resizes every photo to the first one's size (centre crop), so put a large, sharp
photo first.

To compare checkpoints, pick them in `compare_lora_2` to `compare_lora_4` (see
[Comparing checkpoints](#comparing-checkpoints)); leave them on `none` for a normal probe.

To probe with a turbo / lightning LoRA and few steps, pick it in `speed_lora` (see
[Speed LoRA](#speed-lora)). Renders are kept on disk and reused across runs and restarts (see
[Render cache and saved images](#render-cache-and-saved-images)).

## What it renders

* `base` (no LoRA) and `full` (the whole LoRA)
* for each group of `group_size` blocks: `drop_bXX-YY` (LoRA without that group) and
  `keep_bXX-YY` (LoRA with only that group)
* with `layer_variants`: attention only, MLP only, k/v only (each across every block; MiniMax H3
  has no separate k/v), and the text side alone and removed (Krea 2 `keep_txtfusion` /
  `drop_txtfusion`, MiniMax H3 `keep_token_refiner` / `drop_token_refiner`)
* any `custom_variants`, one per line as `name=selection` (see below)
* then up to `test_rounds` suggested subsets: if one misses `target_identity`, the next is widened

### Custom variants

A selection is `blocks:layers`. Blocks are numbers and ranges (`12`, `8-15`, `0-3,12`), `all`, or
`txt` for the text-side blocks (Krea 2 text fusion, MiniMax H3 token refiner). Layers are `attn`,
`mlp`, `kv` (not MiniMax H3), `all` or layer names (Qwen-Image 2.1: `q k v o mlp_in mlp_out`;
Krea 2: `q k v gate o mlp_gate mlp_up mlp_down`; MiniMax H3: `qkv o gate_compress mlp_in mlp_out
adaln`).

```
b12_mlp=12:mlp                 # only block 12's MLP
b12_attn=12:attn               # only block 12's attention
no_b12_mlp=-12:mlp             # the whole LoRA without block 12's MLP
no_b12_attn=-12:attn           # the whole LoRA without block 12's attention
mix=8-15:mlp + 12:attn         # parts joined with +
mid_no12=8-23 except 12:attn   # blocks 8-23 without block 12's attention
mlp_mid_txt=8-15,txt:mlp       # Krea 2: MLP of blocks 8-15 and of the text fusion
```

`example_variants/` has ready-made lists that keep and drop the attention and the MLP of every
block one at a time (`krea2_per_block_attn_mlp.txt`: 116 variants, `qwen21_per_block_attn_mlp.txt`:
128). Paste one into `custom_variants`, set `sweep` to "no block groups", turn `layer_variants` off
and set `test_rounds` to 0. Long runs split each seed's contact sheet into parts of 36 rows, each
part starting with `base` and `full`.

A leading `-` means "everything except", so a `-` variant is the full LoRA with that part
removed: it shows how much the identity *needs* that part. The plain form keeps only that part:
it shows whether that part *carries* the identity on its own. A single block is a small piece of
the LoRA, so expect small differences and run 2 or more seeds before reading much into them;
the ± after "identity kept" is the noise level.

Custom variants are added on top of the rest. Set `sweep` to "no block groups" and turn
`layer_variants` off to run only `base`, `full` and your own.

Each variant costs (identity + control prompts) × seeds renders, the same as that many normal
generations at your settings. For a first pass use 1 seed and a moderate size, then rerun with
`custom_variants` around what you found. Renders are cached on disk: a cancelled or repeated run
picks up where it stopped, also after a restart, as long as the model, LoRA, prompts and sampler
settings are unchanged.

Prompt encoding, the empty latent and the default sampler settings follow ComfyUI's own
text-to-image templates for each model.

## MiniMax H3

MiniMax H3 makes video with sound, so a probe render is a short clip: `video_length` frames (5 by
default, the minimum), scored on its first frame, which has its own latent and is the sharpest. That
keeps a sweep of dozens of renders affordable; full-length videos are for
[LoRA Prober Compare](#lora-prober-compare).
Width and height must be multiples of 32.

**Pruned checkpoints.** ai-toolkit's MiniMax-H3 presets train on Comfy-Org's pruned int8 ConvRot files
(`minimax_h3_fl2va_pruned_int8_convrot`, `minimax_h3_ref2va_pruned_int8_convrot`, FastH3
`fastvideo_fasth3_8step_v2_pruned_int8_convrot`). They keep every block weight but feed the `adaln`
linears an 8-dimensional time table instead of the full time embedding, so an adaln LoRA only fits the
kind of checkpoint it was trained on. LoRA Prober, LoRA Prober Compare and the Block Select loader
check every LoRA layer against the loaded model and leave out the ones that cannot apply, with a
note: adaln from a LoRA trained on the full H3 when you load a pruned file, and `gate_compress` from
a FastH3 LoRA on a checkpoint without it. ComfyUI would otherwise log errors for them and skip them
anyway.

The suggested ai-toolkit config always keeps `adaln_proj` in `ignore_if_contains`, as ai-toolkit's
preset does, and uses its rank 16. ai-toolkit trains the token refiner by default (its block filter
matches `token_refiner.blocks`), so LoRA Prober tests it as its own part.

**Speed.** Use the steps you generate with: 20 with res_multistep, or 8 with the 8-step turbo LoRA in
`speed_lora` (it is then part of `base` too) or with FastH3. `group_size` 10 makes 5 groups of the 50
blocks.

## Reading the diagnostic

* **Verdict**: how much the full LoRA lifts identity, the suggested subset with how it did in its
  own test render, where the identity lives, and (Krea 2, MiniMax H3) how much comes through the text
  side.
* **1 · Weight change**: where the LoRA put its change (brighter = larger); the text-side blocks
  (Krea 2 text fusion, MiniMax H3 token refiner) are the separate columns on the right. Large is not
  the same as important.
* **2 · Block groups**: *necessity* = identity lost when only that group is removed;
  *sufficiency* = identity kept when only that group is active; ticks show leak and drift of the
  group on its own.
* **3 · Every variant**: identity kept (base 0 %, full 100 %), leak (identity resemblance on control
  prompts), drift (how much control images change versus the base model), relative to the full
  LoRA, and prompt variety. ✓ / ✗ mark suggested subsets that met or missed the target.

*Prompt variety* is how different the identity renders of different prompts are from each other
(small colour thumbnails, same seed), relative to the base model (100 %). A LoRA that puts the same
pose, framing, lighting or background into every prompt scores low.

Everything is also saved to `ComfyUI/output/lora_prober/<lora>_<time>/`: `diagnostic.png`, one
`contact_sheet_seed<N>.png` per seed (`_part<K>` files when there are more than 36 variants),
`report.txt`, `ai_toolkit_network.yaml` and `results.json` with every score.

## Speed LoRA

`speed_lora` applies a LoRA to the base model for every render, including the no-LoRA baseline. A
turbo / lightning / distillation LoRA lets you probe in a few steps: set `steps` and `cfg` to what
that LoRA is made for. Everything is then measured on top of it, so `base` means base model + speed
LoRA. It is part of the render cache key, so renders with and without it, or at another strength,
are kept apart. Probe with the speed settings you will actually generate with: layer reliance can
look different with a distilled sampler.

## Render cache and saved images

With `disk_cache` on (the default), every render is saved once in `ComfyUI/output/lora_prober/cache/`
and reused whenever the same image is needed again: within a run (the no-LoRA renders are shared by
every checkpoint), in later runs, and after restarting ComfyUI. Adding variants, prompts, seeds or
another checkpoint only renders what is new.

A render is "the same" when all of these match: the model file (or a sample of its weights), anything
applied to the model before it reaches the node (LoRAs, a sampling shift, other patches), the speed
LoRA and its strength, the VAE, the probed LoRA file, its strength and the exact layers the variant
keeps, the prompt as the text encoder encodes it, the negative prompt when cfg is not 1, seed, steps,
cfg, sampler, scheduler and size. The identity scorer and the reference photos are not part of it:
switching to ArcFace or changing the references re-scores the stored images without rendering.

Each run also checks the cache: it renders the first no-LoRA image fresh and compares it with the
stored one. If they differ, something about the model changed that the key cannot see; that run
then renders everything again, replaces the old renders and says so in the verdict. A run whose
renders are all cached costs that one render plus scoring.

Every run folder gets a `renders/` folder with the images by variant, for example
`renders/keep_b12_mlp/id1_seed1.png`. They are hard links to the cache, so they take no extra space,
and `results.json` names the image behind every score. Each PNG also carries its settings in a
`lora_prober` text field.

LoRA Prober Compare's videos are stored the same way: the clip with its sound (`.mp4`), the sound alone
(`.wav`) and, in the `.npz`, the audio latent, so connecting an audio VAE later adds sound to cached
clips without rendering them again. Its image renders share keys with LoRA Prober's `full` renders, so
what one node rendered the other reuses.

The cache grows by about 1 MB per render at 768 px. Deleting `output/lora_prober/cache/` (or any part
of it) is safe at any time; images linked into run folders stay there, and keep their disk space
until those are deleted too. Turn `disk_cache` off to keep renders in memory only (LoRA Prober
Compare still files its renders in the run folder, since its sheets and videos are made from them).

## Comparing checkpoints

Pick up to three more LoRA files in `compare_lora_2`, `compare_lora_3` and `compare_lora_4`, usually
other saves of the same ai-toolkit run (`my_char_000001000.safetensors`,
`my_char_000002000.safetensors`, the final `my_char.safetensors`). LoRA Prober runs the same test on
each file, with the same prompts, seeds and settings, and adds a comparison. The no-LoRA renders
are shared, so each extra file costs about one probe minus those.

Saves of one run are put in step order. The step comes from the file's metadata (ai-toolkit or
kohya), else from the number at the end of the file name, and the file without one is labelled
*final*. LoRAs from different runs keep the order you gave them and are compared side by side,
not read as a trend.

**The comparison** (the `comparison` output, also shown on the node):

* **Verdict**: likeness at each checkpoint; the earliest checkpoint within 5 % of the best likeness
  (or a note that likeness is still rising); whether leak, drift or prompt variety get worse after
  that point, which are the usual signs of overfitting; where the identity lives at each checkpoint
  and how that moves; how the weight change grows; whether the suggested subsets agree.
* **1 · Trend charts** of the full LoRA at each checkpoint: likeness, leak, drift, prompt variety,
  with the no-LoRA level dashed.
* **2 · Grids, one row per checkpoint**: necessity and sufficiency of each block group, each group's
  share of the weight change, and the identity kept by attention, MLP, k/v (and Krea 2's text
  fusion) alone and by the checkpoint's suggested subset. This is where a later checkpoint leaning
  harder on a few blocks or on one layer type shows up.
* **3 · Table**: each checkpoint's numbers and suggested subset; the green bar marks the pick.

Grid values are relative to each checkpoint's own full LoRA, so compare the pattern rather than
the size; the charts give the absolute values. Rows marked \* had too little identity signal to
trust (very early saves often do) and are greyed out.

With more than one file, the other outputs change too: `diagnostic` is a batch with one diagnostic
per file, `contact_sheet` holds comparison sheets (per seed, the no-LoRA renders and every
checkpoint's full LoRA side by side), `report` starts with the comparison followed by each file's
report, and `ai_toolkit_yaml` is the suggestion for `lora_name`. Files go to
`ComfyUI/output/lora_prober/<lora>_compare_<time>/`: `comparison.png`,
`comparison_sheet_seed<N>.png`, `report.txt`, `comparison.json`, and one numbered folder per
checkpoint with its usual diagnostic, contact sheets, report, YAML and `results.json`.

To keep a comparison quick: 1 seed, `group_size` 8, `test_rounds` 1, or `sweep` "keep-only groups".
For more than four files, several training runs, or videos with sound, use
[LoRA Prober Compare](#lora-prober-compare): it renders each file once per prompt and seed, without
the block sweep.

## LoRA Prober Compare

Renders the same prompts and seeds with every LoRA in a list and lays them side by side. It has no
block sweep: each file costs (prompts + control prompts) × seeds renders.

**The list** (`loras`) takes one entry per line, and each line can match several files:

```
my_char/my_char_000002000.safetensors   one file (the extension is optional)
my_char_v2                              every file whose path contains this: all saves of a run
runs/*_000003000                        a pattern with * or ?
my_char.safetensors | 0.8               another strength
my_char.safetensors | 8-23:mlp          only some layers (LoRA Prober's selection text)
# a comment
```

Saves of one run go in step order (step from the metadata, else the file name; the file without a
step is *final*); runs keep the order of their lines. A pattern that also matches another model's
LoRAs leaves those out with a note. Up to 32 files.

**Layout.** With two or more runs that have several saves each, every run gets a row (the run's name on
the left, the base model at the top left), so equal steps line up. Otherwise the files wrap in reading
order.

**Outputs**

* `comparison`: likeness, leak, drift and prompt variety of every file (one colour and line per run),
  a table with the weight change from the files, and a verdict: the highest likeness, per run the
  earliest save close to its best (within 5 % of the gain over the base model), and warnings when
  no save beats the base model, likeness falls after its peak, prompt variety drops or leak rises
  along a run.
* `sheets`: one per seed, every file (rows) on every prompt (columns), with scores; for videos the
  middle frames.
* Videos, in `output/lora_prober/compare_<file>_<time>/videos/`:
  * **MiniMax H3**: one comparison video per prompt and seed. First every clip plays at once without
    sound, then each clip plays in turn with its own sound while the others wait dimmed. The number on
    each clip is its likeness (the mean over evenly spaced frames). The node previews these videos;
    connect `audio_vae` (minimax_h3_audio_vae) for sound.
  * **Image models**: a flip-book per seed that steps through the files, 1.5 s each.
* `video`: the first comparison video (or flip-book) as a VIDEO, e.g. for Save Video.
* The run folder also has `compare.png`, the sheets, `report.txt`, `compare.json` and every render
  filed by LoRA under `renders/` (hard links to the cache; for H3, clips with their sound).

Without reference photos the node still renders everything and builds the sheets and videos, just
without scores. `length` is the video length for MiniMax H3 (107 frames is about 4.5 s, like
ai-toolkit's samples); image models ignore it. The videos are H.264 MP4, like ComfyUI's Save Video,
and play in any normal browser.

## Important

This is an inference-time ablation of a LoRA that was trained on every layer. It shows where *that*
LoRA keeps the identity. A LoRA trained on fewer layers will push what it learns into the layers it is
given, so retrain with the suggested config and probe the new LoRA to confirm.
