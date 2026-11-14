# PP-OCRv6 Hindi recognition: one-shard fine-tuning pilot

A small, standalone Python project for adapting **PP-OCRv6_medium_rec** to Hindi
word and short-line crops from `himalaya-ai/devanagari_ocr_dataset`. It uses the
traditional PaddleOCR recognizer, not PaddleOCR-VL. No detector is trained.

## What this project does

1. Downloads exactly **one image archive** (batch 001 by default) and the shared
   `devanagari_ocr.json` annotations file.
2. Extracts only basenames starting with `hindi_local`, `iiit_hindi`, or
   `hindidataocr`: the Hindi-named source subsets used in this pilot.
3. Streams annotations using the real `images` + `messages` schema. The older
   `image` + `conversations` schema is also supported.
4. Keeps annotated, valid, Devanagari-containing crops, removes exact decoded
   image duplicates, then creates a sorted/seeded **80/10/10** train/val/test split.
5. Writes PaddleOCR recognition labels and a Unicode codepoint dictionary.
6. Transfers compatible official pretrained parameters, explicitly resets the
   vocabulary-dependent CTC projection and NRTR embedding/output projection, and
   fine-tunes all model parameters using upstream PaddleOCR tools.

This is a new pilot split: sort, seed-42 shuffle, then test/val/train slicing,
applied after filtering this one shard. Do not compare its scores directly with
runs on other shards or different filters. Prefixes identify sources; they are
not a language detector.

## Important scope: words and SHORT lines

At the pinned upstream revision, `PPLCNetV4.forward()` uses
`adaptive_avg_pool2d(x, [1, 40])` in training. Thus, this baseline retains only
labels with at most 40 Unicode codepoints **and** CTC minimum length at most 40
(codepoints plus adjacent repeated-codepoint pairs). Devanagari combining marks
count separately. Raising `max_text_length` or the image width alone would not
remove the fixed 40-step training bottleneck.

The project deliberately leaves the upstream architecture unchanged. Longer
lines are counted in `report.json` as `too_long_for_ctc`, not truncated. Supporting
long lines properly requires a separate sequence-length/pooling change and
validation. Inputs are resized/padded to `[3,48,320]`; very wide lines may still
lose detail. Geometry checks cannot prove a crop is a single line: **inspect
`prepared/crop_preview.jpg` before training**, especially `hindi_local` examples.
Samples narrower than their height are excluded by default; adjust
`--min-aspect-ratio` if that rejects valid words in your data.

## Requirements

- Python **3.12**, the version the pinned dependency set was resolved with.
- Git and internet access to GitHub, Hugging Face, and Paddle's model host.
- Linux and an NVIDIA GPU recommended for training.
  CPU mode is useful for smoke tests. GPU memory needs depend on batch size;
  start at 16 and reduce to 4 or 8 if needed. No VRAM guarantee is implied.
- Allow several GB for one archive (~1.4 GB), the complete annotations (~2.1 GB),
  extracted Hindi images, dependencies, weights, and training checkpoints.
  **One tarball does not eliminate the shared annotation download.**

This repository contains code and a reference upstream YAML; the dataset,
pretrained weights, and the pinned PaddleOCR checkout are downloaded by the
setup and preparation steps. No PyTorch/Transformers stack is needed.

## 1. Install, fetch PaddleOCR, and download pretrained weights

Clone the repository and open a terminal in the project directory.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run.py setup
```

`setup` clones the pinned PaddleOCR source and downloads the official medium
training weights by default. Use `python run.py setup --model-size tiny` or
`python run.py setup --model-size small` to download another variant instead.
Weights are named `weights/PP-OCRv6_<size>_rec_pretrained.pdparams`. It is safe
to repeat setup: an existing checkout is validated and existing weights are
reused. Preparation accepts any of these completed weight downloads, including
when using `prepare.py --offline`.

The same `--model-size` value selects official weights during setup and the
architecture during `configure`. Medium remains the default.

Install **one** PaddlePaddle build. For a conventional NVIDIA CUDA-12.6-compatible
machine, an example is:

```bash
python -m pip install paddlepaddle-gpu==3.2.2 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

If you are using CUDA 13.x, install PaddlePaddle with CUDA 13 support. This also works best for the latest RTX Blackwell GPUs.

```
python -m pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu130/
```

For CPU-only smoke tests instead:

```bash
python -m pip install paddlepaddle==3.2.2
```

Use the [official installer](https://www.paddlepaddle.org.cn/install/quick) for your
OS, driver and GPU. In particular, RTX 50-series/Blackwell may require a newer
CUDA wheel; do not assume the cu126 example fits that hardware. Do not install
both CPU and GPU Paddle packages in the same environment.

Then install the training dependencies resolved for the pinned source and check the framework:

```bash
python -m pip install -r requirements-training.txt
python -c "import paddle; print(paddle.__version__); paddle.utils.run_check()"
python -c "import paddle; print('CUDA build:', paddle.is_compiled_with_cuda()); print('GPUs:', paddle.device.cuda.device_count())"
```

`setup` pins PaddleOCR to `dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf`, reviewed
for its v6 architecture, config, checkpoint loader, label encoders and tools.
It refuses a different revision or dirty checkout rather than silently changing
training behavior. Downloads use a temporary `.part` file before moving completed
weights into place. Dependency ranges in this project and upstream requirements
are not a complete cross-platform lockfile. Save `pip freeze` for each real run.

## 2. Prepare batch 001, Hindi prefixes only

```bash
python prepare.py --batch 1
```

This downloads only `data/images_batch_001.tar.gz` plus `devanagari_ocr.json`.
The downloader resolves one Hugging Face dataset revision for both files and
records it in `prepared/dataset_source.json` (new downloads).
Only images from that selected archive are eligible, even if other images exist
on disk. Other tarballs are never downloaded automatically. Annotation orphans
and records referring to other shards do not become samples.

Open these files before proceeding:

- `prepared/report.json`: counts by split and prefix; rejected labels/images;
  dictionary/label hashes; out-of-vocabulary characters.
- `prepared/crop_preview.jpg`: up to 20 actual eligible crops, with filenames.
- `prepared/train.txt`, `val.txt`, `test.txt`: one image path and transcription
  per line, separated by an actual tab, UTF-8, no header.

Example label rows (the gap is a tab):

```text
devanagari_ocr/hindi_local_123.png	भारत एक देश है।
devanagari_ocr/hindidataocr_456.png	हिन्दी
```

These paths are relative to `--data-dir`, not the prepared directory. Spaces
inside labels are preserved. Text is NFC-normalized and stripped at the edges;
embedded tabs/newlines/NUL and empty text are rejected, not silently rewritten.

The dictionary contains one codepoint per line: the predefined Devanagari block,
printable non-space ASCII, plus other codepoints seen in **training** labels.
Space is added once by `use_space_char=True`. Conjuncts such as `क्ष` are encoded
as component codepoints; they are not a single dictionary item. Held-out labels
never determine the vocabulary: uncovered validation/test characters cause a
clear failure with the offending characters listed, because upstream encoders
would otherwise silently skip unknown characters. Resolve using a justified,
predefined inventory before running a new preparation experiment.

Preparation refuses an existing output folder. To experiment with another shard:

```bash
python prepare.py --batch 2 --output prepared_batch2
```

That still trains on one shard when you use `--prepared prepared_batch2` below.
Do not merge prepared directories. Re-running downloads uses Hugging Face's cache;
interrupted extraction can be repeated with a fresh/nonexistent output folder.

## 3. Initialize the Hindi model and create training config

Choose a model size with `--model-size tiny`, `small`, or `medium` (the default):

| Size | Approximate base parameters | Default experiment |
|---|---:|---|
| `tiny` | 1.5M | `output/hindi_rec_tiny` |
| `small` | 7.7M | `output/hindi_rec_small` |
| `medium` | 34.5M | `output/hindi_rec_medium` |

For example:

```bash
python run.py configure --model-size tiny --batch-size 16 --epochs 30 --lr 0.0001 --workers 2
```

For another prepared directory, add `--prepared prepared_batch2` to this command.
Configuration uses absolute data and dictionary paths, so regenerate it if you
move those assets. `--batch-size`, `--epochs`, `--lr`, `--workers`, and
`--eval-every` are configuration-time settings. `--device cpu` can be used for
CPU tests. Use `--output` to override the default experiment directory, then
pass that same output directory to subsequent commands.

This step uses the matching **official training** `.pdparams` downloaded by
`setup`, builds the new vocabulary-sized CTC + auxiliary NRTR model, and writes:

- `output/<experiment>/hindi_initialized.pdparams`: transferred weights plus
  newly initialized vocabulary layers. All other matching tensors, including
  the NRTR decoder, retain their pretrained values.
- `output/<experiment>/initialization.json`: model size, loaded/reset parameter
  names, checkpoint and dictionary hashes, Paddle version, and upstream commit.
- `output/<experiment>/train_config.yml`: the generated train/eval/export
  configuration.
- `output/<experiment>/checkpoint_config.yml`: portable dynamic-inference
  configuration without absolute training-data paths.
- `output/<experiment>/hindi_dict.txt`: the dictionary copy belonging to this
  experiment and its portable checkpoint bundle.

The base parameter counts above change slightly when vocabulary-dependent
layers are resized for Hindi. Initialization also runs synthetic training-mode
and evaluation-mode forwards on CPU to catch layout/API problems before a GPU
job. It fails on unexpected missing or shape-mismatched parameters outside the
explicitly reset vocabulary layers. This avoids accidentally keeping an output
projection with wrong character IDs, even if the dictionary happens to have the
same size as the original.

The new vocabulary layers have no pretrained Hindi recognition knowledge.
Useful accuracy is an empirical training outcome; one shard may be too small or
unrepresentative. All layers are trainable. This is not LoRA.

## 4. Train, resume, evaluate

Train the medium experiment with defaults:

```bash
python run.py train
```

Or select another experiment explicitly:

```bash
python run.py train --output output/hindi_rec_tiny
```

Defaults: 30 epochs, Adam, learning rate 1e-4, two warmup epochs, batch 16,
full training-set sampling, no concatenation augmentation, seed 42, validation
every 200 steps and checkpoint saving every epoch. The fixed-size SimpleDataSet
pipeline replaces upstream multiscale/concatenation augmentation for this first
pilot. Space and punctuation remain significant in upstream recognition metrics.

### Deterministic training and validation subsets

Use `--num-samples` and optionally `--num-eval-samples` on `train` to run a small
experiment without changing the prepared dataset:

```bash
python run.py train \
  --output output/hindi_rec_tiny \
  --num-samples 10000 \
  --num-eval-samples 1000
```

The selected rows are the first N rows of the corresponding prepared file,
which was already sorted and seed-shuffled during preparation. Selection is
therefore deterministic and reused across epochs. Original `train.txt` and
`val.txt` are never modified; the test split is not subsetted. Counts must be
positive and no greater than the available source rows.

Derived files are stored under the experiment:

```text
output/hindi_rec_tiny/subsets/
├── train_10000.txt
├── train_10000.json
├── val_1000.txt
└── val_1000.json
```

Each JSON file records source and subset SHA-256 hashes, source row count,
requested/actual counts, split, and `first_n_of_prepared_shuffle` selection rule.
The effective configuration is saved as `effective_train_config.yml` in the
experiment. A later standalone `eval` or `test` uses the complete configured
validation or test set, not a training-time validation subset.

For a one-epoch CPU smoke test, configure with `--device cpu --epochs 1
--batch-size 4 --eval-every 10 --workers 0`, then use small subset counts such
as `--num-samples 100 --num-eval-samples 32`. Do not interpret a smoke-test
score as a benchmark.

Resume an interrupted training run using the saved optimizer and state files:

```bash
python run.py train \
  --output output/hindi_rec_tiny \
  --resume output/hindi_rec_tiny/latest
```

Resume requires `latest.pdparams`, `latest.pdopt`, and `latest.states`. It reuses
`effective_train_config.yml`, including the original subsets; do not pass new
sample counts when resuming. Starting from weights alone is a new fine-tuning
run, not full optimizer-state resume.

Validate or run the held-out test split after model selection:

```bash
python run.py eval --output output/hindi_rec_tiny
python run.py test --output output/hindi_rec_tiny
```

Both default to `best_accuracy.pdparams` under the experiment directory. Use
`--checkpoint /path/to/checkpoint` to select another checkpoint prefix or
`.pdparams` file. Keep the test split out of tuning decisions.

Upstream evaluation reports recognition accuracy and normalized edit-distance
metrics. **These are not grapheme CER/codepoint CER/WER.** This project does
not claim compatibility with any external metric protocol; recompute with one
fixed protocol on identical held-out samples for fair comparisons.

## 5. Package a checkpoint, predict, and export

### Portable training checkpoint: no export required

Configuration already creates `checkpoint_config.yml` and `hindi_dict.txt` in
the experiment. After training, copy those sidecars and `initialization.json`
alongside the selected trained checkpoint. For example:

```bash
mkdir -p shared/hindi_rec_tiny
cp output/hindi_rec_tiny/{best_accuracy.pdparams,checkpoint_config.yml,hindi_dict.txt,initialization.json} \
  shared/hindi_rec_tiny/

tar -C shared -czf hindi_rec_tiny_checkpoint.tar.gz hindi_rec_tiny
```

The resulting bundle has this layout:

```text
hindi_rec_tiny/
├── best_accuracy.pdparams
├── checkpoint_config.yml
├── hindi_dict.txt
└── initialization.json
```

Only the `.pdparams` file is needed from the trained checkpoint; `.pdopt` and
`.states` are for resume, not inference. If training did not produce
`best_accuracy`, use another valid checkpoint such as `latest.pdparams` in its
place and pass that filename to `--checkpoint`.

After setup on the same or another machine, run dynamic inference without the
original prepared data or experiment directory:

```bash
python run.py infer \
  --checkpoint /path/to/hindi_rec_tiny/best_accuracy.pdparams \
  --image /absolute/path/to/a/hindi_crop.png
```

The checkpoint's parent directory supplies `checkpoint_config.yml`,
`hindi_dict.txt`, and `initialization.json`. Predictions go to
`predictions.txt`. The bundled dictionary hash must match the initialization
report.

Use the same portable checkpoint in Gradio:

```bash
python gradio_app.py \
  --checkpoint /path/to/hindi_rec_tiny/best_accuracy.pdparams
```

This route invokes dynamic `run.py infer` separately for each uploaded image,
so model loading is repeated. For frequent UI inference, use the exported route
below.

### Exported static inference model

Export the selected checkpoint:

```bash
python run.py export \
  --output output/hindi_rec_tiny \
  --checkpoint output/hindi_rec_tiny/best_accuracy.pdparams
```

This writes `output/hindi_rec_tiny/inference/`, including
`inference.pdiparams`, `inference.yml`, and `inference.json` or
`inference.pdmodel`. The Hindi dictionary is embedded in `inference.yml`.
Package the entire directory:

```bash
tar -C output/hindi_rec_tiny \
  -czf hindi_rec_tiny_inference.tar.gz inference
```

Run the faster, cached exported-model route in Gradio:

```bash
python gradio_app.py --inference-dir /path/to/inference
```

`PPOCRV6_INFER_DIR=/path/to/inference` is also supported. The exported directory
must be writable because upstream creates `ppocr_keys.txt` on first load.
This project uses upstream crop inference; integrating custom weights into a
full detector + recognizer pipeline is a later step.

## Project files

| File | Purpose |
|---|---|
| `prepare.py` | One-shard download/extraction, streamed annotations, checks, dictionary and split |
| `run.py` | Pinned checkout and weight setup, checkpoint adaptation, config generation, upstream tool commands |
| `requirements.txt` | Small data/preparation dependencies; Paddle installed separately |
| `requirements-training.txt` | CPU smoke-tested top-level training dependency versions |
| `vendor_config/` | Reference upstream YAML and its Apache license; actual config read from pinned checkout |
| `tests/test_pipeline.py` | Synthetic schema, shard filtering, deduplication, split and config tests |
| `gradio_app.py` | Gradio UI for portable fine-tuned checkpoints, exported models, or stock pretrained inference |
| `copy_test_images.py` | Copies the held-out test-split crops into `data/infer_data` for manual testing |

## Troubleshooting and limitations

- **No usable samples:** the selected shard may have no eligible Hindi crops;
  inspect counters and choose another single batch. No alternate shard is
  downloaded without you selecting it.
- **GPU OOM:** reconfigure with smaller batch size; GPU training has not been
  measured here. Use a clean output directory for a new experiment.
- **Unknown character error:** inspect `report.json`. Do not ignore missing
  characters; upstream label encoding may silently discard them.
- **Different upstream revision or modified checkout:** use a new checkout;
  update pins only after rechecking model and training compatibility.
- **Slow annotation scan:** all ~2.1 GB is scanned once using a streaming parser;
  only selected-shard candidates and image hashes stay in memory. The scan
  prints its report when complete.
- Prefix and aspect-ratio filtering cannot establish language purity or single-line
  layout. Manually review samples. Exact image deduplication does not eliminate
  near-duplicates or shared writer/document leakage.
- NFC is the preparation normalization; do not assume its metrics are identical
  to results computed with a different normalization.
- Keep each configured experiment in its own `--output` directory. Official
  pretrained downloads are shared under `weights/`; initialized weights,
  configuration, subsets, and checkpoints are experiment-specific.

## References

- [Dataset and upstream source credits](https://huggingface.co/datasets/himalaya-ai/devanagari_ocr_dataset)
- [PP-OCRv6 introduction and customization support](https://www.paddleocr.ai/latest/en/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html)
- [Official recognition training workflow and training checkpoints](https://www.paddleocr.ai/latest/en/version3.x/module_usage/text_recognition.html)
- [Pinned recognition configuration](https://github.com/PaddlePaddle/PaddleOCR/blob/dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf/configs/rec/PP-OCRv6/PP-OCRv6_medium_rec.yml)
- [Pinned backbone: 40-step training pooling](https://github.com/PaddlePaddle/PaddleOCR/blob/dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf/ppocr/modeling/backbones/rec_lcnetv4.py)
- [Pinned vocabulary sizing in training](https://github.com/PaddlePaddle/PaddleOCR/blob/dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf/tools/train.py)

The reference YAML and PaddleOCR retain their Apache-2.0 license. Dataset and
model downloads are not redistributed in this repository; review their
respective terms and credit the underlying source datasets for published work.
This project's code is provided under Apache-2.0 (see LICENSE).

## Update log

### 2026-11-14

- `run.py setup --model-size {tiny,small,medium}` now downloads the selected
  official training weights in addition to cloning and validating the pinned
  PaddleOCR checkout. Medium remains the default, and re-running setup reuses
  the existing checkout and weights.
- `configure --model-size` now selects the matching tiny, small, or medium
  architecture and pretrained checkpoint. Generated configuration, initialized
  weights, and metadata are isolated under `output/hindi_rec_<size>/` or a custom
  `--output` experiment directory.
- `train --num-samples` and `--num-eval-samples` create deterministic, hashed
  subsets without modifying prepared labels. Effective configuration is saved
  for consistent resume behavior, and resume verifies source/subset hashes.
- Tiny checkpoint adaptation resets its `fc2` vocabulary projection as well as
  the projection naming used by the other model variants.
- Configuration also writes a path-independent `checkpoint_config.yml` and
  experiment dictionary. A trained `.pdparams` plus these sidecars and
  `initialization.json` forms a portable checkpoint bundle.
- `run.py infer --checkpoint` and `gradio_app.py --checkpoint` now run portable
  fine-tuned checkpoints without the original prepared dataset or mandatory
  export. `gradio_app.py --inference-dir` selects the cached exported route.
- `prepare.py` now requires a completed setup before any dataset download or
  extraction, including offline preparation.
- `configure` uses the weights downloaded by setup rather than downloading them
  during checkpoint initialization.
