# Validation at packaging

Checked on Linux, Python 3.12.14, CPU PaddlePaddle 3.2.2, against PaddleOCR commit
`dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf`.

- Inspected the user's repository at
  `fd30e6e41f0a9774319cb84f2ef7910c667f9334`: actual annotation schema, download
  paths, flattening extraction, split convention, and Hindi prefixes.
- Three automated synthetic tests passed: streamed compact JSON and JSONL;
  annotation extraction, CTC limit/repeated-character handling, duplicate-image
  removal and deterministic/disjoint split; configuration of both decoder heads
  and full training-set sampling. CLI help and Python compilation passed.
- Downloaded the real official v6-medium recognition training checkpoint.
  Transferred **437** matching tensors; explicitly initialized these four tensors:
  `head.ctc_head.fc.weight`, `head.ctc_head.fc.bias`,
  `head.gtc_head.embedding.embedding.weight`,
  `head.gtc_head.tgt_word_prj.weight`.
- Built a small synthetic Hindi dictionary and ran real training/evaluation-mode
  forwards. Evaluation output was `[2,40,5]` for that test dictionary.
- Ran one CPU epoch with four synthetic images and arbitrary Hindi labels:
  two optimizer steps, CTC + NRTR losses, validation and checkpoint saving.
  First-step total loss was 49.530106; second-step loss 30.952821. These values
  show finite execution only, not OCR quality.
- Resumed the saved optimizer/state checkpoint into a second CPU epoch.
- Upstream held-out evaluation, single-crop prediction and inference-model export
  completed using the resulting checkpoint. The export produced inference model
  files and an inference YAML.

The synthetic images contain placeholder text, not a real Hindi training corpus.
The pilot dataset was **not downloaded or trained in full** here. No actual shard
sample counts, Hindi accuracy, GPU memory requirements, download/extraction on
the full archive, or GPU execution are claimed as tested. Dataset preparation
was exercised with small generated tarballs matching the source format.

The ZIP excludes all downloaded checkpoints, synthetic images, generated labels,
training outputs and upstream source checkout. Run the documented preparation and
configuration steps in your own environment before a real experiment.
