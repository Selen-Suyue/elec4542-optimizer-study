# ELEC4542: Comparing optimizers for a small ViT

**Research question:** How do SGD with momentum, Adam, AdamW and Muon with auxiliary AdamW affect convergence, generalization and training cost on the same image-classification task?

This repository provides the fixed ViT, shared image transform, dataset split script and reference starting settings. Students implement the training/evaluation workflow and connect the optimizers using the original references. Form a hypothesis and design an additional controlled experiment to explain your observations.

## 1. Start here

```bash
git clone https://github.com/Selen-Suyue/elec4542-optimizer-study.git
cd elec4542-optimizer-study
python -m venv .venv
source .venv/bin/activate
```

Install a CUDA-compatible PyTorch build using the [official installation selector](https://pytorch.org/get-started/locally/), then install the remaining dependencies with `python -m pip install -r requirements.txt`. Check your environment with `python check_model.py`. This check runs on CPU, performs a forward/backward pass, and verifies identical seeded initialization. It does not train a model.

## 2. Get the data

Download the original [Tiny-ImageNet-200 archive](https://cs231n.stanford.edu/tiny-imagenet-200.zip). It contains 200 classes and 64 × 64 RGB images. Use the original archive for the common baseline so everyone obtains the same class order and split.

```bash
mkdir -p data
curl -fL --retry 3 https://cs231n.stanford.edu/tiny-imagenet-200.zip -o data/tiny-imagenet-200.zip
python prepare_splits.py --archive data/tiny-imagenet-200.zip --output splits
unzip -q data/tiny-imagenet-200.zip -d data
```

The path manifests contain paths relative to `data/tiny-imagenet-200/`:

```text
data/tiny-imagenet-200/
├── wnids.txt                        # class IDs, in the prescribed label order
├── words.txt                        # human-readable names
├── train/<class-id>/images/*.JPEG
└── val/
    ├── images/*.JPEG
    └── val_annotations.txt
splits/
├── train.jsonl                      # 90,000 images; 450 per class
├── val.jsonl                        # 10,000 images; 50 per class
├── test.jsonl                       # 10,000 official labeled validation images
└── manifest.json                    # class order, counts and split hashes
```

Read each JSONL row as `{"path": "train/.../images/....JPEG", "label": 0}`. Open `data/tiny-imagenet-200/<path>` with Pillow, convert to RGB, and create a uint8 tensor with shape `(3,64,64)`. Feed batches into `preprocessing.preprocess` before the model. Images may instead be read directly from the ZIP by adding its `tiny-imagenet-200/` prefix.

**Our final held-out test set is the official labeled validation set.** The course validation set is selected from official training images, by taking 50 out of each class's 500 images with NumPy `default_rng(42)`, following `wnids.txt` order and sorted filenames. Do not use the official unlabeled test set, and do not use held-out test labels to tune settings or select checkpoints. The [Hugging Face mirror](https://huggingface.co/datasets/zh-plus/tiny-imagenet) is useful for browsing; its row/class order should not replace the prescribed manifests without explicit reconciliation.

## 3. Use the fixed ViT

`model.SmallViT()` is randomly initialized and outputs 200 logits. Keep this architecture fixed across the four baseline runs:

| Setting | Value |
|---|---|
| Input / patch size | 64 × 64 / 8 × 8 |
| Patch tokens / class token | 64 / 1 |
| Transformer blocks / width | 6 / 192 |
| Attention heads / MLP ratio | 3 / 4 |
| Normalization / classifier | Pre-LayerNorm / final LayerNorm + CLS-token linear head |
| Dropout / pretrained weights | 0 / none |

The patch embedding is a stride-8 convolution. Attention uses PyTorch scaled dot-product attention. Use the supplied model instead of a pretrained model or a timm architecture with a similar name. The reference initialization uses `torch.manual_seed(42)` immediately before constructing it; and save its initialization hash. Every run starts from the same saved initial state.

## 4. Reference starting settings

The machine-readable reference configuration is [`protocol.json`](protocol.json). These are suggested starting values, not mandatory project requirements. You may choose the training duration, batch size, schedule and method-specific hyperparameters. Keep the model, split, initialization, preprocessing and training budget consistent across the methods being compared, and document changes. If tuning, give each method a comparable validation-based tuning budget; do not tune on held-out test data.

| Shared setting | Value |
|---|---|
| Seed / epochs | 42 / 60 |
| Train batch / evaluation batch | 128 / 512 |
| Loss / gradient clipping | Cross-entropy, no label smoothing / global norm 1.0 |
| LR schedule | 5-epoch linear warmup, then cosine to 1% of each group's base LR; update each step |
| Precision | FP32 parameters, CUDA BF16 autocast when supported |
| Augmentation | Independent square crop: area fraction uniform in [0.6,1.0], bilinear sampling, horizontal flip probability 0.5 |
| Image normalization | Mean (0.485,0.456,0.406), std (0.229,0.224,0.225) |
| Extra regularization | No Mixup, CutMix or label smoothing |

Use `preprocessing.py` for the crop rather than silently substituting torchvision's random resized crop: its aspect-ratio distribution and boundary sampling differ. Use unaugmented, normalized images for validation/test. Keep the last partial training batch. Set `cudnn.benchmark=False`; record TF32 and precision settings. For CUDA reproduction, enable TF32 consistently across runs. Hardware/kernel changes can prevent bitwise equality; save environment details.

The reference configuration uses separate epoch generators: shuffle seed `42+1000+epoch` and augmentation seed `42+2000+epoch`, where epoch starts at 0. Keep identical sample order and augmentation across methods. Apply the same schedule multiplier to **both** Muon and auxiliary AdamW groups.

| Run | Base LR | Weight decay | Other settings |
|---|---:|---:|---|
| SGD + Momentum | 0.1 | 0.0001 | momentum 0.9, Nesterov on |
| Adam | 0.0003 | 0.0001 | betas (0.9,0.999), epsilon 1e-8; coupled L2 |
| AdamW | 0.0003 | 0.05 | betas (0.9,0.999), epsilon 1e-8; decoupled decay |
| Muon + auxiliary AdamW | 0.02 / 0.0003 | 0.05 | Muon momentum 0.95, Nesterov on, 5 Newton–Schulz steps; auxiliary AdamW uses the same betas/epsilon as above |

For SGD/Adam/AdamW, apply the stated decay to tensors with two or more dimensions, and zero decay to vectors/scalars. These suggested coefficients differ **per method**; Adam's coupled L2 and AdamW's decoupled decay are different operations. If using these reference configurations, state that they do not isolate the optimizer rule from every hyperparameter choice. A shared numerical LR is not required across optimizers.

### Muon parameter routing

Follow the [original Muon repository](https://github.com/KellerJordan/Muon) and [author's explanation](https://kellerjordan.github.io/posts/muon/) for the update definition. Cite and record the exact implementation commit you use. For the provided reference configuration:

- **Muon:** parameters whose names start with `blocks.` and whose dimension is exactly 2: attention QKV/projection and MLP matrices (24 tensors).
- **Auxiliary AdamW:** every other parameter, including patch embedding, CLS/position embeddings, normalization, all biases and the classification head. Use decay 0.05 on tensors with ndim ≥ 2 and zero decay otherwise.
- Each trainable tensor belongs to exactly one optimizer. Clip the combined model gradient once before either optimizer step.

The reference setup uses the documented single-device variant with momentum 0.95, five quintic Newton–Schulz iterations (coefficients 3.4445, −4.775, 2.0315), BF16 matrix orthogonalization, normalization epsilon 1e-7 and update scale `sqrt(max(1, rows/columns))`. Record any departure from this variant. The public repository may change, so matching only the name “Muon” is insufficient.

## 5. Build the experiment

1. Prepare the manifests, verify split counts and save the class order and split hashes.
2. Make a one-batch forward/backward check. Confirm that all parameters receive finite gradients, and that Muon/auxiliary groups are disjoint and exhaustive.
3. Use a short development run to check the training/evaluation code, then restart each baseline from the same initialization for your chosen training budget, kept consistent across methods. Do not carry warm-up test weights into the baseline.
4. Record training loss/accuracy, validation loss/accuracy/error once per epoch, and training loss/gradient norm every 25 steps. Error means `1 - top1_accuracy`. Record the data loss consistently; do not add decay penalties to the plotted training loss.
5. Select the checkpoint with the highest validation top-1 accuracy; keep the earliest on a tie. Evaluate the selected checkpoint on held-out test **once after training**.
6. Plot training loss and validation accuracy/error against both epoch and elapsed time. Also report final held-out accuracy/error, total training time and steady-state step time. Synchronize CUDA before timing; use the same hardware, precision and data-loading policy. Include both optimizer steps in the Muon timing.
7. Propose one additional controlled experiment, explain why it tests your hypothesis, and label it separately from your main comparison. For example, investigate LR sensitivity with equal tuning budgets, weight-decay sensitivity, seed variation, or the effect of Newton–Schulz iteration count. Use validation for choices and keep test held out.

Run methods sequentially on one GPU if needed. Eight GPUs are not required. If you cache images on GPU, use that policy for every method and explain what your timing includes. Report empirical conclusions for your settings; a single seed does not establish a universal optimizer ranking.

## 6. What to include in your report

- Your hypothesis and the rationale for the extra controlled experiment.
- A protocol table, model/data hashes and software/hardware details.
- Curves with aligned axes, checkpoint selection rule and final held-out evaluation.
- An explanation connecting observed behavior to momentum, adaptive scaling, weight decay or matrix orthogonalization; discuss limitations and alternative explanations.

## 7. Reading and implementation resources

| Resource | Why read it |
|---|---|
| [Qian (1999), On the momentum term in gradient descent learning algorithms](https://www.columbia.edu/~nq6/publications/momentum.pdf) | Understand momentum; use [PyTorch SGD docs](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html) for the Nesterov variant |
| [Kingma & Ba (ICLR 2015), Adam](https://arxiv.org/abs/1412.6980) | First/second moment adaptation |
| [Loshchilov & Hutter (ICLR 2019), Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) | Explain Adam versus AdamW |
| [Jordan et al. (2024), Muon](https://kellerjordan.github.io/posts/muon/) and [original code](https://github.com/KellerJordan/Muon) | Hidden matrix updates and the need for an auxiliary optimizer |
| [Bernstein, Deriving Muon](https://jeremybernste.in/writing/deriving-muon) | Optional mathematical explanation |
| [Dosovitskiy et al. (ICLR 2021), An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) | Patch tokens and Vision Transformers |
| [PyTorch Adam](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html), [AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html) | Framework implementations and parameter groups |

The code and configuration here are course scaffolding. Dataset terms are governed by the original source; obtain images separately. The repository contains no optimizer implementation, trained checkpoints or benchmark outcomes.
