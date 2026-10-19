# GHOST: Hallucination-Inducing Image Generation for Multimodal LLMs

Official code for **GHOST** (Generating Hallucinations via Optimizing Stealth Tokens), published at [ICLR 2026](https://iclr.cc/).

GHOST stress-tests object hallucination in multimodal LLMs. Given an image that does not contain a target object, it optimizes the image’s CLIP embedding so the victim model answers “Yes,” then decodes that embedding with Stable Diffusion unCLIP. An OWLv2 filter drops generations that actually contain the object.

**Authors:** Aryan Yazdan Parast\*, Parsa Hosseini\*, Hesam Asadollahzadeh, Arshia Soltani Moakhar, Basim Azam, Soheil Feizi, Naveed Akhtar  
*\*Equal contribution*

- **Paper:** [arXiv:2509.25178](https://arxiv.org/abs/2509.25178)
- **OpenReview:** [ICLR 2026](https://openreview.net/forum?id=f4TACE7HhU)
- **Dataset:** [Hugging Face](https://huggingface.co/datasets/Arianyp/GHOST) — the successful Qwen2.5-VL (2,816) and LLaVA-v1.6 (2,462) images from the main COCO experiment

## Repository layout

| Path | Role |
| --- | --- |
| `train_projector.py` | Train the mapper \(\Pi\) that aligns CLIP embeddings with the victim vision encoder |
| `main.py` | Optimize a CLIP embedding and generate hallucination-inducing images |
| `projector/` | MLP mapper |
| `data.py` | COCO loader used by the attack |
| `enhanced_text_representations.py` | Compositional CLIP text target for \(\mathcal{L}_{\text{clip}}\) |
| `evals/validate_projector.py` | Yes/No accuracy of the mapper on held-out COCO classes |
| `evals/fid_eval.py` | FID against COCO val or the original images |
| `evals/SSIM-eval.py` | SSIM against the original images |
| `evals/transfer_eval.py` | Transfer of generated images to another VLM |
| `evals/eval_projector_gpt.py` | GPT-4 reconstruction score for the mapper |

## Setup

Python 3.11 and a CUDA GPU. The paper runs used a single A100 (about 10 seconds per sample, including optimization and generation).

```bash
pip install torch torchvision
pip install transformers diffusers accelerate
pip install open_clip_torch
pip install pillow tqdm matplotlib pandas
pip install torchmetrics scikit-image
```

`evals/eval_projector_gpt.py` also needs the `openai` package and `OPENAI_API_KEY`.

Point Hugging Face and Torch caches at a local directory before the first download:

```bash
export HF_HOME=$PWD/cache
export TORCH_HOME=$PWD/cache/torch
export XDG_CACHE_HOME=$PWD/cache/xdg
```

Gated checkpoints (`meta-llama/Llama-3.2-11B-Vision-Instruct`) need `huggingface-cli login`. The attack itself uses the open models below.

| Role | Checkpoint |
| --- | --- |
| Qwen2.5-VL | `Qwen/Qwen2.5-VL-7B-Instruct` |
| LLaVA-v1.6 | `llava-hf/llava-v1.6-mistral-7b-hf` |
| GLM-4.1V-Thinking | `THUDM/GLM-4.1V-9B-Thinking` |
| CLIP (OpenCLIP) | `ViT-H-14`, pretrained `laion2b_s32b_b79k` |
| Diffusion | `stabilityai/stable-diffusion-2-1-unclip` |
| Object filter | `google/owlv2-base-patch16-ensemble` |

## Data

**Attack** (`main.py --data_path`) expects the standard COCO layout:

```text
COCO/
  images/train2017/
  annotations/instances_train2017.json
  annotations/captions_train2017.json
```

**Mapper training** (`train_projector.py --coco_dir`) expects the image folders directly:

```text
COCO/images/train2017/
COCO/images/val2017/
```

Pass `--coco_dir /path/to/COCO/images` and `--data_path /path/to/COCO`.

With `--sort_images True`, `main.py` loads `./clip_embeddings.pt` from the working directory. The file used for the paper is [`clip_embeddings.pt`](https://huggingface.co/datasets/Arianyp/GHOST/blob/main/clip_embeddings.pt).

The ten COCO classes used in the paper are:

`boat`, `clock`, `knife`, `carrot`, `toilet`, `vase`, `bus`, `suitcase`, `bottle`, `traffic light`

## 1. Train the mapper

The mapper is an MLP. Checkpoints are written to `projector/models/` and the filename encodes `context_dim` and `hidden_dim`, which `main.py` reads back when it loads the weights.

These are the settings from the paper (AdamW, weight decay `0.01`, cosine schedule with `T_max = 10`, 1000 warmup steps, 10 epochs, full COCO). The hidden sizes are the ones used for the released attack runs.

```bash
# Qwen2.5-VL  (batch 32, context 4096, hidden 2048)
python train_projector.py \
    --model_name qwen \
    --coco_dir /path/to/COCO/images \
    --cache_path cache \
    --batch_size 32 \
    --lr 0.0002 \
    --weight_decay 0.01 \
    --scheduler cosine \
    --cosine_tmax 10 \
    --warmup_steps 1000 \
    --epochs 10 \
    --context_dim 4096 \
    --hidden_dim 2048 \
    --coco_subset 1.0

# LLaVA-v1.6  (batch 64, context 4096, hidden 1024)
python train_projector.py \
    --model_name llava \
    --coco_dir /path/to/COCO/images \
    --cache_path cache \
    --batch_size 64 \
    --lr 0.0002 \
    --weight_decay 0.01 \
    --scheduler cosine \
    --cosine_tmax 10 \
    --warmup_steps 1000 \
    --epochs 10 \
    --context_dim 4096 \
    --hidden_dim 1024 \
    --coco_subset 1.0

# GLM-4.1V-Thinking  (batch 32, context 4096, hidden 2048)
python train_projector.py \
    --model_name glm4.1v-thinking \
    --coco_dir /path/to/COCO/images \
    --cache_path cache \
    --batch_size 32 \
    --lr 0.0002 \
    --weight_decay 0.01 \
    --scheduler cosine \
    --cosine_tmax 10 \
    --warmup_steps 1000 \
    --epochs 10 \
    --context_dim 4096 \
    --hidden_dim 2048 \
    --coco_subset 1.0
```

The epoch-10 checkpoints used by the attack are:

```text
projector/models/qwen_bs=32_lr=0.0002_epochs=10_context_dim=4096_hidden_dim=2048_coco_subset=1.0_epoch_10.pt
projector/models/llava_bs=64_lr=0.0002_epochs=10_context_dim=4096_hidden_dim=1024_coco_subset=1.0_epoch_10.pt
projector/models/glm4.1v-thinking_bs=32_lr=0.0002_epochs=10_context_dim=4096_hidden_dim=2048_coco_subset=1.0_epoch_10.pt
```

Optional check after training (the classes used for mapper selection):

```bash
python evals/validate_projector.py \
    --model_name qwen \
    --projector_path projector/models/qwen_bs=32_lr=0.0002_epochs=10_context_dim=4096_hidden_dim=2048_coco_subset=1.0_epoch_10.pt \
    --data_path /path/to/COCO \
    --cache_path cache \
    --target_objects vase,boat,bird,giraffe,car,remote
```

## 2. Generate images

`main.py` drops images where the victim model already answers “Yes,” optimizes the CLIP embedding, generates up to `N` unCLIP samples from a partially noised latent of the original image, and keeps a sample only when the victim says “Yes” and OWLv2 does not detect the target.

Successful images land in:

```text
logs/attack/<model>/<run name>/images/
logs/attack/<model>/<run name>/original/
logs/attack/<model>/<run name>/log.txt
```

Paper hyperparameters (`λ_clip` is `--lambda_contrast`, `τ` is `--threshold`, noise level is `--t`):

| Model | lr | steps | τ | N | guidance | λ_clip | λ_reg | OD thr. | t | inference steps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-VL | 0.1 | 100 | 0.8 | 4 | 5 | 15 | 10 | 0.5 | 30 | 50 |
| LLaVA-v1.6 | 0.1 | 100 | 0.8 | 4 | 5 | 15 | 10 | 0.5 | 30 | 50 |
| GLM-4.1V-Thinking | 0.2 | 125 | 0.5 | 5 | 5 | 0.5 | 1.5 | 0.5 | 30 | 50 |

Qwen, one class:

```bash
python main.py \
    --model_name qwen \
    --projector_path projector/models/qwen_bs=32_lr=0.0002_epochs=10_context_dim=4096_hidden_dim=2048_coco_subset=1.0_epoch_10.pt \
    --data_path /path/to/COCO \
    --cache_path cache \
    --target_object bottle \
    --lr 0.1 \
    --num_steps 100 \
    --threshold 0.8 \
    --num_generation 4 \
    --guidance_scale 5 \
    --lambda_contrast 15.0 \
    --lambda_reg 10.0 \
    --OD_threshold 0.5 \
    --use_enhanced_text True \
    --sort_images True \
    --num_of_inference 50 \
    --t 30 \
    --seed 42
```

LLaVA uses the same attack flags with `--model_name llava` and the LLaVA checkpoint above.

GLM-4.1V-Thinking, all ten classes (this is the submitted run):

```bash
for target_object in boat clock knife carrot toilet vase bus suitcase bottle "traffic light"
do
python main.py \
    --model_name glm4.1v-thinking \
    --projector_path projector/models/glm4.1v-thinking_bs=32_lr=0.0002_epochs=10_context_dim=4096_hidden_dim=2048_coco_subset=1.0_epoch_10.pt \
    --data_path /path/to/COCO \
    --cache_path cache \
    --target_object "$target_object" \
    --lr 0.2 \
    --num_steps 125 \
    --lambda_contrast 0.50 \
    --lambda_reg 1.50 \
    --num_generation 5 \
    --threshold 0.5 \
    --guidance_scale 5 \
    --use_enhanced_text True \
    --sort_images True \
    --OD_threshold 0.5 \
    --num_of_inference 50 \
    --t 30 \
    --seed 42
done
```

`main.py` currently walks only ranked indices `[200, 400)` after CLIP sorting. That window is a leftover debug slice. Delete it to run the full negative pool. The paper keeps the top 1,000 CLIP-sorted negatives per class, then drops images the model already labels as containing the object.

Set `--OD_threshold 1` to skip the OWLv2 filter (attribute and relation queries in the paper do not use a detector).

## 3. Evaluate

Replace `{cls}` with the class name. The run-name string is the directory `main.py` creates under `logs/attack/<model>/`.

FID against COCO validation (distributional realism) or against the saved originals (semantic fidelity):

```bash
python evals/fid_eval.py \
    --classes "boat,clock,knife,carrot,toilet,vase,bus,suitcase,bottle,traffic light" \
    --fake-tmpl "logs/attack/qwen/{cls}_lr=0.1_steps=100_threshold=0.8_num_generation=4_guidance_scale=5.0_lambda_contrast=15.0_lambda_reg=10.0_OD_threshold=0.5__sort=True_t=30_num_inf=50_<projector stem>/images" \
    --gen-tmpl  "logs/attack/qwen/{cls}_lr=0.1_steps=100_threshold=0.8_num_generation=4_guidance_scale=5.0_lambda_contrast=15.0_lambda_reg=10.0_OD_threshold=0.5__sort=True_t=30_num_inf=50_<projector stem>/images" \
    --real-source coco \
    --coco-path /path/to/COCO
```

Use `--real-source original` and `--orig-tmpl ".../original"` for semantic fidelity.

SSIM uses the same directory templates:

```bash
python evals/SSIM-eval.py \
    --classes "boat,bottle" \
    --ghost-tmpl "logs/attack/qwen/{cls}_<run name>/images" \
    --orig-tmpl  "logs/attack/qwen/{cls}_<run name>/original" \
    --gen-tmpl   "logs/attack/qwen/{cls}_<run name>/images"
```

Transfer to another model (yes-rate on images optimized for a source model):

```bash
python evals/transfer_eval.py \
    --model-type qwen \
    --model-id Qwen/Qwen2.5-VL-7B-Instruct \
    --images-dir-template "logs/attack/llava/{cls}_<run name>/images" \
    --classes "boat,clock,knife,carrot,toilet,vase,bus,suitcase,bottle,traffic light" \
    --cache-dir cache \
    --save-dir logs/transfer
```

`--model-type` is one of `qwen`, `llava`, `llama`, `glm-thinking`. For GLM, add `--glm-device-map-auto`.

## Citation

```bibtex
@article{parast2025ghost,
  title={GHOST: Hallucination-Inducing Image Generation for Multimodal LLMs},
  author={Parast, Aryan Yazdan and Hosseini, Parsa and Asadollahzadeh, Hesam and Moakhar, Arshia Soltani and Azam, Basim and Feizi, Soheil and Akhtar, Naveed},
  journal={arXiv preprint arXiv:2509.25178},
  year={2025}
}
```
