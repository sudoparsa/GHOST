#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from typing import Iterable, List, Optional, Set

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance

# Try to import your COCO dataset wrapper if the user selects --real-source=coco
try:
    from data import COCO  # type: ignore
except Exception:
    COCO = None

# ----------------------------
# Utils
# ----------------------------
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
_index_regex = re.compile(r".*?_(\d+)(?:\D.*)?$")


def parse_classes(arg: str) -> List[str]:
    """
    Parses a comma-separated class list like:
      "boat,traffic light,knife"
    Keeps whitespace inside names and trims surrounding spaces.
    """
    return [c.strip() for c in arg.split(",") if c.strip()]


def find_images(root: Path) -> List[Path]:
    return [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def extract_index(path: Path) -> Optional[int]:
    """
    Extract integer index from filenames like 'img_123.png', 'foo_42.jpg', 'prefix_7_suf'.
    Returns None if not found.
    """
    m = _index_regex.match(path.stem)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def batched(tensors: List[torch.Tensor], batch_size: int) -> Iterable[torch.Tensor]:
    for i in range(0, len(tensors), batch_size):
        yield torch.stack(tensors[i:i + batch_size], dim=0)


def load_and_transform(path: Path, tfm: transforms.Compose) -> Optional[torch.Tensor]:
    try:
        img = Image.open(path).convert("RGB")
        return tfm(img)
    except UnidentifiedImageError:
        print(f"[WARN] Unidentified image, skipping: {path}")
        return None
    except ValueError as e:
        if "MAX_TEXT_CHUNK" in str(e):
            print(f"[WARN] Skipping (huge PNG text chunk): {path}")
            return None
        raise
    except Exception as e:
        print(f"[WARN] Could not read {path}: {e}")
        return None


def update_fake_dir(fid: FrechetInceptionDistance, fake_dir: Path, keep_indices: Set[int],
                    tfm: transforms.Compose, batch_size: int) -> int:
    imgs: List[torch.Tensor] = []
    used = 0
    for p in find_images(fake_dir):
        idx = extract_index(p)
        if idx is None or idx not in keep_indices:
            continue
        t = load_and_transform(p, tfm)
        if t is None:
            continue
        imgs.append(t)
        used += 1
    for batch in batched(imgs, batch_size):
        fid.update(batch, real=False)
    return used


def update_real_from_original(fid: FrechetInceptionDistance, orig_dir: Path, keep_indices: Set[int],
                              tfm: transforms.Compose, batch_size: int) -> int:
    imgs: List[torch.Tensor] = []
    used = 0
    for p in find_images(orig_dir):
        idx = extract_index(p)
        if idx is None or idx not in keep_indices:
            continue
        t = load_and_transform(p, tfm)
        if t is None:
            continue
        imgs.append(t)
        used += 1
    for batch in batched(imgs, batch_size):
        fid.update(batch, real=True)
    return used


def update_real_from_coco(fid: FrechetInceptionDistance, coco_path: Path, split: str,
                          resize_hw: int, batch_size: int) -> int:
    if COCO is None:
        raise RuntimeError("COCO class import failed; ensure `from data import COCO` works for --real-source=coco.")
    dset = COCO(str(coco_path), split=split, transform=(resize_hw, resize_hw))
    try:
        indices = list(dset.im_dict.keys())  # type: ignore[attr-defined]
    except Exception:
        indices = list(range(len(dset)))

    tfm = transforms.Compose([
        transforms.Resize((resize_hw, resize_hw)),
        transforms.ToTensor()
    ])

    imgs: List[torch.Tensor] = []
    for idx in indices:
        sample = dset[idx]
        img = sample[0] if isinstance(sample, (list, tuple)) else sample
        if isinstance(img, Image.Image):
            t = tfm(img.convert("RGB"))
        elif torch.is_tensor(img) and img.ndim == 3:
            pil = transforms.ToPILImage()(img)
            t = tfm(pil)
        else:
            # Try best-effort conversion, otherwise skip
            try:
                pil = Image.fromarray(img)  # type: ignore[arg-type]
                t = tfm(pil.convert("RGB"))
            except Exception:
                continue
        imgs.append(t)

    for batch in batched(imgs, batch_size):
        fid.update(batch, real=True)
    return len(imgs)


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compute FID for multiple classes using path templates. "
                    "Templates should include `{cls}` where the class name should be inserted."
    )
    parser.add_argument("--classes", type=str, required=True,
                        help='Comma-separated class names, e.g. "boat,traffic light,knife"')
    parser.add_argument("--fake-tmpl", type=str, required=True,
                        help="Template for FAKE image dir, e.g. "
                             '"logs/attack/llava/{cls}_lr=.../images"')
    parser.add_argument("--gen-tmpl", type=str, required=True,
                        help="Template for GEN index dir (used to pick indices), e.g. "
                             '"logs/attack/llava/{cls}_lr=.../images"')
    parser.add_argument("--real-source", choices=["original", "coco"], required=True,
                        help="Use 'original' to compare against per-class original images matched by index, "
                             "or 'coco' to compare against COCO validation set.")
    parser.add_argument("--orig-tmpl", type=str,
                        help="Template for ORIGINAL image dir (required if --real-source=original), e.g. "
                             '"logs/attack/qwen/{cls}_lr=.../original"')
    parser.add_argument("--coco-path", type=Path,
                        help="Path to COCO dataset root (required if --real-source=coco).")
    parser.add_argument("--coco-split", type=str, default="val",
                        help="COCO split (default: val).")
    parser.add_argument("--resize", type=int, default=299,
                        help="Square resize (H=W) for FID input (default: 299).")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size for FID updates (default: 128).")
    parser.add_argument("--log-file", type=Path, default=Path("fid_scores.txt"),
                        help="File to append final FID (default: fid_scores.txt).")
    args = parser.parse_args()

    classes = parse_classes(args.classes)
    if not classes:
        raise ValueError("--classes parsed to empty list.")

    if args.real_source == "original":
        if not args.orig_tmpl:
            raise ValueError("--orig-tmpl is required when --real-source=original.")
    else:
        if args.coco_path is None or not args.coco_path.exists():
            raise FileNotFoundError("--coco-path is required and must exist when --real-source=coco.")
        if COCO is None:
            raise RuntimeError("COCO class import failed; ensure `from data import COCO` works.")

    tfm = transforms.Compose([
        transforms.Resize((args.resize, args.resize)),
        transforms.ToTensor()
    ])

    fid = FrechetInceptionDistance(normalize=True)
    if args.real_source == "coco":
        real_count = update_real_from_coco(fid, args.coco_path, args.coco_split, args.resize, args.batch_size) 
        print(f"[INFO] Loaded COCO {args.coco_split} real set: {real_count} images")

    total_fake = 0
    total_real_orig = 0

    for cls in classes:
        fake_dir = Path(args.fake_tmpl.format(cls=cls))
        gen_dir = Path(args.gen_tmpl.format(cls=cls))
        if not fake_dir.is_dir():
            print(f"[WARN] Fake dir does not exist for class '{cls}': {fake_dir} (skipping class)")
            continue
        if not gen_dir.is_dir():
            print(f"[WARN] Gen dir does not exist for class '{cls}': {gen_dir} (skipping class)")
            continue

        # Build the index set from gen_dir for this class
        keep_indices: Set[int] = set()
        for p in find_images(gen_dir):
            idx = extract_index(p)
            if idx is not None:
                keep_indices.add(idx)

        if not keep_indices:
            print(f"[WARN] No indices extracted from gen_dir for class '{cls}': {gen_dir} (skipping class)")
            continue

        # Update fake for this class
        used_fake = update_fake_dir(fid, fake_dir, keep_indices, tfm, args.batch_size)
        total_fake += used_fake
        print(f"[INFO] [{cls}] Fake used: {used_fake}")

        # If ORIGINAL is the real source, update real per class from its own orig dir
        if args.real_source == "original":
            orig_dir = Path(args.orig_tmpl.format(cls=cls))  # type: ignore[arg-type]
            if not orig_dir.is_dir():
                print(f"[WARN] Orig dir does not exist for class '{cls}': {orig_dir} (skipping real for this class)")
            else:
                used_real = update_real_from_original(fid, orig_dir, keep_indices, tfm, args.batch_size)
                total_real_orig += used_real
                print(f"[INFO] [{cls}] Real (original) used: {used_real}")

    # Compute FID
    score = fid.compute().item()
    print(f"[RESULT] FID: {score:.6f}")
    if args.real_source == "original":
        print(f"[STATS] Total fake: {total_fake} | Total real (original): {total_real_orig}")
    else:
        print(f"[STATS] Total fake: {total_fake} | Real: COCO({args.coco_split}) preloaded")

    # Log
    try:
        with args.log_file.open("a") as f:
            f.write(
                f"FID real_source={args.real_source} classes={classes} "
                f"fake_tmpl='{args.fake_tmpl}' gen_tmpl='{args.gen_tmpl}' "
                f"{'orig_tmpl='+repr(args.orig_tmpl) if args.real_source=='original' else 'coco_path='+str(args.coco_path)} "
                f"resize={args.resize} batch={args.batch_size}: {score:.6f}\n"
            )
        print(f"[INFO] FID score appended to {args.log_file}")
    except Exception as e:
        print(f"[WARN] Could not write log file {args.log_file}: {e}")


if __name__ == "__main__":
    main()


