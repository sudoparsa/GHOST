#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError
from skimage.metrics import structural_similarity as ssim

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


def load_image_as_array(path: Path, resize_hw: Optional[int] = None) -> Optional[np.ndarray]:
    """
    Load image as numpy array (grayscale for SSIM).
    Returns None on error.
    """
    try:
        img = Image.open(path).convert("RGB")
        if resize_hw:
            img = img.resize((resize_hw, resize_hw), Image.Resampling.LANCZOS)
        return np.array(img)
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


def compute_ssim_pair(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute SSIM between two RGB images.
    Converts to grayscale and computes SSIM.
    """
    # Convert to grayscale
    gray1 = np.dot(img1[..., :3], [0.2989, 0.5870, 0.1140])
    gray2 = np.dot(img2[..., :3], [0.2989, 0.5870, 0.1140])
    
    # Normalize to [0, 1] if needed
    if gray1.max() > 1.0:
        gray1 = gray1 / 255.0
    if gray2.max() > 1.0:
        gray2 = gray2 / 255.0
    
    # Compute SSIM
    score = ssim(gray1, gray2, data_range=1.0)
    return score


def build_index_map(dir_path: Path) -> Dict[int, Path]:
    """
    Build a mapping from index to image path for a directory.
    """
    index_map: Dict[int, Path] = {}
    for p in find_images(dir_path):
        idx = extract_index(p)
        if idx is not None:
            # If multiple images have same index, keep the first one
            if idx not in index_map:
                index_map[idx] = p
    return index_map


def compute_ssim_for_method(
    gen_dir: Path,
    orig_dir: Path,
    keep_indices: Set[int],
    resize_hw: Optional[int] = None
) -> Tuple[List[float], int]:
    """
    Compute SSIM scores between generated images and originals.
    Returns (list of SSIM scores, count of valid pairs).
    """
    gen_map = build_index_map(gen_dir)
    orig_map = build_index_map(orig_dir)
    
    ssim_scores: List[float] = []
    valid_pairs = 0
    
    for idx in keep_indices:
        if idx not in gen_map:
            continue
        if idx not in orig_map:
            continue
        
        gen_img = load_image_as_array(gen_map[idx], resize_hw)
        orig_img = load_image_as_array(orig_map[idx], resize_hw)
        
        if gen_img is None or orig_img is None:
            continue
        
        # Ensure same size
        if gen_img.shape[:2] != orig_img.shape[:2]:
            # Resize gen_img to match orig_img
            gen_pil = Image.fromarray(gen_img)
            orig_pil = Image.fromarray(orig_img)
            gen_pil = gen_pil.resize(orig_pil.size, Image.Resampling.LANCZOS)
            gen_img = np.array(gen_pil)
        
        score = compute_ssim_pair(gen_img, orig_img)
        ssim_scores.append(score)
        valid_pairs += 1
    
    return ssim_scores, valid_pairs


def get_indices_from_gen_dir(gen_dir: Path) -> Set[int]:
    """
    Extract indices from generated images directory.
    """
    keep_indices: Set[int] = set()
    for p in find_images(gen_dir):
        idx = extract_index(p)
        if idx is not None:
            keep_indices.add(idx)
    return keep_indices


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compute SSIM for multiple classes and generation methods. "
                    "Templates should include `{cls}` where the class name should be inserted."
    )
    parser.add_argument("--classes", type=str, required=True,
                        help='Comma-separated class names, e.g. "boat,traffic light,knife"')
    parser.add_argument("--ghost-tmpl", type=str,
                        help="Template for GHOST generated image dir, e.g. "
                             '"logs/attack/llava/{cls}_lr=.../images"')
    parser.add_argument("--unclip-tmpl", type=str,
                        help="Template for unCLIP generated image dir, e.g. "
                             '"logs/attack/llava/{cls}_lr=.../images"')
    parser.add_argument("--sd-tmpl", type=str,
                        help="Template for SD generated image dir, e.g. "
                             '"logs/attack/llava/{cls}_lr=.../images"')
    parser.add_argument("--orig-tmpl", type=str, required=True,
                        help="Template for ORIGINAL image dir, e.g. "
                             '"logs/attack/qwen/{cls}_lr=.../original"')
    parser.add_argument("--gen-tmpl", type=str, required=True,
                        help="Template for GEN index dir (used to pick indices), e.g. "
                             '"logs/attack/llava/{cls}_lr=.../images"')
    parser.add_argument("--resize", type=int, default=None,
                        help="Square resize (H=W) for SSIM input (default: None, use original size).")
    parser.add_argument("--log-file", type=Path, default=Path("ssim_scores.txt"),
                        help="File to append final SSIM scores (default: ssim_scores.txt).")
    args = parser.parse_args()

    classes = parse_classes(args.classes)
    if not classes:
        raise ValueError("--classes parsed to empty list.")

    methods = []
    if args.ghost_tmpl:
        methods.append(("GHOST", args.ghost_tmpl))
    if args.unclip_tmpl:
        methods.append(("unCLIP", args.unclip_tmpl))
    if args.sd_tmpl:
        methods.append(("SD", args.sd_tmpl))
    
    if not methods:
        raise ValueError("At least one generation method (--ghost-tmpl, --unclip-tmpl, or --sd-tmpl) must be provided.")

    all_results: Dict[str, Dict[str, List[float]]] = {}  # method -> class -> scores
    all_counts: Dict[str, Dict[str, int]] = {}  # method -> class -> count

    for method_name, method_tmpl in methods:
        all_results[method_name] = {}
        all_counts[method_name] = {}
        
        for cls in classes:
            gen_dir = Path(method_tmpl.format(cls=cls))
            orig_dir = Path(args.orig_tmpl.format(cls=cls))
            index_dir = Path(args.gen_tmpl.format(cls=cls))
            
            if not gen_dir.is_dir():
                print(f"[WARN] Generated dir does not exist for method '{method_name}', class '{cls}': {gen_dir} (skipping)")
                continue
            if not orig_dir.is_dir():
                print(f"[WARN] Original dir does not exist for class '{cls}': {orig_dir} (skipping)")
                continue
            if not index_dir.is_dir():
                print(f"[WARN] Index dir does not exist for class '{cls}': {index_dir} (skipping)")
                continue

            # Get indices from index_dir
            keep_indices = get_indices_from_gen_dir(index_dir)
            if not keep_indices:
                print(f"[WARN] No indices extracted from index_dir for class '{cls}': {index_dir} (skipping)")
                continue

            # Compute SSIM
            ssim_scores, valid_pairs = compute_ssim_for_method(
                gen_dir, orig_dir, keep_indices, args.resize
            )
            
            if valid_pairs == 0:
                print(f"[WARN] No valid image pairs found for method '{method_name}', class '{cls}'")
                continue
            
            all_results[method_name][cls] = ssim_scores
            all_counts[method_name][cls] = valid_pairs
            mean_ssim = np.mean(ssim_scores)
            std_ssim = np.std(ssim_scores)
            print(f"[INFO] [{method_name}] [{cls}] SSIM: {mean_ssim:.6f} ± {std_ssim:.6f} (n={valid_pairs})")

    # Print summary
    print("\n" + "="*80)
    print("[SUMMARY]")
    print("="*80)
    
    for method_name, method_tmpl in methods:
        if method_name not in all_results:
            continue
        
        all_scores = []
        total_count = 0
        for cls in classes:
            if cls in all_results[method_name]:
                all_scores.extend(all_results[method_name][cls])
                total_count += all_counts[method_name][cls]
        
        if all_scores:
            overall_mean = np.mean(all_scores)
            overall_std = np.std(all_scores)
            print(f"[RESULT] [{method_name}] Overall SSIM: {overall_mean:.6f} ± {overall_std:.6f} (n={total_count})")
    
    print("="*80)

    # Log results
    try:
        with args.log_file.open("a") as f:
            for method_name, method_tmpl in methods:
                if method_name not in all_results:
                    continue
                
                all_scores = []
                total_count = 0
                for cls in classes:
                    if cls in all_results[method_name]:
                        all_scores.extend(all_results[method_name][cls])
                        total_count += all_counts[method_name][cls]
                
                if all_scores:
                    overall_mean = np.mean(all_scores)
                    overall_std = np.std(all_scores)
                    f.write(
                        f"SSIM method={method_name} classes={classes} "
                        f"gen_tmpl='{method_tmpl}' orig_tmpl='{args.orig_tmpl}' "
                        f"gen_index_tmpl='{args.gen_tmpl}' resize={args.resize}: "
                        f"{overall_mean:.6f} ± {overall_std:.6f} (n={total_count})\n"
                    )
        print(f"[INFO] SSIM scores appended to {args.log_file}")
    except Exception as e:
        print(f"[WARN] Could not write log file {args.log_file}: {e}")


if __name__ == "__main__":
    main()

