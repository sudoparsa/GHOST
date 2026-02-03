#!/usr/bin/env python3
"""
Multi-class transferability evaluator (yes/no) for VLMs.

- Evaluates "Is there a <class> in this image?" per class, separately.
- Accepts an images directory *template* with {cls} placeholder, e.g.
    logs/attack/{model}/{cls}_.../images
- Computes per-class summary and an overall mean across classes.
- Supports model types: qwen, llava, llama

Examples:
  # Qwen
  python transfer_eval_multi.py \
    --model-type qwen \
    --model-id Qwen/Qwen2.5-VL-7B-Instruct \
    --images-dir-template "logs/attack/qwen/{cls}_.../images" \
    --classes "carrot,knife,clock,toilet,boat,suitcase,bottle,vase,bus" \
    --save-dir out_qwen

"""
import re

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from transformers import Glm4vForConditionalGeneration, AutoProcessor, PaliGemmaForConditionalGeneration
from utils import *
import torch
from PIL import Image
import pandas as pd

# -----------------------------
# Utilities
# -----------------------------

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

def get_image_paths(root: str) -> List[str]:
    p = Path(root)
    if p.is_file() and p.suffix.lower() in IMG_EXTS:
        return [str(p)]
    if not p.exists():
        return []
    return [str(fp) for fp in p.rglob("*") if fp.suffix.lower() in IMG_EXTS]

def pick_device(explicit: Optional[str] = None) -> torch.device:
    if explicit:
        if explicit.lower() == "cpu":
            return torch.device("cpu")
        if explicit.lower().startswith("cuda"):
            return torch.device(explicit)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_summary(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {"yes_share": 0.0, "yes_probability_avg": 0.0, "len": 0}
    yes_count = sum(1 for r in rows if r.get("predicted_answer") == "yes")
    yes_share = yes_count / len(rows)
    yes_prob_avg = sum(r.get("yes_probability", 0.0) for r in rows) / len(rows)
    return {"yes_share": round(yes_share, 4),
            "yes_probability_avg": round(yes_prob_avg, 4),
            "len": len(rows)}

def parse_classes(s: str) -> List[str]:
    """
    Accepts:
      - comma-separated list: "boat,bus,bottle"
      - path to a text file with one class per line
    """
    p = Path(s)
    if p.exists() and p.is_file():
        with open(p, "r") as f:
            return [ln.strip() for ln in f if ln.strip()]
    return [c.strip() for c in s.split(",") if c.strip()]

def _build_yes_no_id_lists(tokenizer) -> (List[int], List[int]):
    """Collect token ids that decode (after stripping) to 'yes' or 'no'."""
    yes_ids, no_ids = [], []
    vocab_size = tokenizer.vocab_size
    for tok_id in range(vocab_size):
        token_text = tokenizer.decode(tok_id).strip().strip(".").lower()
        if token_text == "yes" or token_text == "Yes" or token_text == "YES":
            yes_ids.append(tok_id)
        elif token_text == "no" or token_text == "No" or token_text == "NO":
            no_ids.append(tok_id)
    # fallback: short variants containing yes/no
    if not yes_ids:
        for tok_id in range(vocab_size):
            t = tokenizer.decode(tok_id).lower().strip()
            if "yes" in t and len(t) <= 4:
                yes_ids.append(tok_id)
    if not no_ids:
        for tok_id in range(vocab_size):
            t = tokenizer.decode(tok_id).lower().strip()
            if "no" in t and len(t) <= 3:
                no_ids.append(tok_id)
    return yes_ids, no_ids

def _dtype_from_str(dtype: Optional[str]) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }.get((dtype or "float16").lower(), torch.float16)

# -----------------------------
# Adapters
# -----------------------------

class QwenYesNoAdapter:
    """
    Qwen/Qwen2.5-VL-* adapter for yes/no probabilities.
    """
    def __init__(self, model_id: str, device: torch.device, cache_dir: Optional[str] = None,
                 dtype: Optional[str] = "float16", hf_token: Optional[str] = None, lora: bool = False, finetuneed_path: Optional[str] = "./Finetuned_qwen/lora-finetuned-best"):
        

        
        qwen,processor = get_model('qwen',cache_path=cache_dir) 
        qwen.to('cuda')
        self.model = qwen  

        self.processor = processor
        
        if lora:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, finetuneed_path, torch_dtype=torch.float16)
            self.model.to(device)
            self.processor = AutoProcessor.from_pretrained(finetuneed_path)

            print("✅ LoRA weights loaded")
            

        self.device = device

        self.yes_ids, self.no_ids = _build_yes_no_id_lists(self.processor.tokenizer)

    @torch.no_grad()
    def yes_no_probabilities(self, image_path: str, question: str, max_new_tokens: int = 1) -> Dict[str, float]:
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            return {"error": f"Could not open image: {e}"}

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": f"{question} Please respond with only 'yes' or 'no'."},
            ],
        }]

        text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text_prompt], images=[img], return_tensors="pt", padding=True).to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1,
                output_scores=True,
                return_dict_in_generate=True
            )
        
        

        if not outputs.scores:
            return {"error": "No scores returned from model generation."}

        scores = outputs.scores[0][0]
        probs = torch.nn.functional.softmax(scores, dim=-1)

        yes_prob = float(sum(probs[i].item() for i in self.yes_ids))
        no_prob  = float(sum(probs[i].item() for i in self.no_ids))
        return {"yes": yes_prob, "no": no_prob}


class LlavaYesNoAdapter:
    """
    LLaVA (LlavaNextForConditionalGeneration + LlavaNextProcessor) adapter for yes/no probabilities.
    """
    def __init__(self, model_id: str, device: torch.device, cache_dir: Optional[str] = None,
                 dtype: Optional[str] = "float16", hf_token: Optional[str] = None):
        from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=_dtype_from_str(dtype),
            cache_dir=cache_dir,
            token=hf_token
        ).to(device)

        self.processor = LlavaNextProcessor.from_pretrained(model_id, cache_dir=cache_dir, token=hf_token)
        # Pad/eos safety tweaks often needed for LLaVA variants
        if getattr(self.processor, "tokenizer", None) is not None:
            tok = self.processor.tokenizer
            if tok.pad_token is None and tok.eos_token is not None:
                tok.pad_token = tok.eos_token
        if getattr(self.model, "config", None) is not None and getattr(self.processor, "tokenizer", None) is not None:
            self.model.config.pad_token_id = self.processor.tokenizer.eos_token_id
            if hasattr(self.model, "generation_config"):
                self.model.generation_config.pad_token_id = self.model.config.eos_token_id

        self.device = device
        self.yes_ids, self.no_ids = _build_yes_no_id_lists(self.processor.tokenizer)

    @torch.no_grad()
    def yes_no_probabilities(self, image_path: str, question: str, max_new_tokens: int = 1) -> Dict[str, float]:
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            return {"error": f"Could not open image: {e}"}

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": f"{question} Please respond with only 'yes' or 'no'."},
            ],
        }]

        text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text_prompt], images=[img], return_tensors="pt", padding=True).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True
        )

        if not outputs.scores:
            return {"error": "No scores returned from model generation."}

        scores = outputs.scores[0][0]
        probs = torch.nn.functional.softmax(scores, dim=-1)

        yes_prob = float(sum(probs[i].item() for i in self.yes_ids))
        no_prob  = float(sum(probs[i].item() for i in self.no_ids))
        return {"yes": yes_prob, "no": no_prob}


class LlamaVisionYesNoAdapter:
    """
    Llama 3.2 Vision (MllamaForConditionalGeneration + AutoProcessor) adapter for yes/no probabilities.
    """
    def __init__(self, model_id: str, device: torch.device, cache_dir: Optional[str] = None,
                 dtype: Optional[str] = "float16", hf_token: Optional[str] = None, device_map_auto: bool = False):
        from transformers import MllamaForConditionalGeneration, AutoProcessor

        # Some Llama-Vision checkpoints prefer device_map="auto" when using multi-GPU.
        common_kwargs = dict(
            torch_dtype=_dtype_from_str(dtype),
            cache_dir=cache_dir,
            token=hf_token,
            low_cpu_mem_usage=True
        )
        if device_map_auto:
            self.model = MllamaForConditionalGeneration.from_pretrained(
                model_id, device_map="auto", **common_kwargs
            )
        else:
            self.model = MllamaForConditionalGeneration.from_pretrained(
                model_id, **common_kwargs
            ).to(device)

        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir, token=hf_token)
        # pad/eos alignment
        if getattr(self.processor, "tokenizer", None) is not None:
            tok = self.processor.tokenizer
            if tok.pad_token is None and tok.eos_token is not None:
                tok.pad_token = tok.eos_token
        if getattr(self.model, "config", None) is not None and getattr(self.processor, "tokenizer", None) is not None:
            self.model.config.pad_token_id = self.processor.tokenizer.eos_token_id
            if hasattr(self.model, "generation_config"):
                self.model.generation_config.pad_token_id = self.model.config.eos_token_id

        self.device = device
        self.yes_ids, self.no_ids = _build_yes_no_id_lists(self.processor.tokenizer)

    @torch.no_grad()
    def yes_no_probabilities(self, image_path: str, question: str, max_new_tokens: int = 1) -> Dict[str, float]:
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            return {"error": f"Could not open image: {e}"}

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": f"{question} Please respond with only 'yes' or 'no'."},
            ],
        }]

        text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text_prompt], images=[img], return_tensors="pt", padding=True)

        # If the model is on a single device, move tensors; with device_map="auto", let HF handle it.
        try:
            inputs = inputs.to(self.model.device)
        except Exception:
            pass

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True
        )

        if not outputs.scores:
            return {"error": "No scores returned from model generation."}

        scores = outputs.scores[0][0]
        probs = torch.nn.functional.softmax(scores, dim=-1)

        yes_prob = float(sum(probs[i].item() for i in self.yes_ids))
        no_prob  = float(sum(probs[i].item() for i in self.no_ids))
        return {"yes": yes_prob, "no": no_prob}

def parse_glm_response(text):
    
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    think = think_match.group(1).strip() if think_match else None

    # extract <answer>
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    answer = answer_match.group(1).strip() if answer_match else None

    # if |begin_of_box| and |end_of_box| exist, extract only that part
    boxed = None
    if answer:
        box_match = re.search(r"<\|begin_of_box\|>(.*?)<\|end_of_box\|>", answer, re.DOTALL)
        if box_match:
            boxed = box_match.group(1).strip()
            answer = re.sub(r"<\|begin_of_box\|>(.*?)<\|end_of_box\|>", boxed, answer, flags=re.DOTALL).strip()
    
    if not answer:
        print("!!! No answer found")
        print(text)

    return {
        "think": think,
        "answer": answer,
        "boxed_answer": boxed
    }

class GlmThinkingYesNoAdapter:
    """
    THUDM/GLM-4.1V-9B-Thinking adapter for yes/no probabilities.
    Notes:
      - GLM 'thinking' models often produce <think>...</think><answer>...</answer>.
      - To read yes/no probabilities reliably, we *prefill* the prompt with '<answer>'
        so the first generated token is the start of the answer content (ideally 'yes'/'no').
    """
    def __init__(self, model_id: str, device: torch.device, cache_dir: Optional[str] = None,
                 dtype: Optional[str] = "float16", hf_token: Optional[str] = None, device_map_auto: bool = True):
        

        common_kwargs = dict(
            torch_dtype=_dtype_from_str(dtype),
            cache_dir=cache_dir,
            token=hf_token,
            low_cpu_mem_usage=True
        )
        self.model = Glm4vForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path=model_id,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir, token=hf_token)

        # pad/eos alignment (usually safe)
        if getattr(self.processor, "tokenizer", None) is not None:
            tok = self.processor.tokenizer
            if tok.pad_token is None and tok.eos_token is not None:
                tok.pad_token = tok.eos_token
        if getattr(self.model, "config", None) is not None and getattr(self.processor, "tokenizer", None) is not None:
            self.model.config.pad_token_id = self.processor.tokenizer.eos_token_id
            if hasattr(self.model, "generation_config"):
                self.model.generation_config.pad_token_id = self.model.config.eos_token_id

        self.device = device
        self.yes_ids, self.no_ids = _build_yes_no_id_lists(self.processor.tokenizer)
        self.model_id = model_id
        self.model = Glm4vForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path=self.model_id,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )

    @torch.no_grad()
    def yes_no_probabilities(self, image_path: str, question: str, max_new_tokens: int = 258) -> Dict[str, float]:
        """
        Let GLM freely produce a <think>...</think> phase and only then locate the
        *first* token after '<answer>' to read the yes/no distribution from that step.
        """
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            return {"error": f"Could not open image: {e}"}

        

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": f"{question} Please respond with only 'yes' or 'no'."},
            ],
        }]
        
        
        inputs = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt"
                ).to(self.model.device)
       
        # Allow enough steps for <think> ... <answer> ... to appear
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True
        )
        if not outputs.scores or "input_ids" not in inputs:
            return {"error": "No scores or input_ids returned from model generation."}

        # Locate the first occurrence of the token sequence for "<answer>" in the GENERATED part.
        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = outputs.sequences[0][prompt_len:]  # only newly generated tokens
        
       
        answer_tag_ids = self.processor.tokenizer.encode("<|begin_of_box|>", add_special_tokens=False)
        # Sliding-window search for the subsequence
        start_idx = -1
        L = len(answer_tag_ids)
        for i in range(0, max(0, generated_ids.shape[0] - L + 1)):
            if torch.equal(generated_ids[i:i+L], torch.tensor(answer_tag_ids, device=generated_ids.device)):
                start_idx = i
                break

        if start_idx == -1:
            # Could not find |begin_of_box|" in the generated text
            return {"error": "answer_tag_not_found"}

        # We want the FIRST token AFTER "|begin_of_box|"
        idx_after = start_idx + L  # index into generated_ids
        if idx_after >= len(outputs.scores):
            return {"error": "no_token_after_answer"}

        # Read the distribution at that step
        scores_next = outputs.scores[idx_after][0]  # scores for the token right after "|begin_of_box|"
        probs = torch.nn.functional.softmax(scores_next, dim=-1)

        yes_prob = float(sum(probs[i].item() for i in self.yes_ids))
        no_prob  = float(sum(probs[i].item() for i in self.no_ids))
        return {"yes": yes_prob, "no": no_prob}


class PaliGemmaYesNoAdapter:
    """
    PaliGemma (PaliGemmaForConditionalGeneration + AutoProcessor) adapter for yes/no probabilities.
    Mirrors the interface/flow of LlamaVisionYesNoAdapter.
    """
    def __init__(
        self,
        model_id: str = "google/paligemma-3b-mix-224",
        device: torch.device = torch.device("cuda"),
        cache_dir: Optional[str] = None,
        dtype: Optional[str] = "bfloat16",
        hf_token: Optional[str] = None,
        device_map_auto: bool = False,
        revision: Optional[str] = "bfloat16",
    ):
        
        common_kwargs = dict(
            torch_dtype=_dtype_from_str(dtype),  # same helper you used in the Llama adapter
            cache_dir=cache_dir,
            token=hf_token,
            low_cpu_mem_usage=True,
            revision=revision,
        )

        if device_map_auto:
            # Let HF dispatch layers across devices if you’re on multi-GPU.
            self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                model_id, device_map="auto", **common_kwargs
            ).eval()
        else:
            self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                model_id, **common_kwargs
            ).to(device).eval()

        self.processor = AutoProcessor.from_pretrained(
            model_id, cache_dir=cache_dir, token=hf_token
        )

        # pad/eos alignment (mirrors your Llama logic)
        if getattr(self.processor, "tokenizer", None) is not None:
            tok = self.processor.tokenizer
            if tok.pad_token is None and tok.eos_token is not None:
                tok.pad_token = tok.eos_token
        if getattr(self.model, "config", None) is not None and getattr(self.processor, "tokenizer", None) is not None:
            self.model.config.pad_token_id = self.processor.tokenizer.eos_token_id
            if hasattr(self.model, "generation_config"):
                self.model.generation_config.pad_token_id = self.model.config.eos_token_id

        self.device = device
        # Reuse your yes/no id builder (works across tokenizers)
        self.yes_ids, self.no_ids = _build_yes_no_id_lists(self.processor.tokenizer)

    @torch.no_grad()
    def yes_no_probabilities(self, image_path: str, question: str, max_new_tokens: int = 1) -> Dict[str, float]:
        # Load image
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            return {"error": f"Could not open image: {e}"}

        # For PaliGemma, a plain text prompt works well. Keep it consistent with your Llama adapter:
        # Nudge the model to answer with a single token 'yes' or 'no'.
        prompt = f"{question} Please respond with only 'yes' or 'no'."

        # Processor packs both text and image
        inputs = self.processor(text=prompt, images=img, return_tensors="pt", padding=True)

        # If single-device, move tensors; with device_map="auto", leave placement to HF
        try:
            inputs = inputs.to(self.model.device)
        except Exception:
            pass

        # We want first-token probabilities for the generated answer.
        # Capture scores and only 1 new token (like the Llama adapter).
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )

        if not outputs.scores:
            return {"error": "No scores returned from model generation."}

        # First generated token distribution
        scores = outputs.scores[0][0]           # [vocab]
        probs = torch.nn.functional.softmax(scores, dim=-1)

        yes_prob = float(sum(probs[i].item() for i in self.yes_ids if 0 <= i < probs.shape[-1]))
        no_prob  = float(sum(probs[i].item() for i in self.no_ids  if 0 <= i < probs.shape[-1]))
        return {"yes": yes_prob, "no": no_prob}
# -----------------------------
# Single-class evaluation
# -----------------------------

def eval_one_class(
    adapter,  # any adapter with yes_no_probabilities()
    images_dir: str,
    object_name: str,
    silent: bool
) -> Dict[str, object]:
    image_paths = get_image_paths(images_dir)
    if not image_paths:
        if not silent:
            print(f"[WARN] No images found for class '{object_name}' in: {images_dir}")
        return {"summary": {"yes_share": 0.0, "yes_probability_avg": 0.0, "len": 0}, "rows": []}

    rows = []
    question = f"Is there a {object_name} in this image?"

    for idx, img_path in enumerate(sorted(image_paths)):
        probs = adapter.yes_no_probabilities(img_path, question)
        if "error" in probs:
            if not silent:
                print(f"[{idx+1}/{len(image_paths)}] {img_path} -> ERROR: {probs['error']}")
            continue

        if probs["yes"] >= probs["no"]:
            pred = "yes"
            conf = probs["yes"]
        else:
            pred = "no"
            conf = probs["no"]
        row = {
            "image_path": img_path,
            "predicted_answer": pred,
            "confidence": conf,
            "yes_probability": probs["yes"],
            "no_probability": probs["no"]
        }
        rows.append(row)

        if not silent:
            print(f"[{idx+1}/{len(image_paths)}] {img_path} -> yes={probs['yes']:.4f}, no={probs['no']:.4f}, pred='{pred}' ({conf:.2%})")

    summary = compute_summary(rows)
    if not silent:
        print(f"Summary for '{object_name}': yes_share={summary['yes_share']}, "
              f"yes_probability_avg={summary['yes_probability_avg']}, len={summary['len']}")
    return {"summary": summary, "rows": rows}

# -----------------------------
# Multi-class runner
# -----------------------------


def run_multi(
    model_type: str,
    model_id: str,
    images_dir_template: str,
    classes: List[str],
    device_str: Optional[str],
    cache_dir: Optional[str],
    dtype: Optional[str],
    save_dir: Optional[str],
    hf_token: Optional[str],
    llama_device_map_auto: bool,
    glm_device_map_auto: bool,
    silent: bool,
    lora: bool = False
):
    device = pick_device(device_str)
    mt = model_type.lower().strip()
    total_images = 0         
    total_yes = 0            
    sum_yes_prob = 0.0       



    if mt == "qwen":
        adapter = QwenYesNoAdapter(model_id=model_id, device=device, cache_dir=cache_dir, dtype=dtype, hf_token=hf_token, lora=lora)
    elif mt == "llava":
        adapter = LlavaYesNoAdapter(model_id=model_id, device=device, cache_dir=cache_dir, dtype=dtype, hf_token=hf_token)
    elif mt == "llama":
        adapter = LlamaVisionYesNoAdapter(model_id=model_id, device=device, cache_dir=cache_dir, dtype=dtype,
                                          hf_token=hf_token, device_map_auto=llama_device_map_auto)
    elif mt == "pali":
        adapter = PaliGemmaYesNoAdapter(model_id=model_id, device=device, cache_dir=cache_dir, dtype=dtype,
                                       hf_token=hf_token, device_map_auto=llama_device_map_auto)
    elif mt in ("glm-thinking", "glm4.1v-thinking", "glm4v-thinking"):
        adapter = GlmThinkingYesNoAdapter(model_id=model_id, device=device, cache_dir=cache_dir, dtype=dtype,
                                          hf_token=hf_token, device_map_auto=glm_device_map_auto)
                                          

    else:
        raise ValueError(f"Unsupported --model-type '{model_type}'. Supported: qwen, llava, llama, glm-thinking")

    per_class_summaries = []
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    for cls in classes:
        cls_images_dir = images_dir_template.format(cls=cls)
        if not silent:
            print(f"\n=== Class: {cls} ===")
            print(f"Images: {cls_images_dir}")
        result = eval_one_class(adapter, cls_images_dir, cls, silent)
        rows = result["rows"]                               
        class_yes = sum(1 for r in rows if r["predicted_answer"] == "yes") 
        class_yes_prob_sum = sum(r["yes_probability"] for r in rows)        
        total_images += len(rows)                             
        total_yes += class_yes                                
        sum_yes_prob += class_yes_prob_sum                   

        summary = result["summary"]
        per_class_summaries.append({"class": cls, **summary})

        # Save per-class CSV + JSON summary if requested
        if save_dir:
            df = pd.DataFrame(result["rows"])
            (Path(save_dir) / f"{cls}_per_image.csv").parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(Path(save_dir) / f"{cls}_per_image.csv", index=False)
            with open(Path(save_dir) / f"{cls}_summary.json", "w") as f:
                json.dump(summary, f, indent=2)

    # Overall mean across classes with N>0
    valid = [s for s in per_class_summaries if s["len"] > 0]

    if valid:
        macro_mean_yes_share = sum(s["yes_share"] for s in valid) / len(valid)
        macro_mean_yes_prob  = sum(s["yes_probability_avg"] for s in valid) / len(valid)
    else:
        macro_mean_yes_share = 0.0
        macro_mean_yes_prob  = 0.0

    # MICRO (global across all images; this is the "accuracy based on all images")
    if total_images > 0:
        micro_global_yes_share = total_yes / total_images
        micro_global_yes_prob  = sum_yes_prob / total_images
    else:
        micro_global_yes_share = 0.0
        micro_global_yes_prob  = 0.0

    overall_macro = {
        "mean_yes_share": round(macro_mean_yes_share, 4),
        "mean_yes_probability_avg": round(macro_mean_yes_prob, 4),
        "num_classes_evaluated": len(valid),
        "total_images": total_images  # included for reference
    }

    overall_micro = {
        # This is the "accuracy based on all images"
        "global_yes_share": round(micro_global_yes_share, 4),
        "global_yes_probability_avg": round(micro_global_yes_prob, 4),
        "total_yes": int(total_yes),
        "total_images": int(total_images)
    }

    if not silent:
        print("\n==== Per-class summaries ====")
        for s in per_class_summaries:
            print(f"{s['class']:>12} | yes_share={s['yes_share']:.4f} | yes_prob_avg={s['yes_probability_avg']:.4f} | N={s['len']}")
        print("\n==== Overall (MACRO: mean across classes with N>0) ====")
        print(f"mean_yes_share={overall_macro['mean_yes_share']:.4f} | "
            f"mean_yes_probability_avg={overall_macro['mean_yes_probability_avg']:.4f} | "
            f"num_classes_evaluated={overall_macro['num_classes_evaluated']} | "
            f"total_images={overall_macro['total_images']}")
        print("\n==== Overall (MICRO: across all images) ====")
        print(f"global_yes_share={overall_micro['global_yes_share']:.4f} | "
            f"global_yes_probability_avg={overall_micro['global_yes_probability_avg']:.4f} | "
            f"total_yes={overall_micro['total_yes']} | "
            f"total_images={overall_micro['total_images']}")

    # Save aggregate files
    if save_dir:
        pd.DataFrame(per_class_summaries).to_csv(Path(save_dir) / "per_class_summary.csv", index=False)
        # Keep backward-compatible name for macro:
        with open(Path(save_dir) / "overall_mean.json", "w") as f:
            json.dump(overall_macro, f, indent=2)
        # And also save the micro (all-images) accuracy:
        with open(Path(save_dir) / "overall_micro.json", "w") as f:
            json.dump(overall_micro, f, indent=2)

    return {"per_class": per_class_summaries, "overall_macro": overall_macro, "overall_micro": overall_micro}
# -----------------------------
# Argparse
# -----------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate transferability (yes/no) per class for a given VLM.")
    ap.add_argument("--model-type", type=str, required=True,
                    help="Model family: qwen | llava | llama | glm-thinking")
    ap.add_argument("--model-id", type=str, required=True,
                    help=('HF model id, e.g. '
                          '"Qwen/Qwen2.5-VL-7B-Instruct", '
                          '"llava-hf/llava-v1.6-mistral-7b-hf", '
                          '"meta-llama/Llama-3.2-11B-Vision-Instruct", '
                          '"THUDM/GLM-4.1V-9B-Thinking"'))
    ap.add_argument("--images-dir-template", type=str, required=True,
                    help="Directory template containing {cls}, e.g., 'logs/.../{cls}/images'")
    ap.add_argument("--classes", type=str, required=True,
                    help='Comma-separated list or a path to a file with one class per line.')
    ap.add_argument("--device", type=str, default=None, help='Device override, e.g. "cuda:0" or "cpu".')
    ap.add_argument("--cache-dir", type=str, default=os.environ.get("HF_HOME", None),
                    help="HF cache dir (defaults to HF_HOME).")
    ap.add_argument("--dtype", type=str, default="float16",
                    choices=["float16", "bfloat16", "float32"], help="Model dtype.")
    ap.add_argument("--save-dir", type=str, default=None,
                    help="If set, saves per-image CSVs and summaries here.")
    ap.add_argument("--hf-token", type=str, default=None,
                    help="Optional Hugging Face token for gated models.")
    ap.add_argument("--llama-device-map-auto", action="store_true",
                    help="Use device_map='auto' for Llama Vision (helpful on multi-GPU).")
    ap.add_argument("--glm-device-map-auto", action="store_true",
                    help="Use device_map='auto' for GLM Thinking (recommended).")
    ap.add_argument("--silent", action="store_true", help="Reduce stdout logging.")
    ap.add_argument("--lora", action="store_true", help="Use LoRA weights for Qwen (if available).")
    return ap.parse_args()

def main():
    args = parse_args()
    classes = parse_classes(args.classes)
    _ = run_multi(
        model_type=args.model_type,
        model_id=args.model_id,
        images_dir_template=args.images_dir_template,
        classes=classes,
        device_str=args.device,
        cache_dir='cache',
        dtype=args.dtype,
        save_dir=args.save_dir,
        hf_token=args.hf_token,
        llama_device_map_auto=args.llama_device_map_auto,
        glm_device_map_auto=args.glm_device_map_auto,
        silent=args.silent,
        lora=args.lora
    )

if __name__ == "__main__":
    main()