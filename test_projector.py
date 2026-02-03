import argparse
import os
from projector.projector import *
from data import COCO
from utils import *
from torch.utils.data import DataLoader, Subset
import torch
from tqdm import tqdm
import torch.nn.functional as F
import logging
import matplotlib.pyplot as plt
import re
import subprocess
import random


def get_args():
    parser = argparse.ArgumentParser(description="Hallucination Adversarial Attack")
    parser.add_argument("--model_name", type=str, required=True, choices=["llava", "qwen", "llama", "glm4.1v-thinking"], help="Name of the victim model")
    parser.add_argument("--projector_path", type=str, required=True, help="Path to the projector checkpoint")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--cache_path", type=str, default="PathtoCache", help="Path to cache directory for HF models")
    parser.add_argument("--target_objects", type=str, required=True, help="Target object for the attack")

    # Others
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    return parser.parse_args()

def get_log_name(args):
    return f"{args.target_objects}_{''.join(os.path.basename(args.projector_path).split('.')[:-1])}"

def get_probabilities(inputs,model):
    logits = model(**inputs).logits.float()
    logits_step = logits[:, -1, :]
    probs = torch.softmax(logits_step, dim=-1)

    return probs
def test(args):
    logger.info("Loading model...")
    model, processor = get_model(args.model_name, args.cache_path)
    model.eval().cuda()
    logger.info("Loading CLIP Model...")
    clip_model, clip_preprocess = get_clip_model(args.cache_path)
    clip_model.eval().cuda()
    tokenizer = open_clip.get_tokenizer('ViT-H-14')
    logger.info("Loading Projector...")
    num_tokens, target_dim = get_num_tokens(args.model_name)
    context_dim = re.search(r"context_dim=(\d+)", args.projector_path)
    hidden_dim = re.search(r"hidden_dim=(\d+)", args.projector_path)

    context_dim = int(context_dim.group(1)) if context_dim else 4096
    hidden_dim = int(hidden_dim.group(1)) if hidden_dim else 4096

    checkpoint = torch.load(args.projector_path, map_location='cpu')
    # projector = TokenMLPRes(context_dim=context_dim, hidden_dim1=hidden_dim, hidden_dim2=4096, hidden_dim3=4096, num_tokens=num_tokens, target_dim=target_dim).cuda()
    projector = TokenMLP(context_dim=context_dim, hidden_dim=hidden_dim, num_tokens=num_tokens, target_dim=target_dim).cuda()
    #projector = TokenMLPRes(num_tokens=num_tokens, context_dim=context_dim, clip_dim=1024, hidden_dim=hidden_dim, target_dim=target_dim)
    projector.load_state_dict(checkpoint)
    projector.eval().cuda()
    logger.info(f"Projector loaded from {args.projector_path} with context_dim={context_dim}, hidden_dim={hidden_dim}, target_dim={target_dim}")
    logger.info("Loading Dataset...")
    dset = COCO(args.data_path, split='train', transform=(336, 336))
    present = [x  for cat in dset.get_all_supercategories() for x in dset.get_categories(cat) ]
    accurecies_p_n = {}
    accurecies_p_y = {}
    accurecis_mllm_n = {}
    accurecis_mllm_y = {}
    probs_all_yes = {}
    prob_all_no = {}

    for target_object in args.target_objects.split(','):
        cat_spur_all = dset.get_imgIds_by_class(present_classes=present, absent_classes=[target_object])
        random.shuffle(cat_spur_all)
        logger.info(f"Number of images without {target_object}: {len(cat_spur_all)}")

        prompts = get_prompt_templates()
        prompts = [p.format(obj=target_object) for p in prompts]
        logger.info("Using prompts: \n" + '\n'.join(prompts))

        yes_id = processor.tokenizer("Yes", add_special_tokens=False)["input_ids"][0]
        no_id = processor.tokenizer("No", add_special_tokens=False)["input_ids"][0]

        correct_p = 0
        correct_mllm = 0
        probs_yes = 0
        probs_no = 0

        total = 0

        for i, img_id in enumerate(tqdm(cat_spur_all[:100])):  # Evaluate only first 100 images
            image, path = dset[img_id]
            prompt = random.choice(prompts)

            # Get projected tokens from CLIP
            clip_emb = get_clip_image_features([image], clip_model, clip_preprocess).clone().detach()
            image_features = projector(clip_emb).half().squeeze(0)
            

            # Prepare input and generate
            inputs = vllm_standard_preprocessing(processor, prompt, image)
            inputs = get_model_inputs(args.model_name, inputs, model, image_features)
            if args.model_name != "llama":
                inputs['input_ids'] = None
            #output = model(**inputs)
            probs = get_probabilities(inputs, model)
            prob_no = probs[0, no_id].item()
            prob_yes = probs[0, yes_id].item()
            probs_no += prob_no
            
            if prob_no > prob_yes:
                correct_p += 1
                decoded_p = 'No'
            else:
                decoded_p = 'Yes'

            decoded_m = get_vllm_output(model, processor, prompt, image, max_new_tokens=512)
            if args.model_name == "glm4.1v-thinking":
                decoded_m = parse_glm_response(decoded_m)['answer'].strip()
            if decoded_m and "no" in decoded_m.lower():
                correct_mllm += 1

            logger.info(f"[{i}] Prompt: '{prompt}' | Output projector: '{decoded_p}'| Probabilities: No={prob_no:.4f}, Yes={prob_yes:.4f} | Output MLLM: '{decoded_m}' ")
            
            total += 1

        acc_p = correct_p / total * 100
        logger.info(f"Object: {target_object} | Accuracy: {acc_p:.2f}% ({correct_p}/{total})")
        acc_mllm = correct_mllm / total * 100
        logger.info(f"Object: {target_object} | MLLM Accuracy: {acc_mllm:.2f}% ({correct_mllm}/{total})")
        accurecies_p_n[target_object] = acc_p
        accurecis_mllm_n[target_object] = acc_mllm

        correct_p = 0
        correct_mllm = 0
        total = 0

        cat_spur_all = dset.get_imgIds_by_class(present_classes=[target_object])
        logger.info(f"Number of images with {target_object}: {len(cat_spur_all)}")

        for i, img_id in enumerate(tqdm(cat_spur_all[:100])):  # Evaluate only first 100 images
            image, path = dset[img_id]
            prompt = random.choice(prompts)

            # Get projected tokens from CLIP
            clip_emb = get_clip_image_features([image], clip_model, clip_preprocess).clone().detach()
            image_features = projector(clip_emb).half().squeeze(0)

            # Prepare input and generate
            inputs = vllm_standard_preprocessing(processor, prompt, image)
            inputs = get_model_inputs(args.model_name, inputs, model, image_features)
            if args.model_name != "llama":
                inputs['input_ids'] = None
            #output = model(**inputs)
            probs = get_probabilities(inputs, model)
            prob_no = probs[0, no_id].item()
            prob_yes = probs[0, yes_id].item()
            probs_yes += prob_yes


            if prob_yes > prob_no:
                decoded_p = 'Yes'
                correct_p += 1
            else:
                decoded_p = 'No'
            

            decoded_m = get_vllm_output(model, processor, prompt, image, max_new_tokens=512)
            if args.model_name == "glm4.1v-thinking":
                decoded_m = parse_glm_response(decoded_m)['answer'].strip()
            
            if decoded_m and "yes" in decoded_m.lower():
                correct_mllm += 1

            logger.info(f"[{i}] Prompt: '{prompt}' | Output projector: '{decoded_p}' | | Probabilities: No={prob_no:.4f}, Yes={prob_yes:.4f} | Output MLLM: '{decoded_m}'")
            

            total += 1

        acc_p = correct_p / total * 100
        logger.info(f"Object: {target_object} | Accuracy: {acc_p:.2f}% ({correct_p}/{total})")
        acc_mllm = correct_mllm / total * 100
        logger.info(f"Object: {target_object} | MLLM Accuracy: {acc_mllm:.2f}% ({correct_mllm}/{total})")
        accurecies_p_y[target_object] = acc_p
        accurecis_mllm_y[target_object] = acc_mllm
        probs_all_yes[target_object] = probs_yes / total
        prob_all_no[target_object] = probs_no / total
    
    
    for target_object in args.target_objects.split(','):
        logger.info(f"Final Accuracies for Projector ({target_object} No): {accurecies_p_n[target_object]}")
        logger.info(f"Final Accuracies for Projector ({target_object} Yes): {accurecies_p_y[target_object]}")
        logger.info(f"Final Accuracies for Projector ({target_object}): {(accurecies_p_n[target_object] + accurecies_p_y[target_object]) / 2}")
        logger.info(f"Final Probabilities for Projector ({target_object} No): {prob_all_no[target_object]}")
        logger.info(f"Final Probabilities for Projector ({target_object} Yes): {probs_all_yes[target_object]}")
        logger.info(f"Final Accuracies for MLLM ({target_object} No): {accurecis_mllm_n[target_object]}")
        logger.info(f"Final Accuracies for MLLM ({target_object} Yes): {accurecis_mllm_y[target_object]}")
        logger.info(f"Final Accuracies for MLLM ({target_object}): {(accurecis_mllm_n[target_object] + accurecis_mllm_y[target_object]) / 2}")

    mean_probs_yes = sum(probs_all_yes.values()) / len(probs_all_yes)
    mean_probs_no = sum(prob_all_no.values()) / len(prob_all_no)
    logger.info(f"Mean Probabilities for Projector (Yes): {mean_probs_yes}")
    logger.info(f"Mean Probabilities for Projector (No): {mean_probs_no}")
    mean_acc_p_n = sum(accurecies_p_n.values()) / len(accurecies_p_n)
    mean_acc_p_y = sum(accurecies_p_y.values()) / len(accurecies_p_y)
    logger.info(f"Mean Accuracies for Projector (No): {mean_acc_p_n}")
    logger.info(f"Mean Accuracies for Projector (Yes): {mean_acc_p_y}")
    logger.info(f"Mean Accuracies for Projector: {(mean_acc_p_n + mean_acc_p_y) / 2}")  
    mean_acc_mllm_n = sum(accurecis_mllm_n.values()) / len(accurecis_mllm_n)
    mean_acc_mllm_y = sum(accurecis_mllm_y.values()) / len(accurecis_mllm_y)
    logger.info(f"Mean Accuracies for MLLM (No): {mean_acc_mllm_n}")
    logger.info(f"Mean Accuracies for MLLM (Yes): {mean_acc_mllm_y}")
    logger.info(f"Mean Accuracies for MLLM: {(mean_acc_mllm_n + mean_acc_mllm_y) / 2}")
    
    


                





if __name__ == "__main__":
    args = get_args()

    logging_level = logging.DEBUG if args.debug else logging.INFO

    # create folder
    os.makedirs(f"logs", exist_ok=True)
    os.makedirs(f"logs/projector", exist_ok=True)
    os.makedirs(f"logs/projector/test", exist_ok=True)
    os.makedirs(f"logs/projector/test/{args.model_name}/{get_log_name(args)}", exist_ok=True)

    logging.basicConfig(format="### %(message)s ###")

    logger = logging.getLogger("HallucinationAttack")
    logger.setLevel(level=logging_level)

    logger.addHandler(logging.FileHandler(f"logs/projector/test/{args.model_name}/{get_log_name(args)}/log.txt", mode='w'))

    # Setting Seed
    set_seed(args.seed)

    logger.info(get_log_name(args))
    logger.info(f"Arguments: {args}")

    test(args)