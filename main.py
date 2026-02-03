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
from enhanced_text_representations import EnhancedTextRepresentations


def get_args():
    parser = argparse.ArgumentParser(description="Hallucination Adversarial Attack")
    parser.add_argument("--model_name", type=str, required=True, choices=["llava", "qwen", "llama", "glm4.1v-thinking"], help="Name of the victim model")
    parser.add_argument("--projector_path", type=str, required=True, help="Path to the projector checkpoint")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--cache_path", type=str, default="PathtoCache", help="Path to cache directory for HF models")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save generated images")
    parser.add_argument("--target_object", type=str, required=True, help="Target object for the attack")

    # Attack parameters
    parser.add_argument("--lr", type=float, default=1e-1, help="Learning rate for the attack")
    parser.add_argument("--num_steps", type=int, default=40, help="Number of optimization steps")
    parser.add_argument("--lambda_contrast", type=float, default=5.0, help="Encourages contrast against the target object in embedding space")
    parser.add_argument("--lambda_reg", type=float, default=5.0, help="Regularization term for the embedding space")
    parser.add_argument("--num_generation", type=int, default=4, help="Number of images to generate per instance")
    parser.add_argument("--threshold", type=float, default=0.99, help="Threshold for optimization")
    parser.add_argument("--OD_threshold", type=float, default=0.5, help="Threshold for object detector")
    parser.add_argument("--guidance_scale", type=float, default=10, help="Guidance scale for the diffusion model")
    parser.add_argument("--use_enhanced_text", type=bool, default=True, help="Use enhanced text representations for the target object")
    parser.add_argument("--sort_images", type=bool, default=False, help="Sort images based on similarity to target object using CLIP")
    parser.add_argument("--t", type=int, default=0, help="Number of timesteps for the diffusion model")
    parser.add_argument("--num_of_inference", type=int, default=50, help="Number of max inference steps for the diffusion model")
    

    # Others
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    return parser.parse_args()

def get_log_name(args):
    return f"{args.target_object}_lr={args.lr}_steps={args.num_steps}_threshold={args.threshold}_num_generation={args.num_generation}_guidance_scale={args.guidance_scale}_lambda_contrast={args.lambda_contrast}_lambda_reg={args.lambda_reg}_OD_threshold={args.OD_threshold}__sort={args.sort_images}_t={args.t}_num_inf={args.num_of_inference}_{''.join(os.path.basename(args.projector_path).split('.')[:-1])}"
    




def contains_obj_owlvit(image: Image.Image,processor_owl,model_owl, obj_hallucination, score_threshold=0.1):
    if model_owl is None or processor_owl is None:
        return False
    texts = [[obj_hallucination]] 
    inputs = processor_owl(text=texts, images=image, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model_owl(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])  # (H, W)
    results = processor_owl.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=score_threshold)[0]

    for score, label in zip(results["scores"], results["labels"]):
        if label == 0 and score > score_threshold:  
            return True
    return False

def attack(args):
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
    logger.info(subprocess.check_output("nvidia-smi", text=True))
    checkpoint = torch.load(args.projector_path, map_location='cpu')
    projector = TokenMLP(num_tokens=num_tokens, context_dim=context_dim, clip_dim=1024, hidden_dim=hidden_dim, target_dim=target_dim)
    projector.load_state_dict(checkpoint)
    projector.eval().cuda()

    logger.info(f"Projector loaded from {args.projector_path} with context_dim={context_dim}, hidden_dim={hidden_dim}, target_dim={target_dim}")
    logger.info(subprocess.check_output("nvidia-smi", text=True))
    
    logger.info("Loading Diffusion Model...")
    pipe = get_diffusion_model(args.cache_path)
    preprocess = get_diffusione_model_preprocess()

    if args.OD_threshold < 1:
        logger.info("Loading Owl-ViT Model...")
        model_owl, processor_owl = get_owl_model(args.cache_path)
        model_owl.eval().cuda()
    else:
        model_owl, processor_owl = None, None
        logger.info("Skipping loading Owl-ViT model as OD_threshold >= 1")

    logger.info("Loading Dataset...")
    dset = COCO(args.data_path, split='train', transform=(336, 336))
    present = [x  for cat in dset.get_all_supercategories() for x in dset.get_categories(cat) ]
    #present = [x  for x in dset.get_categories('vehicle') ]
    cat_spur_all = dset.get_imgIds_by_class(present_classes=present, absent_classes=[args.target_object])
    if args.sort_images:
        logger.info("Sorting images based on similarity to target object...")
        cat_spur_all, _ = load_and_compute_similarity(clip_model, cat_spur_all, args.target_object, embeddings_path="clip_embeddings.pt")
    else:
        target_object_supercat = None
        for cat in dset.get_all_supercategories():
            for c in dset.get_categories(cat):
                if c == args.target_object:
                    target_object_supercat = cat
                    break
        print("Target object supercat:", target_object_supercat)
        present = [x for x in dset.get_categories(target_object_supercat)]
        print("Present categories:", present)
        cat_spur_all = dset.get_imgIds_by_class(present_classes=present, absent_classes=[args.target_object])
        print("Befor",cat_spur_all[0])
        random.shuffle(cat_spur_all)    
        print("After",cat_spur_all[0])
        logger.info("Using images without sorting...")
    
    logger.info(f"Number of images without {args.target_object}: {len(cat_spur_all)}")

    prompts = get_prompt_templates()
    prompts = [p.format(obj=args.target_object) for p in prompts]
    logger.info("Using prompts: \n" + '\n'.join(prompts))

    yes_id = processor.tokenizer("Yes", add_special_tokens=False)["input_ids"][0]
    no_id = processor.tokenizer("No", add_special_tokens=False)["input_ids"][0]
    got_id = processor.tokenizer("Got", add_special_tokens=False)["input_ids"][0]
    think_id = processor.tokenizer("<think>", add_special_tokens=False)["input_ids"][0]
    text_enhancer = EnhancedTextRepresentations(clip_model, tokenizer, f'{args.data_path}/annotations/captions_train2017.json')
    compositional_embedding = text_enhancer.get_compositional_embeddings(args.target_object, 'cuda').detach()
    # text_tokens = tokenizer(args.target_object).to('cuda')

           

    logger.info(subprocess.check_output("nvidia-smi", text=True))
    asr = 0
    total_images_generated = 0
    total_images_optimized = 0
    for i, img_id in enumerate(cat_spur_all):
        if i < 200:
            continue
        if i >= 400:  # Limit to first 10 images for debugging
            break
        img_id = cat_spur_all[i]
        image, path = dset[int(img_id)]
        image.save(f"logs/attack/{args.model_name}/{get_log_name(args)}/original/{i}_{img_id}_original.png")
        logger.info(f"##### Processing image {i}/{len(cat_spur_all)} id={img_id}: {path} #####")

        prompt = random.choice(prompts)
        max_new_tokens = 512 if args.model_name == "glm4.1v-thinking" else 128
        output = get_vllm_output(model, processor, prompt, image.resize((756,756)), max_new_tokens=max_new_tokens)
        logger.info(f"Prompt: {prompt} \nOutput: {output}")
        if args.model_name == "glm4.1v-thinking":
            output = parse_glm_response(output)['answer'].strip()
        if output and output.lower().startswith("yes"):
            logger.info(f"Image {img_id} already contains {args.target_object}. Skipping...")
            continue
        if not output:
            logger.info(f"Model is thinking more than {max_new_tokens} tokens. Skipping...")
            continue
        total_images_optimized += 1
        clip_emb = get_clip_image_features([image], clip_model, clip_preprocess)
        clip_emb = nn.Parameter(clip_emb).cuda()
        clip_emb.requires_grad = True
        optimizer = torch.optim.AdamW([clip_emb], lr=args.lr)
        clip_emb_initial = clip_emb.clone().detach()
        
        pbar = tqdm(range(args.num_steps), desc=f"Image {i+1}/{len(cat_spur_all)}")
        vae = get_vae_features([image], pipe, preprocess)
        g = 0
        for step in pbar:
            prompt = random.choice(prompts)
            image_features = projector(clip_emb).half().squeeze(0)  # [num_tokens, target_dim]
            inputs = vllm_standard_preprocessing(processor, prompt, image)
            inputs = get_model_inputs(args.model_name, inputs, model, image_features)
            
            if args.model_name != "llama":
                inputs['input_ids'] = None

            logits = model(**inputs).logits.float()
            logits_step = logits[:, -1, :]
            probs = torch.softmax(logits_step, dim=-1)

            # Calculate loss / probabilities
            # Special-case GLM 4.1 Thinking: do two differentiable forwards.
            if args.model_name == "glm4.1v-thinking":
                # Step 0: next-token distribution (should strongly favor <think>)
                step0_probs = probs
                top0_id = int(torch.argmax(step0_probs[0]).item())
                top0_str = processor.tokenizer.decode([top0_id], skip_special_tokens=False)

                # Build a new inputs dict with '<think>' appended to the token embeddings
                inputs_embeds_0 = inputs['inputs_embeds']
                attn_mask_0 = inputs['attention_mask']
                embed_layer = model.model.get_input_embeddings()
                think_token_ids = torch.tensor([[think_id]], device=inputs_embeds_0.device)
                think_embed = embed_layer(think_token_ids).to(dtype=inputs_embeds_0.dtype)

                inputs_embeds_1 = torch.cat([inputs_embeds_0, think_embed], dim=1)
                attn_mask_1 = torch.cat([attn_mask_0, torch.ones_like(attn_mask_0[:, :1])], dim=1)

                inputs_step1 = dict(inputs)
                inputs_step1['inputs_embeds'] = inputs_embeds_1
                inputs_step1['attention_mask'] = attn_mask_1

                logits_1 = model(**inputs_step1).logits.float()
                logits_step1 = logits_1[:, -1, :]
                step1_probs = torch.softmax(logits_step1, dim=-1)

                prob_yes = step1_probs[0, yes_id]
                prob_no = step1_probs[0, no_id]
                prob_got = step1_probs[0, got_id]
                logger.info(f"[GLM-thinking] Step={step}: first token top='{top0_str.strip()}' p={step0_probs[0, top0_id].item():.2f}; second token Yes p={prob_yes.item():.2f}, No p={prob_no.item():.2f}, Got p={prob_got.item():.2f}")
                log_prob_yes = -torch.log(prob_yes + 1e-8)
            else:
                prob_yes = probs[0, yes_id]
                prob_no = probs[0, no_id]
                log_prob_yes = -torch.log(prob_yes + 1e-8)

            compositional_embedding_copy = compositional_embedding.clone().detach()
            #text_embedding = clip_model.encode_text(text_tokens).detach()
            sim1 = F.cosine_similarity(clip_emb, compositional_embedding_copy)
            sim2 = F.mse_loss(clip_emb, clip_emb_initial)
            

            loss = log_prob_yes + args.lambda_contrast * sim1 + args.lambda_reg * sim2

            pbar.set_postfix({"Loss": loss.item(), "Prob Yes": prob_yes.item(), "Prob No": prob_no.item()})
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # During success check, recompute with updated embeddings; handle GLM with two differentiable forwards
            if args.model_name == "glm4.1v-thinking":
                image_features_chk = projector(clip_emb).half().squeeze(0)
                inputs_chk = vllm_standard_preprocessing(processor, prompt, image)
                inputs_chk = get_model_inputs(args.model_name, inputs_chk, model, image_features_chk)
                if args.model_name != "llama":
                    inputs_chk['input_ids'] = None

                logits0 = model(**inputs_chk).logits.float()
                probs0 = torch.softmax(logits0[:, -1, :], dim=-1)
                top0_id = int(torch.argmax(probs0[0]).item())
                top0_str = processor.tokenizer.decode([top0_id], skip_special_tokens=False)

                inputs_embeds_0 = inputs_chk['inputs_embeds']
                attn_mask_0 = inputs_chk['attention_mask']
                embed_layer = model.model.get_input_embeddings()
                think_token_ids = torch.tensor([[think_id]], device=inputs_embeds_0.device)
                think_embed = embed_layer(think_token_ids).to(dtype=inputs_embeds_0.dtype)
                inputs_embeds_1 = torch.cat([inputs_embeds_0, think_embed], dim=1)
                attn_mask_1 = torch.cat([attn_mask_0, torch.ones_like(attn_mask_0[:, :1])], dim=1)
                inputs_step1 = dict(inputs_chk)
                inputs_step1['inputs_embeds'] = inputs_embeds_1
                inputs_step1['attention_mask'] = attn_mask_1

                logits1 = model(**inputs_step1).logits.float()
                probs1 = torch.softmax(logits1[:, -1, :], dim=-1)
                gen_yes_prob = probs1[0, yes_id].item()
                gen_no_prob = probs1[0, no_id].item()
                gen_got_prob = probs1[0, got_id].item()
                logger.info(f"[GLM-thinking] Generation check: first token top='{top0_str.strip()}', second token Yes p={gen_yes_prob:.2f}, No p={gen_no_prob:.2f}, Got p={gen_got_prob:.2f}")
            else:
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True)
                gen_probs = torch.softmax(generated_ids.scores[0], dim=-1)

                gen_yes_prob = gen_probs[0, yes_id].item()
                gen_no_prob = gen_probs[0, no_id].item()
                gen_got_prob = gen_probs[0, got_id].item()
            pbar.set_postfix({"Gen Yes Prob": gen_yes_prob, "Gen No Prob": gen_no_prob, "Gen Got Prob": gen_got_prob})
            
            if gen_yes_prob > args.threshold:
                attack_success = False
                if g < args.num_generation:
                    logger.info(f"Step={step}, Gen={g}: Generating image for path={path}, id={img_id}, Yes Prob={gen_yes_prob}")
                    generated = generate_image(pipe,vae,clip_emb.half(),t=args.t,num_inference_steps=args.num_of_inference, noise_level = 0, prompt=prompt, negative_prompt="low quality, ugly, unrealistic",guidance_scale=args.guidance_scale)
                    total_images_generated += 1
                    torch.cuda.empty_cache()

                    output = get_vllm_output(model, processor, prompt, generated, max_new_tokens=max_new_tokens)
                    od_flag = contains_obj_owlvit(generated,processor_owl,model_owl,args.target_object, args.OD_threshold)
                    temp = output
                    if args.model_name == "glm4.1v-thinking":
                        temp = parse_glm_response(output)['answer'].strip()
                    if temp and temp.lower().startswith("yes") and not od_flag:
                        logger.info(f"Attack successful for image {img_id} at step {step}")
                        logger.info(f"Prompt: {prompt}")
                        logger.info(f"Output: {output}")
                        output_path = f"logs/attack/{args.model_name}/{get_log_name(args)}/images/{i}_{img_id}_{step}_{g}.png"
                        generated.save(output_path)
                        logger.info(f"Saved generated image to {output_path}")
                        asr += 1
                        attack_success = True
                        break
                    else:
                        logger.info(f"Generated image did not trigger attack for image {img_id} at step {step}, generation {g}. Output: {output}")
                        if od_flag:
                            logger.info(f"Object detection confirmed presence of {args.target_object} in generated image.")
                        output_path = f"logs/attack/{args.model_name}/{get_log_name(args)}/failed/{i}_{img_id}_{step}_{g}.png"
                        #generated.save(output_path)
                        logger.info(f"Saved failed generated image to {output_path}")
                    g += 1 
                    if g >= args.num_generation:
                        logger.info(f"Reached maximum generations ({args.num_generation}) for image {img_id}. Stopping further generations.")
                        break
                

            if step == args.num_steps - 1:
                logger.info(f"Attack FAILED for image {img_id} after {args.num_steps} steps. Prob Yes: {gen_yes_prob}, Prob No: {gen_no_prob}, Prob Got: {gen_got_prob}")
    
    logger.info(f"Attack completed. Total images optimized: {total_images_optimized} Total images generated: {total_images_generated} Total Images Successful: {asr / total_images_optimized if total_images_optimized > 0 else 0:.2f} {total_images_generated/total_images_optimized}")






if __name__ == "__main__":
    args = get_args()

    logging_level = logging.DEBUG if args.debug else logging.INFO

    # create folder
    os.makedirs(f"logs", exist_ok=True)
    os.makedirs(f"logs/attack", exist_ok=True)
    os.makedirs(f"logs/attack/{args.model_name}", exist_ok=True)
    os.makedirs(f"logs/attack/{args.model_name}/{get_log_name(args)}", exist_ok=True)
    os.makedirs(f"logs/attack/{args.model_name}/{get_log_name(args)}/images", exist_ok=True)
    os.makedirs(f"logs/attack/{args.model_name}/{get_log_name(args)}/failed", exist_ok=True)
    os.makedirs(f"logs/attack/{args.model_name}/{get_log_name(args)}/original", exist_ok=True)

    logging.basicConfig(format="### %(message)s ###")

    logger = logging.getLogger("HallucinationAttack")
    logger.setLevel(level=logging_level)

    logger.addHandler(logging.FileHandler(f"logs/attack/{args.model_name}/{get_log_name(args)}/log.txt", mode='w'))

    # Setting Seed
    set_seed(args.seed)

    logger.info(get_log_name(args))
    logger.info(f"Arguments: {args}")

    attack(args)