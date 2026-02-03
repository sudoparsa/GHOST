from utils import *
import argparse
import os
import logging
from data import COCO
from projector.projector import *
import re
import random
from openai import OpenAI


REF_MODEL_PROMPT = """You are an AI visual assistant that can analyze a single image. You receive five sentences, each describing the same image you are observing. In addition, specific object locations within the image are given, along with detailed coordinates. These coordinates are in the form of bounding boxes, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.

Using the provided caption and bounding box information, describe the scene in a detailed manner.

Instead of directly mentioning the bounding box coordinates, utilize this data to explain the scene using natural language. Include details like object counts, position of the objects, relative position between the objects.

When using the information from the caption and coordinates, directly explain the scene, and do not mention that the information source is the caption or the bounding box.  Always answer as if you are directly looking at the image.

{captions}

{bounding_boxes}
"""

JUDGE_SYS_PROMPT = "You are a helpful and precise assistant for checking the quality of the answer."

JUDGE_PROMPT = """
We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above. The user asks the question on observing an image. For your reference, the visual content in the image is represented with five descriptive sentences describing the same image and the bounding box coordinates of each object in the scene. These coordinates are in the form of bounding boxes, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.\nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall integer score on a scale of 1 to 10, where a higher score indicates better overall performance. Ties are allowed.

On the first line, print exactly:
Assistant1=<score_1> Assistant2=<score_2>
where <score_1> is the score for Assistant 1 and <score_2> is the score for Assistant 2.
In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.

[Visual Context]
{context}    
[Question]
{question}

[Assistant 1]
{answer_1}
[End of Assistant 1]

[Assistant 2]
{answer_2}
[End of Assistant 2]
"""



def get_args():
    parser = argparse.ArgumentParser(description="Evaluate the projector with GPT-as-a-Judge")
    parser.add_argument("--projector_path", type=str, required=True, help="Path to the projector model")
    parser.add_argument("--model_name", type=str, required=True, choices=["llava", "qwen", "glm4.1v-thinking"], help="Name of the model")
    parser.add_argument("--coco_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--cache_path", type=str, help="Path to cache directory for HF models")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for more verbose logging")
    parser.add_argument("--K", type=int, default=30, help="Number of images to sample for evaluation")
    parser.add_argument("--P", type=int, default=2, help="Number of prompts to use for each image")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum number of new tokens to generate")
    return parser.parse_args()


def get_log_name(args):
    return f"{args.model_name}_K={args.K}_P={args.P}_{''.join(os.path.basename(args.projector_path).split('.')[:-1])}_maxnew={args.max_new_tokens}"


def get_eval_prompts():
    desc_prompts = [
    "Describe the image concisely.",
    "Provide a brief description of the given image.",
    "Offer a succinct explanation of the picture presented.",
    "Summarize the visual content of the image.",
    "Give a short and clear explanation of the subsequent image.",
    "Share a concise interpretation of the image provided.",
    "Present a compact description of the photo’s key features.",
    "Relay a brief, clear account of the picture shown.",
    "Render a clear and concise summary of the photo.",
    "Write a terse but informative summary of the picture.",
    "Create a compact narrative representing the image presented.",
    "Describe the following image in detail",
    "Provide a detailed description of the given image",
    "Give an elaborate explanation of the image you see",
    "Share a comprehensive rundown of the presented image",
    "Offer a thorough analysis of the image",
    "Explain the various aspects of the image before you",
    "Clarify the contents of the displayed image with great detail",
    "Characterize the image using a well-detailed description",
    "Break down the elements of the image in a detailed manner",
    "Walk through the important details of the image",
    "Portray the image with a rich, descriptive narrative",
    "Narrate the contents of the image with precision",
    "Analyze the image in a comprehensive and detailed manner",
    "Illustrate the image through a descriptive explanation",
    "Examine the image closely and share its details",
    "Write an exhaustive depiction of the given image",
    ]
    return desc_prompts


def parse_judge_response(response):
    try:
        scores_line = response.split('\n')[0]
        scores = re.findall(r'Assistant1=(\d+)\s+Assistant2=(\d+)', scores_line)[0]
        return int(scores[0]), int(scores[1])
    except Exception as e:
        logger.error(f"Error parsing judge response: {e}")
        return None, None

def main(args):
    logger.info("Loading the dataset...")
    dset = COCO(args.coco_path, split='val', transform=(336, 336))
    img_ids = dset.get_imgIds()
    random_ids = random.sample(img_ids, args.K)
    logger.info(f"Randomly selected {len(random_ids)} images for evaluation.")

    client = OpenAI()

    logger.info("Loading the model...")
    model, processor = get_model(args.model_name, args.cache_path)
    model.eval().to("cuda")
    clip_model, clip_preprocess = get_clip_model(args.cache_path)
    clip_model.eval().cuda()

    logger.info("Loading the projector...")
    num_tokens, target_dim = get_num_tokens(args.model_name)
    context_dim = re.search(r"context_dim=(\d+)", args.projector_path)
    hidden_dim = re.search(r"hidden_dim=(\d+)", args.projector_path)

    context_dim = int(context_dim.group(1)) if context_dim else 4096
    hidden_dim = int(hidden_dim.group(1)) if hidden_dim else 512


    checkpoint = torch.load(args.projector_path, map_location='cpu')
    projector = TokenMLP(num_tokens=num_tokens, context_dim=context_dim, clip_dim=1024, hidden_dim=hidden_dim, target_dim=target_dim)
    # projector = TokenMLPRes(num_tokens=num_tokens, context_dim=context_dim, clip_dim=1024, hidden_dim1=hidden_dim, target_dim=target_dim)
    projector.load_state_dict(checkpoint)
    projector.eval().cuda()

    total_rel_score_base = 0
    total_rel_score_proj = 0
    for idx in random_ids:
        image, path = dset[idx]
        logger.info(f"Processing image {idx} from {path}")
        captions = "\n".join(dset.get_captions(idx))
        bounding_boxes = dset.get_bounding_boxes(idx)
        bounding_boxes_str = "\n".join([f"{box[0]}: {box[1]}" for box in bounding_boxes])
        ref_model_prompt = REF_MODEL_PROMPT.format(captions=captions, bounding_boxes=bounding_boxes_str)
        logger.info(f"Prompt for image {idx}:\n{ref_model_prompt}\n")
        gpt_desc = get_gpt_response(client, ref_model_prompt)
        logger.info(f"GPT-4o Description for image {idx}:\n{gpt_desc}\n")

        desc_prompts = get_eval_prompts()
        desc_prompts = random.sample(desc_prompts, k=args.P)

        rel_score_base = 0
        rel_score_proj = 0
        for desc_prompt in desc_prompts:
            logger.info(f"Using prompt for image {idx}: {desc_prompt}")

            base_output = get_vllm_output(model, processor, desc_prompt, image, max_new_tokens=args.max_new_tokens)

            logger.info(f"Model Output:\n{base_output}\n")

            if args.model_name == "glm4.1v-thinking":
                base_output = parse_glm_response(base_output)['answer'].strip()

            # Get image features
            clip_emb = get_clip_image_features([image], clip_model, clip_preprocess)
            image_features = projector(clip_emb).half().squeeze(0)
            inputs = vllm_standard_preprocessing(processor, desc_prompt, image)
            inputs = get_model_inputs(args.model_name, inputs, model, image_features)
            output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
            proj_output = vllm_decoding(inputs, output_ids, processor)

            logger.info(f"Projector Output:\n{proj_output}\n")

            if args.model_name == "glm4.1v-thinking":
                proj_output = parse_glm_response(proj_output[0])['answer']
            else:
                proj_output = proj_output[0]

            # Judge
            judge_prompt = JUDGE_PROMPT.format(context=captions+"\n"+bounding_boxes_str, question=desc_prompt, answer_1=base_output, answer_2=gpt_desc)
            logger.info(f"Judge Prompt Base:\n{judge_prompt}\n")
            judge_response = get_gpt_response(client, judge_prompt, instructions=JUDGE_SYS_PROMPT)
            logger.info(f"Judge Response Base:\n{judge_response}\n")
            score_base_ref, score_ref_base = parse_judge_response(judge_response)

            judge_prompt = JUDGE_PROMPT.format(context=captions+"\n"+bounding_boxes_str, question=desc_prompt, answer_1=proj_output, answer_2=gpt_desc)
            logger.info(f"Judge Prompt Proj:\n{judge_prompt}\n")
            judge_response = get_gpt_response(client, judge_prompt, instructions=JUDGE_SYS_PROMPT)
            logger.info(f"Judge Response Proj:\n{judge_response}\n")
            score_proj_ref, score_ref_proj = parse_judge_response(judge_response)

            rel_score_base_ref = score_base_ref / score_ref_base * 100
            rel_score_proj_ref = score_proj_ref / score_ref_proj * 100
            rel_score_base += rel_score_base_ref
            rel_score_proj += rel_score_proj_ref

        rel_score_base /= args.P
        rel_score_proj /= args.P
        logger.info(f"Base Relative Score: {rel_score_base}")    
        logger.info(f"Proj Relative Score: {rel_score_proj}")
        total_rel_score_base += rel_score_base
        total_rel_score_proj += rel_score_proj

    total_rel_score_base /= args.K
    total_rel_score_proj /= args.K
    logger.info(f"Total Base Relative Score: {total_rel_score_base}")
    logger.info(f"Total Proj Relative Score: {total_rel_score_proj}")
    logger.info(f"Total Proj Score: {total_rel_score_proj / total_rel_score_base * 100}") 



if __name__ == "__main__":
    args = get_args()

    logging_level = logging.DEBUG if args.debug else logging.INFO

    # create folder
    os.makedirs(f"logs", exist_ok=True)
    os.makedirs(f"logs/projector", exist_ok=True)
    os.makedirs(f"logs/projector/eval", exist_ok=True)

    logging.basicConfig(format="### %(message)s ###")

    logger = logging.getLogger("EvalProjector")
    logger.setLevel(level=logging_level)

    logger.addHandler(logging.FileHandler(f"logs/projector/eval/{get_log_name(args)}.txt", mode='w'))

    # Setting Seed
    set_seed(args.seed)

    logger.info(get_log_name(args))
    logger.info(f"Arguments: {args}")

    main(args)