import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, MllamaForConditionalGeneration, Glm4vForConditionalGeneration,PaliGemmaForConditionalGeneration
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import open_clip
import random
import numpy as np
from transformers.utils import is_torchdynamo_compiling
from diffusers import DiffusionPipeline
from torchvision import transforms
from transformers.image_processing_utils import select_best_resolution
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import re
from io import BytesIO
import base64
import torch.nn.functional as F
#from google.genai import types
#from google import genai
import time



def get_llava_image_features(images, model, processor, avg_pool=False, device="cuda"):
    """
    Extract mean-pooled LLaVA features from a batch of images.

    Args:
        images: List of PIL images
        model: LlavaForConditionalGeneration
        processor: LlavaProcessor
        device: "cuda" or "cpu"

    Returns:
        Tensor of shape [B, N_tokens, D] 
    """
    # Preprocess all images
    inputs = processor.image_processor(images=list(images), return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)      # [B, N_crops, 3, H, W]
    image_sizes = inputs["image_sizes"].to(device)        # [B, 2]

    vision_feature_layer = model.config.vision_feature_layer
    vision_feature_select_strategy = model.config.vision_feature_select_strategy

    image_features = model.model.get_image_features(
        pixel_values=pixel_values,
        image_sizes=image_sizes,
        vision_feature_layer=vision_feature_layer,
        vision_feature_select_strategy=vision_feature_select_strategy,
    )




    return torch.stack(image_features)  # shape: [B, D]

def get_llama_image_features(images, model, processor, device="cuda"):
    """
    Extract Llama image features from a batch of images.
    Args:
        images: List of PIL images
        model: LlamaForConditionalGeneration
        processor: LlamaProcessor
        device: "cuda" or "cpu"
    Returns:
        Tensor of shape [B, N_tokens, D]
    """
    inputs = processor.image_processor(images=list(images), return_tensors="pt")
    vision_outputs = model.vision_model(
        pixel_values=inputs['pixel_values'].to(device),
        aspect_ratio_ids=inputs['aspect_ratio_ids'].to(device),
        aspect_ratio_mask=inputs['aspect_ratio_mask'].to(device),
        )
    cross_attention_states = vision_outputs.last_hidden_state 
    cross_attention_states = model.model.multi_modal_projector(cross_attention_states).reshape(
    -1, cross_attention_states.shape[-2]*cross_attention_states.shape[-3], model.model.hidden_size)
    return  cross_attention_states

def get_qwen_image_features(images, model, processor, device="cuda"):
    """
    Extract Qwen image features from a batch of images.
    Args:
        images: List of PIL images
        model: Qwen2_5_VLForConditionalGeneration
        processor: AutoProcessor
        device: "cuda" or "cpu"
    Returns:
        Tensor of shape [B, N_tokens, D]
    """
    B = len(images)
    inputs = processor.image_processor(images=list(images), return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)      
    image_grid_thw = inputs["image_grid_thw"].to(device)        

    pixel_values = pixel_values.type(model.visual.dtype)
    image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)
    image_embeds = image_embeds.view(B, image_embeds.shape[0] // B, -1)
    return image_embeds


def get_glm_image_features(images, model, processor, device="cuda"):
    """
    Extract GLM image features from a batch of images.
    Args:
        images: List of PIL images
        model: Glm4vForConditionalGeneration
        processor: AutoProcessor
        device: "cuda" or "cpu"
    Returns:
        Tensor of shape [B, N_tokens, D]
    """
    B = len(images)
    inputs = processor.image_processor(images=list(images), return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)      
    image_grid_thw = inputs["image_grid_thw"].to(device)        

    pixel_values = pixel_values.type(model.model.visual.dtype)
    image_embeds = model.model.visual(pixel_values, grid_thw=image_grid_thw)
    image_embeds = image_embeds.view(B, image_embeds.shape[0] // B, -1)
    return image_embeds


def get_llava_inputs(inputs, model, image_features, device="cuda"):
    """
    Prepare inputs for LLaVA model using precomputed image features.

    Args:
        inputs: Preprocessed inputs from LlavaProcessor
        model: LlavaForConditionalGeneration
        _features: Precomputed image features

    Returns:
        Dict: Inputs ready for model.forward()
    """
    inputs = inputs.to(device)
    inputs_embeds = model.get_input_embeddings()(inputs['input_ids'])
    special_image_mask = (inputs['input_ids'] == model.config.image_token_index).unsqueeze(-1)
    special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
    if not is_torchdynamo_compiling() and inputs_embeds[special_image_mask].numel() != image_features.numel():
        n_image_tokens = (inputs['input_ids'] == model.config.image_token_index).sum()
        n_image_features = image_features.shape[0]
        raise ValueError(
            f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
        )
    image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
    inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

    inputs['inputs_embeds'] = inputs_embeds
    inputs['pixel_values'] = None
    

    return inputs


def get_qwen_inputs(inputs, model, image_embeds, device="cuda"):
    """
    Prepare inputs for Qwen model using precomputed image features.
    Args:
        inputs: Preprocessed inputs from AutoProcessor
        model: Qwen2_5_VLForConditionalGeneration
        image_embeds: Precomputed image features
    Returns:
        Dict: Inputs ready for model.forward()
    """

    inputs_embeds = model.get_input_embeddings()(inputs['input_ids'])
    n_image_tokens = (inputs['input_ids'] == model.config.image_token_id).sum().item()
    n_image_features = image_embeds.shape[0]
    if n_image_tokens != n_image_features:
        raise ValueError(
            f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
        )
    mask = inputs['input_ids'] == model.config.image_token_id
    mask_unsqueezed = mask.unsqueeze(-1)
    mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
    image_mask = mask_expanded.to(inputs_embeds.device)

    image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    inputs['inputs_embeds'] = inputs_embeds
    inputs['pixel_values'] = None
    

    return inputs

def get_glm_inputs(inputs, model, image_embeds, device="cuda"):
    """
    Prepare inputs for GLM model using precomputed image features.
    Args:
        inputs: Preprocessed inputs from AutoProcessor
        model: Glm4vForConditionalGeneration
        image_embeds: Precomputed image features
    Returns:
        Dict: Inputs ready for model.forward()
    """

    inputs_embeds = model.model.get_input_embeddings()(inputs['input_ids'])
    # image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
    image_mask, _ = model.model.get_placeholder_mask(inputs['input_ids'], inputs_embeds, image_features=image_embeds)
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    inputs['inputs_embeds'] = inputs_embeds
    inputs['pixel_values'] = None

    return inputs

def get_llama_inputs(inputs, model, image_features, max_num_tokens=1,device="cuda"):
    """
    Prepare inputs for LLamA model using precomputed image features.

    Args:
        inputs: Preprocessed inputs from LlamaProcessor
        model: LlamaForConditionalGeneration
        image_features: Precomputed image features

    Returns:
        Dict: Inputs ready for model.forward()
    """
    inputs['cross_attention_states'] = image_features
    cross_attention_mask = inputs['cross_attention_mask']
    inputs['cross_attention_mask'] = cross_attention_mask.repeat(1,1,1,max_num_tokens)
    inputs['pixel_values'] = None
   
    return inputs

def get_model_inputs(model_name, inputs, model, image_features, max_num_tokens=1, device="cuda"):
    if model_name == "llava":
        return get_llava_inputs(inputs, model, image_features, device=device)
    elif model_name == "qwen":
        return get_qwen_inputs(inputs, model, image_features, device=device)
    elif model_name == "llama":
        return get_llama_inputs(inputs, model, image_features, max_num_tokens, device=device)
    elif model_name == "glm4.1v-thinking":
        return get_glm_inputs(inputs, model, image_features, device=device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")



def get_model_image_features(model_name, images, model, processor, device="cuda"):
    if model_name == "llava":
        return get_llava_image_features(images, model, processor, device=device)
    elif model_name == "qwen":
        return get_qwen_image_features(images, model, processor, device=device)
    elif model_name == "llama":
         return get_llama_image_features(images, model, processor, device=device)
    elif model_name == "glm4.1v-thinking":
        return get_glm_image_features(images, model, processor, device=device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")



def llava_image_size_to_num_patches(image_size, grid_pinpoints, patch_size: int):
    """
    Calculate the number of patches after the preprocessing for images of any resolution.

    Args:
        image_size (`torch.LongTensor` or `np.ndarray` or `tuple[int, int]`):
            The size of the input image in the format (height, width). ?
        grid_pinpoints (`List`):
            A list containing possible resolutions. Each item in the list should be a tuple or list
            of the form `(height, width)`.
        patch_size (`int`):
            The size of each image patch.

    Returns:
        int: the number of patches
    """
    if not isinstance(grid_pinpoints, list):
        raise TypeError("grid_pinpoints should be a list of tuples or lists")

    # ! VERY IMPORTANT if image_size is tensor, must convert to into tuple, otherwise it will cause wrong calculate
    if not isinstance(image_size, (list, tuple)):
        if not isinstance(image_size, (torch.Tensor, np.ndarray)):
            raise TypeError(f"image_size invalid type {type(image_size)} with value {image_size}")
        image_size = image_size.tolist()

    best_resolution = select_best_resolution(image_size, grid_pinpoints)
    height, width = best_resolution
    num_patches = 0
    # consider change to ceil(height/patch_size)*ceil(width/patch_size) + 1
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            num_patches += 1
    # add the base patch
    num_patches += 1
    return num_patches

def get_clip_image_features(images, clip_model, clip_preprocess):
    """
    Extract CLIP features from a batch of images.

    Args:
        images: List of PIL images
        clip_model: CLIP model
        clip_preprocess: Preprocessing function for CLIP

    Returns:
        Tensor of shape [B, D]: one feature vector per image
    """
    # Preprocess all images
    inputs = torch.stack([clip_preprocess(image) for image in images]).to("cuda")  # [B, 3, H, W]

    image_features = clip_model.encode_image(inputs)  # [B, D]

    return image_features # shape: [B, D]

def get_model(model_name, cache_path):
    if model_name == "llava":
        model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
        processor = LlavaNextProcessor.from_pretrained(model_id, cache_dir=cache_path)
        model = LlavaNextForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, cache_dir=cache_path)
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
    elif model_name == "qwen":
        model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
        processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_path)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, dtype=torch.float16, low_cpu_mem_usage=True, cache_dir=cache_path)
    elif model_name == "llama":
        model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_path)
        model = MllamaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True, cache_dir=cache_path)
    elif model_name == "glm4.1v-thinking":
        model_id = "THUDM/GLM-4.1V-9B-Thinking"
        model = Glm4vForConditionalGeneration.from_pretrained(model_id, cache_dir=cache_path, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True)
        processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_path, use_fast=True)
    elif model_name == "pali":
        model_id = "google/paligemma-3b-mix-224"
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            cache_dir=cache_path,
            device_map='cuda',
        )
        processor = AutoProcessor.from_pretrained(model_id)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model, processor


def get_clip_model(cache_path):
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k', cache_dir=cache_path)
    return clip_model, clip_preprocess


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you use multi-GPU


def decode_latents(pipe, latents):

    latents = 1 / pipe.vae.config.scaling_factor * latents
    image = pipe.vae.decode(latents, return_dict=False)[0]
    image = (image).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    image = image.cpu().permute(0, 2, 3, 1).float().detach().numpy()
    return image


def get_diffusione_model_preprocess():
    preprocess = transforms.Compose([
        transforms.Resize((768, 768)),
        transforms.ToTensor(),  
    ])
    return preprocess
def generate_image(pipe, vae,cls_h,t=25,num_inference_steps=50, noise_level = 0, prompt='', negative_prompt=None,  do_classifier_free_guidance = True, guidance_scale = 7.5):

    with torch.no_grad():       
        pipe.scheduler.set_timesteps(num_inference_steps, device='cuda')
        timesteps = pipe.scheduler.timesteps
        timestep_idx = min(t, len(pipe.scheduler.timesteps) - 1)
        timestep = timesteps[timestep_idx]
        
        eps = torch.randn_like(vae)
        latents = pipe.scheduler.add_noise(vae, eps, timestep)
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt=prompt,
            device='cuda',
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            lora_scale=None,
        )
       
        
        
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        
        
        noise_level = torch.tensor([noise_level], device='cuda')
        image_embeds = pipe._encode_image(
            image=None,
            device='cuda',
            batch_size=1,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            noise_level=noise_level,
            generator=None,
            image_embeds=cls_h
        )
        extra_step_kwargs = pipe.prepare_extra_step_kwargs(None, None)
        
        
        for i, t in enumerate(pipe.progress_bar(timesteps[timestep_idx:])):
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        
            # predict the noise residual
            noise_pred = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                class_labels=image_embeds,
                cross_attention_kwargs=None,
                return_dict=False,
            )[0]
        
            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
            # compute the previous noisy sample x_t -> x_t-1
            latents = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]


        image = decode_latents(pipe,latents)
        image = pipe.image_processor.numpy_to_pil(image)
        
        return image[0]
        


def get_vae_features(images,pipe, preprocess):
    tensors = []
    for img in images:
        tensors.append(preprocess(img).half().to('cuda'))

    z_t = pipe.vae.encode(torch.stack(tensors)).latent_dist.sample() * pipe.vae.config.scaling_factor
    
    return z_t


def get_messages(prompt, image):
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt}
            ]}
    ]
    images = [image]
    return messages, images

def vllm_standard_preprocessing(processor, prompt, image, **processor_kwargs):
    messages, images = get_messages(prompt, image)
    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        text=[text_prompt], images=images, padding=True, return_tensors="pt",
        **processor_kwargs
    ).to(device='cuda')
    return inputs

def vllm_decoding(inputs, output_ids, processor) -> str:
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    return output_text


def get_vllm_output(model, processor, prompt, image, max_new_tokens=512):
    if model == 'gpt-4o':
        return
    inputs = vllm_standard_preprocessing(processor, prompt, image)
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    decoded = vllm_decoding(inputs, output_ids, processor)
    
    return decoded[0]



def get_num_tokens(model_name):
    if model_name == "llava":
        return 1176, 4096
    elif model_name == "qwen":
        return 144, 3584
    elif model_name == "glm4.1v-thinking":
        return 144, 4096
    elif model_name == "llama":
        return 4*1601, 4096
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def get_diffusion_model(cache_path):
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-unclip",
        torch_dtype=torch.float16,
        cache_dir=cache_path,
    ).to("cuda")
    return pipe

def get_owl_model(cache_path):
    model_id = "google/owlv2-base-patch16-ensemble"
    processor_owl = Owlv2Processor.from_pretrained(model_id, cache_dir=cache_path)
    model_owl = Owlv2ForObjectDetection.from_pretrained(model_id, cache_dir=cache_path,).to('cuda')
    return model_owl, processor_owl


def get_prompt_templates():
    templates = [
        "Do you see a {obj} in the image? Answer with 'Yes' or 'No'.",
        "Is there a {obj} here? Answer with 'Yes' or 'No'.",
        "Does the image contain a {obj}? Answer with 'Yes' or 'No'.",
        "Can you find a {obj} in this picture? Answer with 'Yes' or 'No'.",
        "Would you say there's a {obj} here? Answer with 'Yes' or 'No'.",
        "Is a {obj} present in this image? Answer with 'Yes' or 'No'.",
        ]
    return templates



def get_gpt_response(client, prompt, instructions=None):
    response = client.responses.create(
        model="gpt-4o",
        instructions=instructions,
        input=prompt
    )
    return response.output_text

def parse_glm_response(text):
    # extract <think>
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


def encode_image(image):
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    image.save('test.jpg')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    return image_base64


def get_gpt_output(client, prompt, image):
    base64_image = encode_image(image)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                        },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                ],
            }
        ],
    )
    return response.choices[0].message.content


def get_gemini_output(client, prompt, image_path, max_retries=5, wait_time=10):

    with open(image_path, 'rb') as f:
        image_bytes = f.read()

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type='image/jpeg',
                    ),
                    prompt
                ]
            )
            if response and getattr(response, "text", None):
                return response.text
            print(f"Empty response for image {image_path} on attempt {attempt+1}")
            return "[EMPTY RESPONSE]"

        except Exception as e:
            print(f"Attempt {attempt+1} failed with error: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("Max retries reached.")
                return "FAILED"
        
        client = genai.Client()

def get_aya_output(co, prompt, image_path):
    model = "c4ai-aya-vision-32b"
    with open(image_path, "rb") as img_file:
        base64_image_url = f"data:image/jpeg;base64,{base64.b64encode(img_file.read()).decode('utf-8')}"
    response = co.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": base64_image_url},
                    },
                ],
            }
        ],
        temperature=0.3,
    )
    return response.message.content[0].text



def load_and_compute_similarity(
    clip_model,
    exclude_indices,
    object_text,
    embeddings_path="clip_embeddings.pt",
    device="cuda"
):
    # Determine compute device from the CLIP model if not provided
    compute_device = next(clip_model.parameters()).device if device is None else torch.device(device)

    # Keep tensors on CPU for indexing/masking first, then move to compute_device
    checkpoint = torch.load(embeddings_path, map_location="cpu")
    clip_embeds = checkpoint["clip_embeds"]  # CPU tensor initially
    indices = checkpoint["indices"]
    if not torch.is_tensor(indices):
        indices = torch.tensor(indices)

    if exclude_indices is not None:
        mask = torch.isin(indices, torch.tensor(exclude_indices))
        clip_embeds = clip_embeds[mask]
        indices = indices[mask]

    tokenizer = open_clip.get_tokenizer('ViT-H-14')
    tokens = tokenizer(object_text).to(compute_device)
    with torch.no_grad():
        text_embed = clip_model.encode_text(tokens).to(compute_device)
        text_embed = F.normalize(text_embed, dim=-1)[0]

    clip_embeds = clip_embeds.to(compute_device)
    clip_embeds = F.normalize(clip_embeds, dim=-1)
    similarities = torch.matmul(clip_embeds, text_embed)
    sorted_similarities, sorted_idx = torch.sort(similarities, descending=True)
    sorted_indices = indices[sorted_idx.cpu()]

    return sorted_indices, sorted_similarities



