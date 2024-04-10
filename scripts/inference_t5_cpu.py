
import argparse
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))
import os
import random
import torch
from torchvision.utils import save_image
from diffusion import IDDPM, DPMS, SASolverSampler
from diffusers.models import AutoencoderKL
from tools.download import find_model
from datetime import datetime
from typing import List, Union
import gradio as gr
import numpy as np
from gradio.components import Textbox, Image 
from transformers import T5EncoderModel, T5Tokenizer
import gc
from PIL import Image as PilImage

from diffusion.model.t5 import T5Embedder
from diffusion.model.utils import prepare_prompt_ar, resize_and_crop_tensor
from diffusion.model.nets import PixArtMS_XL_2, PixArt_XL_2
from torchvision.utils import _log_api_usage_once, make_grid
from diffusion.data.datasets.utils import *
from asset.examples import examples
from diffusion.utils.dist_utils import flush
import  piexif

from PIL.ExifTags import GPSTAGS

MAX_SEED = np.iinfo(np.int32).max


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', default=1024, type=int)
    parser.add_argument('--version', default='sigma', type=str)
    parser.add_argument('--model_path', default='output/pretrained_models/PixArt-XL-2-1024-MS.pth', type=str)
    parser.add_argument('--sdvae', action='store_true', help='sd vae')
    parser.add_argument(
        "--pipeline_load_from", default='output/pretrained_models/pixart_sigma_sdxlvae_T5_diffusers',
        type=str, help="Download for loading text_encoder, "
                       "tokenizer and vae from https://huggingface.co/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers"
    )
    parser.add_argument('--port', default=7788, type=int)

    return parser.parse_args()


@torch.no_grad()
def ndarr_image(tensor: Union[torch.Tensor, List[torch.Tensor]], **kwargs,) -> None:
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(save_image)
    grid = make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    return ndarr


def set_env(seed=0):
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    for _ in range(30):
        torch.randn(1, 4, args.image_size, args.image_size)


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


@torch.inference_mode()
def generate_img(prompt, negative_prompt, sampler, t5_load_Mode, sample_steps, scale, seed=0, randomize_seed=False):
    flush()
    gc.collect()
    torch.cuda.empty_cache()
    
    seed = int(randomize_seed_fn(seed, randomize_seed))
    set_env(seed)

    
    
    os.makedirs(f'output/demo/online_demo_prompts/', exist_ok=True)
        
    print(prompt)
    prompt_clean, prompt_show, hw, ar, custom_hw = prepare_prompt_ar(prompt, base_ratios, device=device)      # ar for aspect ratio
    prompt_clean = prompt_clean.strip()
    if isinstance(prompt_clean, str):
        prompts = [prompt_clean]

    # caption_token = tokenizer(prompts, max_length=max_sequence_length, padding="max_length", truncation=True, return_tensors="pt").to(device)
    caption_token = tokenizer(prompts, max_length=max_sequence_length, padding="max_length", truncation=True, return_tensors="pt").to("cpu")
    caption_embs = text_encoder(caption_token.input_ids, attention_mask=caption_token.attention_mask)[0]
    emb_masks = caption_token.attention_mask


    # caption_negative_token = tokenizer([negative_prompt], max_length=max_sequence_length, padding="max_length", truncation=True, return_tensors="pt").to(device)
    caption_negative_token = tokenizer([negative_prompt], max_length=max_sequence_length, padding="max_length", truncation=True, return_tensors="pt").to("cpu")
    caption_negative_embs = text_encoder(caption_negative_token.input_ids, attention_mask=caption_negative_token.attention_mask)[0]

    
    caption_embs = caption_embs[:, None]
    caption_negative_embs = caption_negative_embs[:, None]

    # move from cpu to cuda
    caption_embs = caption_embs.to(device)
    caption_negative_embs = caption_negative_embs.to(device)
    emb_masks = emb_masks.to(device)
    #null_y = null_caption_embs.repeat(len(prompts), 1, 1)[:, None]

    latent_size_h, latent_size_w = int(hw[0, 0]//8), int(hw[0, 1]//8)
    # Sample images:
    if sampler == 'iddpm':
        # Create sampling noise:
        n = len(prompts)
        z = torch.randn(n, 4, latent_size_h, latent_size_w, device=device).repeat(2, 1, 1, 1)
        model_kwargs = dict(y=torch.cat([caption_embs, caption_negative_embs]),
                            cfg_scale=scale, data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks)
        diffusion = IDDPM(str(sample_steps))
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True,
            device=device
        )
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    elif sampler == 'dpm-solver':
        # Create sampling noise:
        n = len(prompts)
        z = torch.randn(n, 4, latent_size_h, latent_size_w, device=device)
        model_kwargs = dict(data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks)
        dpm_solver = DPMS(model.forward_with_dpmsolver,
                          condition=caption_embs,
                          uncondition=caption_negative_embs,
                          cfg_scale=scale,
                          model_kwargs=model_kwargs)
        samples = dpm_solver.sample(
            z,
            steps=sample_steps,
            order=2,
            skip_type="time_uniform",
            method="multistep",
        )
    elif sampler == 'sa-solver':
        # Create sampling noise:
        n = len(prompts)
        model_kwargs = dict(data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks)
        sa_solver = SASolverSampler(model.forward_with_dpmsolver, device=device)
        samples = sa_solver.sample(
            S=sample_steps,
            batch_size=n,
            shape=(4, latent_size_h, latent_size_w),
            eta=1,
            conditioning=caption_embs,
            unconditional_conditioning=caption_negative_embs,
            unconditional_guidance_scale=scale,
            model_kwargs=model_kwargs,
        )[0]

    samples = samples.to(weight_dtype)
    samples = vae.decode(samples / vae.config.scaling_factor).sample
    samples = resize_and_crop_tensor(samples, custom_hw[0,1], custom_hw[0,0])
    
    
    suffix=random.randrange(1,999999)    
    im = PilImage.fromarray(ndarr_image(samples, normalize=True, value_range=(-1, 1)))
    
    filename ="output/demo"+str(suffix)+".jpg"
         
    im.save(filename)
    
    im = PilImage.open(filename)
    exif_dict = piexif.load(filename)
    
    exif_dict["ImageDescription"]= "test de description éè èé"
    exif_dict["seed"] = str(seed)
    exif_dict["prompt"] = prompt
    exif_dict["negative_prompt"] = negative_prompt
    exif_dict["sampler"] = sampler
    exif_dict["sample_steps"] = str(sample_steps)
    exif_dict["scale"] = str(scale)
    exif_bytes = piexif.dump(exif_dict)
    im.save("output/demo_EXIF_"+str(suffix)+".jpg", "jpeg", exif=exif_bytes)
    
    
    display_model_info = f'Model path: {args.model_path},\nBase image size: {args.image_size}, \nSampling Algo: {sampler}'
    return ndarr_image(samples, normalize=True, value_range=(-1, 1)), prompt_show, display_model_info, seed
    
    
def get_exif_data(image):
    """Returns a dictionary from the exif data of an PIL Image item. Also converts the GPS Tags"""
    exif_data = {}
    info = image.getexif()
    print ("FGP in exif, getexif: ["+str(info)+"]")
    
    if info:
        for tag, value in info.items():
            decoded = TAGS.get(tag, tag)
            if decoded == "GPSInfo":
                gps_data = {}
                for t in value:
                    sub_decoded = GPSTAGS.get(t, t)
                    gps_data[sub_decoded] = value[t]

                exif_data[decoded] = gps_data
            else:
                exif_data[decoded] = value
    return exif_data        
                
def importImageInfo(img):

    exif_data= get_exif_data(img)
    
    print ("fgp exif data: ["+str(exif_data)+"]")

    # print("FGP in importImageInfo")
    # print("FGP type of the image: ["+str(type(img))+"]")
    # print("FGP img.info: ["+str(img.info)+"]")
    
    # seed=img.info["seed"]
    # prompt=img.info["prompt"]
    # negative_prompt=img.info["negative_prompt"]
    # sampler=img.info["sampler"]
    # sample_steps=img.info["sample_steps"]
    # scale=img.info["scale"]
    # print("FGP seed:["+seed+"]")
    # print("FGP prompt:["+prompt+"]")
    # print("FGP negative_prompt:["+negative_prompt+"]")
    # print("FGP sampler:["+sampler+"]")
    # print("FGP sample_steps:["+sample_steps+"]")
    # print("FGP scale:["+scale+"]")
    
    return "empty"

def update(name):
    return f"Welcome to Gradio, {name}!"


if __name__ == '__main__':

    args = get_args()
    
    from diffusion.utils.logger import get_root_logger
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print ("FGP device: ["+device+"]")
    logger = get_root_logger()

    assert args.image_size in [256, 512, 1024, 2048], \
        "We only provide pre-trained models for 256x256, 512x512, 1024x1024 and 2048x2048 resolutions."
    pe_interpolation = {256: 0.5, 512: 1, 1024: 2, 2048: 4}
    latent_size = args.image_size // 8
    max_sequence_length = {"alpha": 120, "sigma": 300}[args.version]
    weight_dtype = torch.float16
    micro_condition = True if args.version == 'alpha' and args.image_size == 1024 else False
    if args.image_size in [512, 1024, 2048, 2880]:
        model = PixArtMS_XL_2(
            input_size=latent_size,
            pe_interpolation=pe_interpolation[args.image_size],
            micro_condition=micro_condition,
            model_max_length=max_sequence_length,
        ).to(device)
    else:
        model = PixArt_XL_2(
            input_size=latent_size,
            pe_interpolation=pe_interpolation[args.image_size],
            model_max_length=max_sequence_length,
        ).to(device)
    state_dict = find_model(args.model_path)
    if 'pos_embed' in state_dict['state_dict']:
        del state_dict['state_dict']['pos_embed']
    missing, unexpected = model.load_state_dict(state_dict['state_dict'], strict=False)
    logger.warning(f'Missing keys: {missing}')
    logger.warning(f'Unexpected keys: {unexpected}')
    model.to(weight_dtype)
    
     
    model.eval()
    base_ratios = eval(f'ASPECT_RATIO_{args.image_size}_TEST')

    if args.sdvae:
        # pixart-alpha vae link: https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/sd-vae-ft-ema
        vae = AutoencoderKL.from_pretrained("output/pretrained_models/sd-vae-ft-ema").to(device).to(weight_dtype)
    else:
        # pixart-Sigma vae link: https://huggingface.co/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers/tree/main/vae
        vae = AutoencoderKL.from_pretrained(f"{args.pipeline_load_from}/vae").to(device).to(weight_dtype)

    tokenizer = T5Tokenizer.from_pretrained(args.pipeline_load_from, subfolder="tokenizer")
    
    T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
    # text_encoder = T5EncoderModel.from_pretrained(args.pipeline_load_from, subfolder="text_encoder" ).to(device)
    text_encoder = T5EncoderModel.from_pretrained(args.pipeline_load_from, subfolder="text_encoder" ).to("cpu")
    #text_encoder.to_bettertransformer()

    #null_caption_token = tokenizer("", max_length=max_sequence_length, padding="max_length", truncation=True, return_tensors="pt").to(device)
    #null_caption_embs = text_encoder(null_caption_token.input_ids, attention_mask=null_caption_token.attention_mask)[0]

    title = f"""
        '' Unleashing your Creativity \n ''
        <div style='display: flex; align-items: center; justify-content: center; text-align: center;'>
            <img src='https://raw.githubusercontent.com/PixArt-alpha/PixArt-sigma-project/master/static/images/logo-sigma.png' style='width: 400px; height: auto; margin-right: 10px;' />
            {args.image_size}px
        </div>
    """
    DESCRIPTION = f"""# PixArt-Sigma {args.image_size}px
            ## If PixArt-Sigma is helpful, please help to ⭐ the [Github Repo](https://github.com/PixArt-alpha/PixArt-sigma) and recommend it to your friends ��'
            #### [PixArt-Sigma {args.image_size}px](https://github.com/PixArt-alpha/PixArt-sigma) is a transformer-based text-to-image diffusion system trained on text embeddings from T5. This demo uses the [PixArt-Sigma](https://huggingface.co/PixArt-alpha/PixArt-Sigma) checkpoint.
            #### English prompts ONLY; 提示词仅限英文
            """
    if not torch.cuda.is_available():
        DESCRIPTION += "\n<p>Running on CPU �� This demo does not work on CPU.</p>"

        
    with gr.Blocks() as demo:
        gr.HTML(value=title)
        gr.HTML(value=DESCRIPTION)
        with gr.Row() as firstRow:
            with gr.Column() as leftColumn:
                # prompt = gr.Textbox(placeholder="What is your name?")
                prompt = gr.Textbox(placeholder="Possitive prompt")
                negative_Prompt = gr.Textbox(placeholder="Negative prompt")
                sampler= gr.Radio(
                    choices=["iddpm", "dpm-solver", "sa-solver"],
                    label=f"Sampler",
                    interactive=True,
                    value='dpm-solver',
                )     
                t5_load_Mode= gr.Radio(
                    choices=["Default", "8bit", "4bit"],
                    label=f"Text Encoder Size",
                    interactive=True,
                    value='Default',
                )
                steps=gr.Slider(
                    label='Sample Steps',
                    minimum=1,
                    maximum=1000,
                    value=14,
                    step=1
                )
                scale=gr.Slider(
                    label='Guidance Scale',
                    minimum=0.1,
                    maximum=30.0,
                    value=4.5,
                    step=0.1
                )
                seed=gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=MAX_SEED,
                    step=1,
                    value=0,
                )
                randomize_seed=gr.Checkbox(label="Randomize seed", value=True)
            with gr.Column():
                outputImage=Image(type="numpy", label="Output Img", show_download_button=False, sources=None)
                cleanPrompt=Textbox(label="clean prompt")
                ModelInfo=Textbox(label="model info")
                usedSeed=gr.Slider(label='seed')
            with gr.Column():
                toLoadImage=Image(type="pil", label="Load previous Image")
                toLoadImage.upload(fn=importImageInfo, inputs=[toLoadImage],outputs=None)
                
        btn = gr.Button("Run")
        inputsFn = [prompt,negative_Prompt,sampler,steps,scale,seed,randomize_seed]
        btn.click(fn=generate_img, inputs=[prompt,negative_Prompt,sampler,t5_load_Mode,steps,scale,seed,randomize_seed], outputs=[outputImage,cleanPrompt,ModelInfo,usedSeed])
        #prompt, negative_prompt, sampler, sample_steps, scale, seed=0, randomize_seed=False, image=None):
    demo.launch(server_name="127.0.0.1", server_port=args.port, debug=True)
    
    
    