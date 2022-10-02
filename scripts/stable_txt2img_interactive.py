import argparse, os, sys, glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm.auto import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid, save_image
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
import gradio as gr

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


class SDPipeline:
    def __init__(
        self,
        config,
        ckpt,
        plms: bool = False,
        ddim_eta: float = 0.0,
        downsampling_factor: int = 8,
        precision: str = "autocast",
        latent_channels: int = 4
    ):
        self.ddim_eta = ddim_eta
        self.downsampling_factor = downsampling_factor
        self.config = OmegaConf.load(config)
        self.precision_scope = autocast if precision=="autocast" else nullcontext
        self.latent_channels = latent_channels
        
        model = self.load_model_from_config(self.config, ckpt)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = model.to(self.device)

        if plms:
            self.sampler = PLMSSampler(self.model)
        else:
            self.sampler = DDIMSampler(self.model)

    @staticmethod
    def load_model_from_config(config, ckpt, verbose=False):
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
        model = instantiate_from_config(config.model)
        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m)
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)

        model.cuda()
        model.eval()
        return model
        

    def inference(self, prompt: str, num_samples: int, width:int=512, height:int=512, num_inference_steps:int=50, guidance_scale:float=7.5, seed=None):
        if seed is not None:
            seed_everything(seed)
        print(
          f"running prompt={prompt}, num_samples={num_samples}, width={width}, height={height}, "
          f"num_inference_steps={num_inference_steps}, guidance_scale={guidance_scale}"
        )
        shape = [self.latent_channels, height // self.downsampling_factor, width // self.downsampling_factor]
        start_code = None 
        #start_code = torch.randn([num_samples, self.latent_channels, height // self.downsampling_factor, width // self.downsampling_factor], device=self.device)
        result = []
        with torch.no_grad():
            with self.precision_scope("cuda"):
                with self.model.ema_scope():
                    tic = time.time()
                    uc = None
                    if guidance_scale != 1.0:
                        uc = self.model.get_learned_conditioning(num_samples * [""])
                    c = self.model.get_learned_conditioning(num_samples * [prompt])
                    print(dict(
                        S=num_inference_steps,
                        conditioning=c.shape,
                        batch_size=num_samples,
                        shape=shape,
                        verbose=False,
                        unconditional_guidance_scale=guidance_scale,
                        unconditional_conditioning=uc.shape,
                        eta=self.ddim_eta
                    ))
                    samples_ddim, _ = self.sampler.sample(
                        S=num_inference_steps,
                        conditioning=c,
                        batch_size=num_samples,
                        shape=shape,
                        verbose=False,
                        unconditional_guidance_scale=guidance_scale,
                        unconditional_conditioning=uc,
                        eta=self.ddim_eta,
                        x_T=start_code
                    )

                    x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                    for x_sample in x_samples_ddim:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        result.append(Image.fromarray(x_sample.astype(np.uint8)))

                    toc = time.time()
        return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--class_name",
        type=str,
        help="name of class in model",
        default="person"
    )
    parser.add_argument(
        "--instance_name",
        type=str,
        help="name of instance in model",
        default="sks"
    )
    parser.add_argument(
        "--example_prompt",
        type=str,
        help="example prompt",
        default="photo of sks person, digital painting"
    )

    opt = parser.parse_args()

    pipeline = SDPipeline(
        config=opt.config,
        ckpt=opt.ckpt,
        plms=opt.plms,
        ddim_eta=opt.ddim_eta,
        downsampling_factor=opt.f,
        precision=opt.precision,
        latent_channels=opt.C
    )
    
    def infer_call(prompt, samples, width, height, num_inference_steps, guidance):
        samples = int(samples)
        width = int(width)
        height = int(height)
        num_inference_steps = int(num_inference_steps)
        result = pipeline.inference(prompt, samples, width=width, height=height, num_inference_steps=num_inference_steps, guidance_scale=guidance)
        print("inference finished")
        return result
        
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                readme = gr.Markdown(f"""# Personalized SD Demo UI
In order to get images with the trained person, you want to include "{opt.instance_name} {opt.class_name}" into your prompt.

Some examples:
* photo of {opt.instance_name} {opt.class_name}
* digital painting of {opt.instance_name} {opt.class_name}

Prompting is hard, experiment and google around to find some hints. You can also draw some inspiration from other people, for example on [lexica.art](https://lexica.art/?).

**WARNING: This UI currently has an issue, that there is a hard timeout, if you generate images for longer than ~30 seconds, it will show an error, and never deliver your results. If you need larger images, or more images use the default CLI.**
""")
                prompt = gr.Textbox(label="Prompt", value=opt.example_prompt)
                run = gr.Button(value="Generate")
                with gr.Row():
                    num_samples = gr.Number(label="Number of Samples", value=4)
                    guidance_scale = gr.Number(label="Guidance Scale", value=7.5)
                with gr.Row():
                    height = gr.Number(label="Height", value=512)
                    width = gr.Number(label="Width", value=512)
                num_inference_steps = gr.Slider(label="Steps", value=50)
            with gr.Column(min_width=512):
                gallery = gr.Gallery()

        run.click(
            infer_call, 
            inputs=[prompt, num_samples, width, height, num_inference_steps, guidance_scale], 
            outputs=gallery
        )
        demo.queue(concurrency_count=1, status_update_rate=5)
        demo.launch(debug=True, share=True)
        


if __name__ == "__main__":
    main()
