from diffusers import (
    StableDiffusionXLPipeline,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
)
import torch

import os
from pathlib import Path

MODULE_PATH = os.path.join(Path(__file__).absolute().parents[0].absolute())

pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
).to("cuda")

pipeline.load_lora_weights(
    os.path.join(MODULE_PATH, "loras/LogoRedmond-LogoLoraForSDXL-V2"),
    adapter_name="logo",
)
pipeline.enable_attention_slicing()


def choose_scheduler(scheduler_name):
    match scheduler_name:
        case "EulerDiscreteScheduler":
            return EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
        case "EulerAncestralDiscreteScheduler":
            return EulerAncestralDiscreteScheduler.from_config(
                pipeline.scheduler.config
            )
        case _:
            raise Exception("Scheduler not supported")()


# def run(
#     prompt,
#     batch_size,
#     negative_prompt,
#     scheduler_name,
#     num_inference_steps,
#     guidance_scale,
#     width,
#     height,
# ):
#     generators = [torch.Generator("cuda").manual_seed(i) for i in range(batch_size)]
#     prompts = [prompt] * batch_size
#     pipeline.scheduler = choose_scheduler(scheduler_name)

#     return pipeline(
#         prompt=prompts,
#         generator=generators,
#         negative_prompt=negative_prompt,
#         num_inference_steps=num_inference_steps,
#         guidance_scale=guidance_scale,
#         width=width,
#         height=height,
#     ).images


def run(
    prompt,
    batch_size,
    negative_prompt,
    scheduler_name,
    num_inference_steps,
    guidance_scale,
    width,
    height,
):
    pipeline.scheduler = choose_scheduler(scheduler_name)
    images = []

    for i in range(batch_size):
        generator = torch.Generator("cuda").manual_seed(i)

        images.append(
            pipeline(
                prompt=prompt,
                generator=generator,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
            ).images[0]
        )
        yield images
