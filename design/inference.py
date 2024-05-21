from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    AutoencoderKL,
)
from PIL import Image
import torch
import numpy as np
import cv2

controlnet = ControlNetModel.from_pretrained(
    "TheMistoAI/MistoLine",
    torch_dtype=torch.float16,
    variant="fp16",
)

vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
)

pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    vae=vae,
    torch_dtype=torch.float16,
).to("cuda")


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


def create_canny_image(wireframe):
    image = np.array(wireframe)
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    return Image.fromarray(image)


def run(
    prompt,
    wireframe_img,
    batch_size,
    negative_prompt,
    # scheduler_name,
    # num_inference_steps,
    controlnet_conditioning_scale,
    # guidance_scale,
    # width,
    # height,
):
    # pipeline.scheduler = choose_scheduler(scheduler_name)
    canny = create_canny_image(wireframe_img)
    images = []

    for i in range(batch_size):
        generator = torch.Generator("cuda").manual_seed(i)

        images.append(
            pipeline(
                prompt=prompt,
                image=canny,
                generator=generator,
                negative_prompt=negative_prompt,
                # num_inference_steps=num_inference_steps,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                # guidance_scale=guidance_scale,
                # width=width,
                # height=height,
            ).images[0]
        )
        yield images
