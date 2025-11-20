import gradio as gr
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch

model_id = "runwayml/stable-diffusion-v1-5"

# Load model in FP32 since CPU does NOT support float16
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    safety_checker=None  # optional: disable to avoid CPU slowdown
)

# Faster sampler
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

pipe.to("cpu")  # you don't have GPU so CPU is correct

def generate(prompt):
    image = pipe(prompt).images[0]
    return image

demo = gr.Interface(
    fn=generate,
    inputs=gr.Textbox(lines=2, placeholder="Describe anything..."),
    outputs=gr.Image(),
    title="AI Image Generator",
    description="Generate images using Stable Diffusion (CPU-friendly version)"
)

demo.launch(share=True)

