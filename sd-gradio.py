import gradio as gr
import argparse
from lightning import StableDiffusionCLI
import os
import time
from threading import Timer
from gradio.themes.utils import colors, sizes

class StableDiffusionGradio:
    def __init__(self, args):
        self.cli = StableDiffusionCLI(
            model_type=args.model,
            float16=args.float16,
            quantize=args.quantize
        )
        self.cfg_weight = args.cfg if args.cfg is not None else 0  # Default value
        self.num_steps = args.steps if args.steps is not None else 1  # Default value
        self.negative_prompt = args.negative_prompt
        print("Model loaded. Ready to generate images.")
        print(f"Using CFG weight: {self.cfg_weight}, Steps: {self.num_steps}")
        if self.negative_prompt:
            print(f"Negative prompt: {self.negative_prompt}")
        
        self.last_prompt = ""
        self.last_negative_prompt = ""
        self.last_prompt_time = 0
        self.last_generation_time = 0  # New variable to track when the last image was generated

    def generate_image(self, prompt, negative_prompt, width, height, cfg, steps, seed):
        output_file = "gradio_output.png"
        self.cli.generate_image(
            prompt, 
            output=output_file, 
            width=width, 
            height=height, 
            seed=seed,
            cfg_weight=cfg,
            num_steps=steps,
            negative_prompt=negative_prompt
        )
        return output_file

    def prompt_changed(self, prompt, negative_prompt):
        self.last_prompt = prompt
        self.last_negative_prompt = negative_prompt
        self.last_prompt_time = time.time()
        return gr.update()

def main():
    parser = argparse.ArgumentParser(description="MLX Stable Diffusion Turbo Interface by u/mark-lord")
    parser.add_argument("--model", choices=["sd", "sdxl"], default="sdxl")
    parser.add_argument("--no-float16", dest="float16", action="store_false")
    parser.add_argument("--quantize", "-q", action="store_true")
    parser.add_argument("--cfg", type=float, default=0, help="CFG weight")
    parser.add_argument("--steps", type=int, default=1, help="Number of steps")
    parser.add_argument("--negative-prompt", type=str, default="", help="Negative prompt")
    args = parser.parse_args()

    sd_gradio = StableDiffusionGradio(args)

    # Define a custom theme
    theme = gr.themes.Soft(
        primary_hue="slate",
        secondary_hue="slate",
        neutral_hue="slate",
    ).set(
        button_primary_background_fill="*secondary_500",
        button_primary_background_fill_hover="*secondary_600",
        button_primary_text_color="white",
        block_title_text_weight="600",
        block_border_width="3px",
        block_shadow="*shadow_drop_lg",
        button_shadow="*shadow_drop_lg",
        button_large_padding="*spacing_md",
    )

    with gr.Blocks(theme=theme) as demo:
        gr.Markdown(
            """
            # 🎨 MLX-SD-Turbo Interactive Image Generator
            Create images in near-realtime with the power of MLX and SD-Turbo! By u/mark-lord
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the image you want to generate...",
                    lines=3
                )
                with gr.Row():
                    width_input = gr.Slider(256, 2048, value=512, step=64, label="Width")
                    height_input = gr.Slider(256, 2048, value=512, step=64, label="Height")
                with gr.Row():
                    steps_input = gr.Slider(1, 50, value=sd_gradio.num_steps, step=1, label="Steps")
                    seed_input = gr.Slider(1, 1000000, value=1, step=1, label="Seed")
                generate_button = gr.Button("Generate Image", variant="primary")
                
                with gr.Accordion("🔧 Advanced Settings", open=False):
                    gr.Markdown(
                        """
                        **Note:** CFG and negative prompt may not work as expected due to implementation 
                        issues, MLX limitations, or SD-turbo's response to these parameters.
                        """
                    )
                    cfg_input = gr.Slider(0, 20, value=sd_gradio.cfg_weight, step=0.5, label="CFG")
                    negative_prompt_input = gr.Textbox(
                        label="Negative Prompt",
                        placeholder="Describe what you don't want in the image...",
                        lines=2
                    )
            
            with gr.Column(scale=1):
                image_output = gr.Image(label="Generated Image", elem_id="output-image")

        def update_image(prompt, negative_prompt, width, height, cfg, steps, seed):
            current_time = time.time()
            
            # Check if the prompt has changed
            if prompt != sd_gradio.last_prompt or negative_prompt != sd_gradio.last_negative_prompt:
                sd_gradio.prompt_changed(prompt, negative_prompt)
                sd_gradio.last_generation_time = 0  # Reset the last generation time
                return gr.update()
            
            # Check if it's time to generate an image
            if current_time - sd_gradio.last_prompt_time >= 0.1 and sd_gradio.last_generation_time == 0:
                sd_gradio.last_generation_time = current_time  # Set the generation time
                return sd_gradio.generate_image(prompt, negative_prompt, width, height, cfg, steps, seed)
            
            return gr.update()

        generate_button.click(
            fn=sd_gradio.generate_image,
            inputs=[prompt_input, negative_prompt_input, width_input, height_input, cfg_input, steps_input, seed_input],
            outputs=image_output
        )

        prompt_input.change(
            fn=update_image,
            inputs=[prompt_input, negative_prompt_input, width_input, height_input, cfg_input, steps_input, seed_input],
            outputs=image_output,
            every=0.01
        )

        negative_prompt_input.change(
            fn=update_image,
            inputs=[prompt_input, negative_prompt_input, width_input, height_input, cfg_input, steps_input, seed_input],
            outputs=image_output,
            every=0.01
        )

    demo.queue().launch()

if __name__ == "__main__":
    main()
