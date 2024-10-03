import argparse
import mlx.core as mx
import mlx.nn as nn
from PIL import Image
import numpy as np
from tqdm import tqdm
from stable_diffusion import StableDiffusion, StableDiffusionXL
from prompt_toolkit import PromptSession
from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.containers import Window, VSplit, HSplit
from prompt_toolkit.layout.controls import FormattedTextControl, BufferControl
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.formatted_text import HTML
import asyncio
import threading
import os
import sys

class StableDiffusionCLI:
    def __init__(self, model_type="sdxl", float16=True, quantize=False):
        self.model_type = model_type
        self.float16 = float16
        self.quantize = quantize
        self.sd = self.load_model()

    def load_model(self):
        if self.model_type == "sdxl":
            sd = StableDiffusionXL("stabilityai/sdxl-turbo", float16=self.float16)
            if self.quantize:
                nn.quantize(sd.text_encoder_1, class_predicate=lambda _, m: isinstance(m, nn.Linear))
                nn.quantize(sd.text_encoder_2, class_predicate=lambda _, m: isinstance(m, nn.Linear))
                nn.quantize(sd.unet, group_size=32, bits=8)
        else:
            sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=self.float16)
            if self.quantize:
                nn.quantize(sd.text_encoder, class_predicate=lambda _, m: isinstance(m, nn.Linear))
                nn.quantize(sd.unet, group_size=32, bits=8)
        return sd

    def generate_image(self, prompt, output="out.png", width=512, height=512, seed=1, cfg_weight=None, num_steps=None, negative_prompt=""):
        # Set default values for cfg_weight and num_steps if not provided
        cfg_weight = cfg_weight if cfg_weight is not None else (0.0 if self.model_type == "sdxl" else 7.5)
        num_steps = num_steps if num_steps is not None else (1 if self.model_type == "sdxl" else 20)

        # Calculate latent size
        latent_height, latent_width = height // 8, width // 8

        # Generate the latent vectors using diffusion
        latents = self.sd.generate_latents(
            prompt,
            n_images=1,
            cfg_weight=cfg_weight,
            num_steps=num_steps,
            negative_text=negative_prompt,
            seed=seed,
            latent_size=(latent_height, latent_width)
        )
        for x_t in tqdm(latents, total=num_steps):
            mx.eval(x_t)

        # Decode the latents into an image
        decoded = self.sd.decode(x_t)
        mx.eval(decoded)

        # Post-process the image
        x = mx.pad(decoded, [(0, 0), (8, 8), (8, 8), (0, 0)])
        x = (x * 255).astype(mx.uint8)

        # Convert to PIL Image and resize
        im = Image.fromarray(np.array(x[0]))
        im = im.resize((width, height), Image.LANCZOS)

        # Save the image
        im.save(output)
        print(f"Image saved as {output}")

        return output

async def main():
    parser = argparse.ArgumentParser(description="Stable Diffusion CLI")
    parser.add_argument("--model", choices=["sd", "sdxl"], default="sdxl")
    parser.add_argument("--no-float16", dest="float16", action="store_false")
    parser.add_argument("--quantize", "-q", action="store_true")
    parser.add_argument("--cfg", type=float, help="CFG weight")
    parser.add_argument("--steps", type=int, help="Number of steps")
    parser.add_argument("--negative-prompt", type=str, default="", help="Negative prompt")
    parser.add_argument("--choose-output-name", action="store_true", help="Choose output filename for each generation")
    args = parser.parse_args()

    cli = StableDiffusionCLI(
        model_type=args.model, 
        float16=args.float16, 
        quantize=args.quantize
    )
    print("Model loaded. Ready to generate images.")
    print(f"Default CFG weight: {args.cfg or (0.0 if args.model == 'sdxl' else 7.5)}, Default Steps: {args.steps or (1 if args.model == 'sdxl' else 20)}")
    if args.negative_prompt:
        print(f"Default Negative prompt: {args.negative_prompt}")

    buffer = Buffer()
    kb = KeyBindings()

    @kb.add('c-c', eager=True)
    @kb.add('c-q', eager=True)
    def _(event):
        event.app.exit()

    prompt_control = FormattedTextControl(HTML('<b>Enter your prompt: </b>'))
    input_control = BufferControl(buffer=buffer)
    
    layout = Layout(
        HSplit([
            Window(prompt_control, height=1),
            Window(input_control)
        ])
    )

    app = Application(layout=layout, key_bindings=kb, full_screen=True)

    last_prompt = ""
    generation_lock = threading.Lock()

    def clear_console():
        os.system('cls' if os.name == 'nt' else 'clear')

    def refresh_display(prompt):
        clear_console()
        print("Enter your prompt:", end='')
        print(prompt, end='')
        sys.stdout.flush()

    def generate_image_thread(prompt, output):
        with generation_lock:
            cli.generate_image(
                prompt, 
                output=output, 
                cfg_weight=args.cfg,
                num_steps=args.steps,
                negative_prompt=args.negative_prompt
            )
            refresh_display(prompt)

    async def check_for_changes():
        nonlocal last_prompt
        while True:
            await asyncio.sleep(0.01)  # Check every 0.1 seconds
            current_prompt = buffer.text
            if current_prompt != last_prompt and current_prompt.strip():
                # Wait for 1 second and check again
                await asyncio.sleep(0.3)
                if buffer.text == current_prompt:
                    last_prompt = current_prompt
                    output = "out.png"
                    if args.choose_output_name:
                        output = input("Enter output filename (default: out.png): ") or "out.png"
                    threading.Thread(target=generate_image_thread, args=(current_prompt, output)).start()

    asyncio.create_task(check_for_changes())

    await app.run_async()

if __name__ == "__main__":
    asyncio.run(main())