# MLX-SD-Turbo Interactive Image Generator

![Demo Image](mainimage.png)

## Overview

MLX-SD-Turbo Interactive Image Generator is a user-friendly interface for generating images using the Stable Diffusion model, optimized with MLX for Apple Silicon. This tool allows users to create stunning images from text prompts with ease and speed.

## Features

- Interactive web interface powered by Gradio
- Text-to-image generation using Stable Diffusion
- Optimized performance on Apple Silicon with MLX
- Customizable image size, generation steps, and seed
- Advanced settings for fine-tuning (CFG, negative prompts)

## Installation

1. Clone the repository: `git clone https://github.com/mark-lord/mlx-sdturbo-webui.git`, then CD into it: `cd mlx-sdturbo-webui`

2. Install the required dependencies (recommend using a virtual environment, but not strictly necessary): `pip install -r requirements.txt`

3. Model weights *should* install on first-time set-up when you run the application. If they don't, then download the folder from `https://github.com/ml-explore/mlx-examples/tree/main/stable_diffusion` and run `txt2img.py`. It should download the SD-turbo weights. Then come back to this app.

If you get errors in your console, try doing pip **un**install -r requirements.txt. Then reinstall again.

## Usage

Run the application with: `python sd-gradio.py`

This will start the Gradio interface. Open the provided URL in your web browser to access the image generator.

### Basic Usage:

1. Enter a text prompt describing the image you want to generate.
2. Adjust the width and height sliders to set the image size.
3. Set the number of generation steps and seed value.
4. The script will auto-generate as you type, but you can also manually run it with the Generate Image button.

If you change the width, height or seed, you'll either need to edit your prompt to run it again, or click the Generate Image button.

The first few images on start-up are likely not to generate instantly; but once you've generated a few it tends to stabilise.

### Advanced Settings:

- CFG (Classifier Free Guidance) Scale: Adjust to control how closely the image follows the text prompt.
- Negative Prompt: Describe elements you want to avoid in the generated image.

Currently, these don't seem to actually change the image. Would recommend just ignoring these.

## Contributing

Contributions to improve MLX-SD-Turbo Interactive Image Generator are welcome. Please feel free to submit pull requests or open issues to suggest improvements or report bugs.

## License

Apache 2.0

## Acknowledgements

This project uses the Stable Diffusion model and is optimized with MLX for Apple Silicon. Special thanks to the creators of these technologies and the open-source community.
