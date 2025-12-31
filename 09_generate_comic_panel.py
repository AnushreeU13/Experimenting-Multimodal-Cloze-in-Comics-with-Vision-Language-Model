import json
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw, ImageFont
import os

def generate_comic_panel():
    # 1. Load Parameters
    print("Loading parameters...")
    try:
        with open("input.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: input.json not found.")
        return

    image_id = data.get("image_id", "output_image")
    prompt = data.get("image_generation_prompt", "")
    text_overlay = data.get("text_overlay", "")

    if not prompt:
        print("Error: No prompt provided.")
        return

    print(f"Generating image for ID: {image_id}")
    print(f"Prompt: {prompt}")

    # 2. Setup Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 3. Load Model
    # Using v1-5 as planned. 
    model_id = "runwayml/stable-diffusion-v1-5"
    print(f"Loading model: {model_id}")
    
    # Check if we can use float16 (faster on GPU)
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=dtype,
        use_safetensors=True
    )
    pipe = pipe.to(device)

    # 4. Generate Image
    print("Generating image (this may take a while)...")
    # Using 50 steps by default
    image = pipe(prompt).images[0]

    # 5. Overlay Text
    if text_overlay:
        print(f"Overlaying text: {text_overlay}")
        draw = ImageDraw.Draw(image)
        
        # Load a font - try generic defaults
        try:
            # Arial for Windows, DejaVuSans for Linux usually
            font_name = "arial.ttf" 
            font = ImageFont.truetype(font_name, 20)
        except OSError:
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 20)
            except OSError:
                print("Warning: Custom fonts not found, using default.")
                font = ImageFont.load_default()

        # Calculation for text box
        W, H = image.size
        # Estimate text size
        bbox = draw.textbbox((0, 0), text_overlay, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        
        padding = 10
        box_h = text_h + (padding * 2)
        
        # Create a semi-transparent overlay at the bottom
        # Since PIL doesn't support alpha on draw properties easily without layers,
        # we'll composite.
        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        draw_ov = ImageDraw.Draw(overlay)
        
        # Bottom rectangle (white background with some opacity)
        rect_y0 = H - box_h - 20
        rect_y1 = H - 20
        # White bubble with slightly rounded corners feeling? Just rect for now.
        draw_ov.rectangle(
            [(20, rect_y0), (W - 20, rect_y1)], 
            fill=(255, 255, 255, 220), 
            outline=(0, 0, 0, 255)
        )
        
        # Draw text centered in the box
        text_x = (W - text_w) / 2
        text_y = rect_y0 + padding
        
        draw_ov.text((text_x, text_y), text_overlay, fill=(0, 0, 0, 255), font=font)
        
        image = Image.alpha_composite(image.convert("RGBA"), overlay)

    # 6. Save Image
    output_filename = f"{image_id}.png"
    image.save(output_filename)
    print(f"Successfully saved to {output_filename}")

if __name__ == "__main__":
    generate_comic_panel()
