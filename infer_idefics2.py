import requests
import torch
from PIL import Image
from io import BytesIO
import time

from transformers import AutoProcessor, AutoModelForCausalLM, AwqConfig
from transformers.image_utils import to_numpy_array, PILImageResampling, ChannelDimension
from transformers.image_transforms import resize, to_channel_dimension_format


MODE = "quantized"
DEVICE = "cuda:0"

# Load model
if MODE == "regular":
    model = AutoModelForCausalLM.from_pretrained("HuggingFaceM4/idefics2", torch_dtype=torch.float16, trust_remote_code=True).to(DEVICE)
elif MODE == "quantized":
    quant_path = "/admin/home/victor/code/idefics2-awq"
    model = AutoModelForCausalLM.from_pretrained(quant_path, trust_remote_code=True).to(DEVICE)
elif MODE == "fused_quantized":
    quant_path = "/admin/home/victor/code/idefics2-awq"
    quantization_config = AwqConfig(
        bits=4,
        fuse_max_seq_len=4096,
        modules_to_fuse={
            "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "mlp": ["gate_proj", "up_proj", "down_proj"],
            "layernorm": ["input_layernorm", "post_attention_layernorm", "norm"],
            "use_alibi": False,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "hidden_size": 4096,
        }
    )
    model = AutoModelForCausalLM.from_pretrained(quant_path, quantization_config=quantization_config, trust_remote_code=True).to(DEVICE)
else:
    raise ValueError("Unknown mode")

processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2")

image_seq_len = model.config.perceiver_config.resampler_n_latents
BOS_TOKEN = processor.tokenizer.bos_token
BAD_WORDS_IDS = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

def convert_to_rgb(image):
    # `image.convert("RGB")` would only work for .jpg images, as it creates a wrong background
    # for transparent images. The call to `alpha_composite` handles this case
    if image.mode == "RGB":
        return image

    image_rgba = image.convert("RGBA")
    background = Image.new("RGBA", image_rgba.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, image_rgba)
    alpha_composite = alpha_composite.convert("RGB")
    return alpha_composite


# The processor is the same as the Idefics processor except for the BILINEAR interpolation,
# so this is a hack in order to redefine ONLY the transform method
def custom_transform(x):
    x = convert_to_rgb(x)
    x = to_numpy_array(x)

    height, width = x.shape[:2]
    aspect_ratio = width / height
    if width >= height and width > 980:
        width = 980
        height = int(width / aspect_ratio)
    elif height > width and height > 980:
        height = 980
        width = int(height * aspect_ratio)
    width = max(width, 378)
    height = max(height, 378)

    x = resize(x, (height, width), resample=PILImageResampling.BILINEAR)
    x = processor.image_processor.rescale(x, scale=1 / 255)
    x = processor.image_processor.normalize(
        x,
        mean=processor.image_processor.image_mean,
        std=processor.image_processor.image_std
    )
    x = to_channel_dimension_format(x, ChannelDimension.FIRST)
    x = torch.tensor(x)
    return x


def download_image(url):
    try:
        # Send a GET request to the URL to download the image
        response = requests.get(url)
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Open the image using PIL
            image = Image.open(BytesIO(response.content))
            # Return the PIL image object
            return image
        else:
            print(f"Failed to download image. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# Create text token inputs
image_seq = '<image>' * image_seq_len
inputs = processor.tokenizer(
    [
        f"{BOS_TOKEN}<fake_token_around_image>{image_seq}<fake_token_around_image>In this image, we see",
        f"{BOS_TOKEN}bla bla<fake_token_around_image>{image_seq}<fake_token_around_image>{image_seq}<fake_token_around_image>",
    ],
    return_tensors="pt",
    add_special_tokens=False,
    padding=True,
)

# Create pixel inputs
image1 = download_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")
image2 = download_image("https://cdn.britannica.com/59/94459-050-DBA42467/Skyline-Chicago.jpg")
image3 = download_image("https://thumbs.dreamstime.com/b/golden-gate-bridge-san-francisco-purple-flowers-california-echium-candicans-36805947.jpg")
raw_images = [
    [image1],
    [image2, image3],
]
output_images = [
    [processor.image_processor(img, transform=custom_transform) for img in img_list]
    for img_list in raw_images
]
total_batch_size = len(output_images)
max_num_images = max([len(img_l) for img_l in output_images])
max_height = max([i.size(2) for img_l in output_images for i in img_l])
max_width = max([i.size(3) for img_l in output_images for i in img_l])
padded_image_tensor = torch.zeros(total_batch_size, max_num_images, 3, max_height, max_width)
padded_pixel_attention_masks = torch.zeros(
    total_batch_size, max_num_images, max_height, max_width, dtype=torch.bool
)
for batch_idx, img_l in enumerate(output_images):
    for img_idx, img in enumerate(img_l):
        im_height, im_width = img.size()[2:]
        padded_image_tensor[batch_idx, img_idx, :, :im_height, :im_width] = img
        padded_pixel_attention_masks[batch_idx, img_idx, :im_height, :im_width] = True

inputs["pixel_values"] = padded_image_tensor
inputs["pixel_attention_mask"] = padded_pixel_attention_masks
inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

start = time.time()
generated_ids = model.generate(**inputs, bad_words_ids=BAD_WORDS_IDS, max_new_tokens=500)
generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

print(generated_texts)
print(time.time() - start)
