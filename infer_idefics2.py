import requests
import torch
from PIL import Image
from io import BytesIO
import time

from transformers import AutoProcessor, AutoModelForVision2Seq, AwqConfig


MODE = "fused_quantized"
DEVICE = "cuda:0"
PROCESSOR = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-tfrm-compatible")
BAD_WORDS_IDS = PROCESSOR.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids
EOS_WORDS_IDS = PROCESSOR.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids + [PROCESSOR.tokenizer.eos_token_id]

# Load model
if MODE == "regular":
    model = AutoModelForVision2Seq.from_pretrained(
        "HuggingFaceM4/idefics2-tfrm-compatible",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        _attn_implementation="flash_attention_2",
        revision="3dc93be345d64fb6b1c550a233fe87ddb36f183d",
    ).to(DEVICE)
elif MODE == "quantized":
    quant_path = "HuggingFaceM4/idefics2-tfrm-compatible-AWQ"
    model = AutoModelForVision2Seq.from_pretrained(
        quant_path,
        trust_remote_code=True
    ).to(DEVICE)
elif MODE == "fused_quantized":
    quant_path = "HuggingFaceM4/idefics2-tfrm-compatible-AWQ"
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
    model = AutoModelForVision2Seq.from_pretrained(quant_path, quantization_config=quantization_config, trust_remote_code=True).to(DEVICE)
else:
    raise ValueError("Unknown mode")


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


# Create inputs
image1 = download_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")
image2 = download_image("https://cdn.britannica.com/59/94459-050-DBA42467/Skyline-Chicago.jpg")

prompts = ["User:", image1, image2, "Describe these two images.<end_of_utterance>\n", "Assistant:"]
inputs = PROCESSOR(prompts)
inputs = {k: torch.tensor(v).to(DEVICE) for k, v in inputs.items()}

# Generate
start = time.time()
generated_ids = model.generate(**inputs, bad_words_ids=BAD_WORDS_IDS, max_new_tokens=500)
generated_texts = PROCESSOR.batch_decode(generated_ids, skip_special_tokens=True)

print(generated_texts)
print(time.time() - start)
