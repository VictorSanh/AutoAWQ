import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, AutoProcessor, AwqConfig

model_path = "HuggingFaceM4/idefics2"
quant_path = "/admin/home/victor/code/idefics2-awq"

quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version":"GEMM"}

# Load model
model = AutoAWQForCausalLM.from_pretrained(
    model_path, safetensors=True, torch_dtype=torch.float16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
processor = AutoProcessor.from_pretrained(model_path)

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# Modify config file
quantization_config = AwqConfig(
    bits=quant_config["w_bit"],
    group_size=quant_config["q_group_size"],
    zero_point=quant_config["zero_point"],
    version=quant_config["version"].lower(),
    modules_to_not_convert=["model.vision_model", "model.modality_projection", "model.perceiver_resampler"],
).to_dict()
model.model.config.quantization_config = quantization_config

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
processor.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')