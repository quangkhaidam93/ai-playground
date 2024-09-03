from transformers import AutoTokenizer, AutoModelForCausalLM
from helper import model_name, gen
from main import bnb_config, device_map
from peft import PeftModel
import torch

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=device_map,
    quantization_config=bnb_config,
    use_auth_token=True,
    trust_remote_code=True,
)

ft_model = PeftModel.from_pretrained(
    base_model,
    "./checkpoints/checkpoint-500",
    torch_dtype=torch.float16,
    is_trainable=False,
)

ft_model.save_pretrained_gguf("./saved_model/ft_model")

# prompt = f"Instruct: Truyên Harry Potter do ai viết? Trả lời ngắn gọn.\nOutput:\n"

# peft_model_res = gen(
#     ft_model,
#     prompt,
#     1024,
# )
# peft_model_output = peft_model_res[0].split("\nOutput:\n")[1]


# prefix, success, result = peft_model_output.partition("###")

# print(result)

# print("\n\n")

# base_model_res = gen(
#     base_model,
#     prompt,
#     1024,
# )

# basemodel_output = base_model_res[0]

# print(basemodel_output)
