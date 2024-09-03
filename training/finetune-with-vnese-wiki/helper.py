from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

eval_tokenizer = AutoTokenizer.from_pretrained(
    model_name, add_bos_token=True, trust_remote_code=True, use_fast=False
)
eval_tokenizer.pad_token = eval_tokenizer.eos_token


def gen(model, p, maxlen=100, sample=True):
    toks = eval_tokenizer(p, return_tensors="pt")

    res = model.generate(
        **toks.to("cuda"),
        max_new_tokens=maxlen,
        do_sample=sample,
        num_return_sequences=1,
        temperature=0.1,
        num_beams=1,
        top_p=0.95,
    ).to("cpu")

    print(eval_tokenizer.batch_decode(res, skip_special_tokens=True))

    return eval_tokenizer.batch_decode(res, skip_special_tokens=True)


def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"
