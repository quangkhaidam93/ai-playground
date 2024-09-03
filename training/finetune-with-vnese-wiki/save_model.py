from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "./saved_model/ft_model",
    max_seq_length=1024,
)

model.save_pretrained_gguf("./saved_model/ft_vn_model", tokenizer=tokenizer)
