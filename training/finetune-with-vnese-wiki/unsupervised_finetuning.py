import hf_transfer
from huggingface_hub import interpreter_login
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# from helper import model_name
from utils.check_torch_version import check_torch_version
from unsloth import FastLanguageModel
import os
import wandb
from peft import prepare_model_for_kbit_training

os.environ["WANDB_LOG_MODEL"] = "false"  # Disable logging of model artifacts
# os.environ["WANDB_DISABLED"] = "true"

model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Login to Hugging Face Interpreter
# interpreter_login(new_session=False)
check_torch_version()

max_seq_length = 512
cache_dir = "./cache"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name,
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,
    cache_dir=cache_dir,
)

# Loading check ???
for n, p in model.named_parameters():
    if p.device.type == "meta":
        print(f"{n} is on meta")

print(model.config.max_position_embeddings)
print(model.config.eos_token_id)
print(tokenizer.bos_token)
print(tokenizer.eos_token)

tokenizer.eos_token = "<|eot_id|>"

# What is pad token -> model requires input has the same length -> pad token is used to fill the empty space
pad_token = "<|reserved_special_token_250|>"

if pad_token not in tokenizer.get_vocab():
    print(f"Pad token not in tokenizer, adding {pad_token} token")

    tokenizer.add_tokens(
        [pad_token],
        special_tokens=True,
    )

    tokenizer.pad_token = pad_token
else:
    print(
        f"{pad_token} token already in tokenizer and current pad token is {tokenizer.pad_token}"
    )

model.pad_token_id = tokenizer.pad_token_id
model.config.pad_token_id = tokenizer.pad_token_id

assert (
    model.pad_token_id == tokenizer.pad_token_id
), "Model and tokenizer pad token id must be the same"

model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=16)
print("Number of tokens in tokenizer: ", tokenizer.vocab_size)

# Test the chat template
messages = [
    {"role": "user", "content": "write a quick sort algorithm in go"},
    {"role": "assistant", "content": "Here is a quick sort algorithm in go"},
    {"role": "user", "content": "great."},
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenizer=False,
    add_generation_prompt=True,
)

# print(tokenizer.decode(inputs))

test_prompt = "TEST prompt"

tokens = tokenizer.encode(test_prompt, add_special_tokens=True)
# print(tokens)

# print(tokenizer.decode(tokens))

# Compare the vocabulary sizes
tokenizer_vocab_size = len(tokenizer)
model_embedding_size = model.get_input_embeddings().weight.size(0)

if tokenizer_vocab_size == model_embedding_size:
    print("Tokenizer and model vocabulary size are the same")
else:
    print(
        f"Tokenizer and model vocabulary size are different: {tokenizer_vocab_size} vs {model_embedding_size}"
    )

# print("Tokenizer vocab size: ", tokenizer_vocab_size)
# print("Model embedding size: ", model_embedding_size)

# print(model)

# model = prepare_model_for_kbit_training(model)

# print(model)

model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        # "o_proj",
        # "gate_proj",
        # "up_proj",
        # "down_proj",
    ],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=True,
    random_state=3407,
    use_rslora=True,
    modules_to_save=["embed_tokens", "lm_head"],
)

model.print_trainable_parameters()

# load the dataset
chosen_dataset_id = "vietgpt/wikipedia_vi"

dataset = load_dataset(chosen_dataset_id)

# print(dataset)

# Train the model
new_model_name = "khaidq/meta-llama-3.1-8b-finetune-with-vietnamese-wiki"
epochs = 1
scheduler_type = "constant"
batch_size = 1
grad_accum = 1
hf_username = "tobi931998"
save_dir = "./checkpoints"
completions_only = False
collator = None

trainer = SFTTrainer(
    tokenizer=tokenizer,
    model=model,
    train_dataset=dataset["train"],
    args=SFTConfig(
        # max_steps=1,  # for testing
        save_steps=1,
        save_total_limit=1,
        packing=True,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        logging_steps=10,
        num_train_epochs=epochs,
        output_dir=save_dir,
        do_eval=False,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        log_level="debug",
        bf16=False,
        fp16=True,
        lr_scheduler_type=scheduler_type,
        hub_private_repo=True,
        warmup_ratio=0.03,
        optim="adamw_torch",
        learning_rate=5e-5,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"user_reentrant": True},
        report_to="wandb",
    ),
    data_collator=collator,
)

tag = "all-data-rows"
run_name = tag

# wandb.init(project="fine-tune-meta-llama-with-vietnameses-wiki", name=run_name)

model.config.use_cache = False

trainer.train()
