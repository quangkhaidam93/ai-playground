from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    GenerationConfig,
    set_seed,
)
from tqdm import tqdm
from trl import SFTTrainer, SFTConfig
import torch
import time
import pandas as pd
import numpy as np
from huggingface_hub import interpreter_login
import os
from helper import gen, model_name, print_number_of_trainable_model_parameters
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import transformers

# Quantization config
compute_dtype = getattr(torch, "float16")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

device_map = {"": 0}


def main():
    # Disable Weights & Biases

    # Login to Hugging Face Interpreter
    interpreter_login(new_session=False)

    # Load dataset
    vietnamese_wiki_dataset = "vietgpt/wikipedia_vi"
    dataset = load_dataset(vietnamese_wiki_dataset)

    # Load model

    original_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        quantization_config=bnb_config,
        use_auth_token=True,
        trust_remote_code=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_size="left",
        add_eos_token=True,
        add_bos_token=True,
        use_fast=False,
    )

    tokenizer.pad_token = tokenizer.eos_token

    original_model = prepare_model_for_kbit_training(original_model)

    config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj"],
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    original_model.gradient_checkpointing_enable()

    peft_model = get_peft_model(original_model, config)

    print(print_number_of_trainable_model_parameters(peft_model))

    out_dir = "./checkpoints"

    peft_training_args = SFTConfig(
        output_dir=out_dir,
        warmup_steps=1,
        dataset_text_field="text",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=500,
        learning_rate=2e-4,
        optim="paged_adamw_8bit",
        logging_steps=25,
        logging_dir="./logs",
        save_strategy="steps",
        save_steps=25,
        # evaluation_strategy="steps",
        # eval_steps=25,
        do_eval=False,
        gradient_checkpointing=True,
        report_to="none",
        overwrite_output_dir=True,
        group_by_length=True,
    )

    peft_model.config.use_cache = False

    peft_trainer = SFTTrainer(
        model=peft_model,
        train_dataset=dataset["train"],
        args=peft_training_args,
        data_collator=transformers.DataCollatorForLanguageModeling(
            tokenizer, mlm=False
        ),
    )

    peft_trainer.train()


if __name__ == "__main__":
    main()
