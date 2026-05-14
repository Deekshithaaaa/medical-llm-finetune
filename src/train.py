"""
QLoRA Fine-tuning of Mistral 7B on MedQA USMLE dataset.
Run on Kaggle with GPU T4 x2 enabled.
Results: Training loss 1.33 -> 0.91 over 2 epochs
"""
import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login

BASE_MODEL = "mistralai/Mistral-7B-v0.1"
HF_MODEL_ID = "deekshitha-urs/medical-mistral-7b-finetuned"
NUM_SAMPLES = 500
NUM_EPOCHS = 2

def format_example(example):
    options_str = "\n".join([f"{k}: {v}" for k, v in example["options"].items()])
    answer_text = example["options"][example["answer_idx"]]
    return {"text": (
        f"<s>[INST] You are a medical expert taking the USMLE exam. "
        f"Answer the following question by selecting the best option.\n\n"
        f"Question: {example['question']}\n\nOptions:\n{options_str}\n\n"
        f"Select the correct answer. [/INST]\n\n"
        f"The correct answer is {example['answer_idx']}: {answer_text}. </s>"
    )}

def main():
    login(token="YOUR_HF_TOKEN_HERE")

    raw = load_dataset("GBaker/MedQA-USMLE-4-options")
    data = [format_example(ex) for ex in raw["train"]][:NUM_SAMPLES]
    dataset = Dataset.from_list(data)
    print(f"Training on {len(dataset)} examples")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    model.enable_input_require_grads()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir="./medical-mistral",
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            bf16=True,
            logging_steps=50,
            warmup_steps=100,
            dataset_text_field="text",
            push_to_hub=True,
            hub_model_id=HF_MODEL_ID,
            hub_strategy="end",
        ),
    )
    trainer.train()
    print("Training complete!")

if __name__ == "__main__":
    main()
