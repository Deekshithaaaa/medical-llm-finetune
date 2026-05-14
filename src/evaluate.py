"""
Evaluation script — compares base vs fine-tuned Mistral 7B on MedQA USMLE.
Results: Base Mistral 7B: 46% | Fine-tuned: 54% | Improvement: +8%
Run on Kaggle with GPU T4 x2 enabled.
"""
import torch, re, json, gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset

BASE_MODEL = "mistralai/Mistral-7B-v0.1"
FINETUNED_MODEL = "deekshitha-urs/medical-mistral-7b-finetuned"
NUM_TEST = 50

def generate_answer(model, tokenizer, question, options):
    options_str = "\n".join([f"{k}: {v}" for k, v in options.items()])
    prompt = (
        f"[INST] You are a medical expert. "
        f"Question: {question}\n\nOptions:\n{options_str}\n\n"
        f"Reply with ONLY the letter A, B, C, or D. [/INST] Answer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=512).to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=5,
            do_sample=False, temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip().upper()
    match = re.search(r'\b([ABCD])\b', response)
    return match.group(1) if match else "A"

def evaluate_model(model, tokenizer, test_data):
    correct = 0
    for i, ex in enumerate(test_data):
        ans = generate_answer(model, tokenizer, ex["question"], ex["options"])
        if ans == ex["answer_idx"]:
            correct += 1
        if (i+1) % 10 == 0:
            print(f"  {i+1}/{len(test_data)} | Correct: {correct}")
    return correct

if __name__ == "__main__":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset("GBaker/MedQA-USMLE-4-options")
    test_data = list(dataset["test"])[:NUM_TEST]

    print("Evaluating base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config,
        device_map="auto", low_cpu_mem_usage=True,
    )
    base_correct = evaluate_model(base_model, tokenizer, test_data)
    del base_model
    torch.cuda.empty_cache()
    gc.collect()

    print("Evaluating fine-tuned model...")
    base_for_ft = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config,
        device_map="auto", low_cpu_mem_usage=True,
    )
    ft_model = PeftModel.from_pretrained(base_for_ft, FINETUNED_MODEL)
    ft_correct = evaluate_model(ft_model, tokenizer, test_data)

    print(f"\n{'='*45}")
    print(f"FINAL RESULTS — {NUM_TEST} USMLE Questions")
    print(f"{'='*45}")
    print(f"Base Mistral 7B:    {base_correct}/{NUM_TEST} ({base_correct*2}%)")
    print(f"Fine-tuned model:   {ft_correct}/{NUM_TEST} ({ft_correct*2}%)")
    print(f"Improvement:        +{(ft_correct-base_correct)*2}%")
    print(f"{'='*45}")

    results = {
        "base_accuracy": f"{base_correct*2}%",
        "finetuned_accuracy": f"{ft_correct*2}%",
        "improvement": f"+{(ft_correct-base_correct)*2}%",
        "test_questions": NUM_TEST,
    }
    with open("evaluation/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to evaluation/results.json")
