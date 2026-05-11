from datasets import load_dataset
import json
import os

def format_example(example):
    """Convert one MedQA example into Mistral instruction format."""
    question = example["question"]
    options = example["options"]
    answer_idx = example["answer_idx"]
    answer_text = options[answer_idx]

    # Format options as A, B, C, D
    options_str = "\n".join(
        [f"{k}: {v}" for k, v in options.items()]
    )

    # Mistral instruction format
    prompt = (
        f"[INST] You are a medical expert taking the USMLE exam. "
        f"Answer the following question by selecting the best option.\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{options_str}\n\n"
        f"Select the correct answer. [/INST]\n\n"
        f"The correct answer is {answer_idx}: {answer_text}. "
    )
    return {"text": prompt}

def main():
    print("Loading MedQA dataset...")
    dataset = load_dataset("GBaker/MedQA-USMLE-4-options")

    print("Formatting training data...")
    train_formatted = [
        format_example(ex) for ex in dataset["train"]
    ]

    print("Formatting test data...")
    test_formatted = [
        format_example(ex) for ex in dataset["test"]
    ]

    # Save to data/ folder
    os.makedirs("data", exist_ok=True)

    with open("data/train_formatted.json", "w") as f:
        json.dump(train_formatted, f, indent=2)

    with open("data/test_formatted.json", "w") as f:
        json.dump(test_formatted, f, indent=2)

    print(f"\n✅ Done!")
    print(f"Train examples: {len(train_formatted)}")
    print(f"Test examples:  {len(test_formatted)}")
    print(f"\n--- Sample formatted example ---")
    print(train_formatted[0]["text"])

if __name__ == "__main__":
    main()
