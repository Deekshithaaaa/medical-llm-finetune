from datasets import load_dataset
import pandas as pd

# Load MedQA
dataset = load_dataset("GBaker/MedQA-USMLE-4-options")

print(dataset)
print("\n--- Sample question ---")
sample = dataset["train"][0]
print("Question:", sample["question"])
print("Options:", sample["options"])
print("Answer:", sample["answer_idx"])

print("\nTrain size:", len(dataset["train"]))
print("Test size:", len(dataset["test"]))

# Look at 5 examples
print("\n--- 5 Examples ---")
for i in range(5):
    s = dataset["train"][i]
    print(f"\nQ{i+1}: {s['question'][:100]}...")
    print(f"Answer: {s['answer_idx']}")
