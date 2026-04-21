
# Day 3: Prompt Engineering Basics
# Techniques: Zero-shot, Few-shot, Temperature control

from transformers import pipeline
import gc

gc.collect()

# Load generator
generator = pipeline("text-generation", model="distilgpt2")
classifier = pipeline("zero-shot-classification")

# ==========================================
# TECHNIQUE 1: Zero-Shot Prompting
# ==========================================
print("ZERO-SHOT PROMPTING:")
prompt1 = "Summarize machine learning for a 10-year-old:"
result1 = generator(prompt1, max_length=100, temperature=0.7, do_sample=True)
print(f"Prompt: {prompt1}")
print(f"Output: {result1[0]['generated_text']}\n")

# ==========================================
# TECHNIQUE 2: Few-Shot (Classification)
# ==========================================
print("FEW-SHOT CLASSIFICATION:")
reviews = [
    "This product is absolutely amazing!",
    "Terrible quality, broke after one day",
    "It's okay, nothing special"
]

for review in reviews:
    result = classifier(review, candidate_labels=["positive", "negative", "neutral"])
    print(f"'{review}' → {result['labels'][0].upper()}")

# ==========================================
# TECHNIQUE 3: Temperature Control
# ==========================================
print("\nTEMPERATURE COMPARISON:")
prompt = "The future of artificial intelligence"

for temp in [0.3, 0.7, 1.5]:
    print(f"\nTemperature {temp}:")
    result = generator(prompt, max_length=40, temperature=temp, do_sample=True)
    print(result[0]['generated_text'])
