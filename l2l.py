import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "src")
sys.path.insert(0, src_path) # this was basically only necessary b/c I moved this over from another path
from weird_ai.tokenizer import SimpleCharacterTokenizer
from collections import Counter # easier way to do it
import re

with open("the_verdict.txt", "r", encoding="utf-8") as f:
    text = f.read()

lines = text.splitlines()

non_empty_lines = [
    line.strip()
    for line in lines
    if line.strip()
]

average_length = sum(len(line) for line in non_empty_lines) / len(non_empty_lines)

print(f"Avg line length: {average_length:.2f}")

tokenizer = SimpleCharacterTokenizer(text)

print("\n", tokenizer.chars)
print(len(tokenizer.chars), "separate characters")

tokens = re.findall(r'\b\w+\b', text.lower())
word_counts = Counter(tokens)
unique_once_count = sum(1 for word, count in word_counts.items() if count == 1)

print("\n# of words that appear exactly once:", unique_once_count)
print("\n", word_counts.most_common(20))

vocabulary = set(tokens)
vocabulary_size = len(vocabulary)

print("\nFull vocabulary size:", vocabulary_size)

encoded = tokenizer.encode("hello class")

print(encoded)

decoded = tokenizer.decode(encoded)

print(decoded)