"""
Recreation of GPT-1
"""

import torch

text = ""
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("The number of characters is: ", len(text))

# unique chars vocabulary
vocabulary = sorted(list(set(text)))
vocabulary_size = len(vocabulary)
print("The vocabulary is: ", "".join(vocabulary))
print("There are ", vocabulary_size ," chars")

# tokenizer
char_to_int = { v: i for i, v in enumerate(vocabulary)}
int_to_char = { v: i for i, v in enumerate(vocabulary)}

def encode(word: str) -> list[int]:
    return [char_to_int[c] for c in word]

def decode(word: list[int]) -> str:
    return "".join([int_to_char[i] for i in word])


# Start encoding
input_data = torch.tensor(encode(text), dtype=torch.long)
print("input shape: ", input_data.shape, " input type: ", input_data.dtype)


# Use the first 90% for training set
# Use the last 10% for validation set
train_data_size = int(0.9 * len(input_data))
train_data = input_data[:train_data_size]
validation_data = input_data[train_data_size:]


# training batches and validation
torch.manual_seed(1337)
batch_size = 4 # number of independent sequences processed in parallel
block_size = 8 # Context length for predictions

def get_batch(split: str):
    data = train_data if split == "train" else validation_data
    # get batch_size number of randomly generated offsets
    offsets = torch.randint(len(input_data) - block_size, (batch_size,))
    x = torch.stack([data[i: i + block_size] for i in offsets]) # create a batch_size x block_size tensor for training
    y = torch.stack([data[i + 1: i + block_size + 1] for i in offsets]) # create a batch_size x block_size tensor for validation
    return x, y



