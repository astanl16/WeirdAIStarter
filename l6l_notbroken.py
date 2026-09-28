import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

torch.manual_seed(123)

device = torch.device("cuda")
print(f"Using device: {device}\n")


# sec 2
data = [
    {"text": "I walked alone beneath the rain and wondered where you were", "label": 0},
    {"text": "The moon was cold, the night was long, my heart forgot the tune", "label": 0},
    {"text": "Every road reminds me of the love I left behind", "label": 0},
    {"text": "The silence in this empty room still whispers your name", "label": 0},
    {"text": "I kept your letter folded by the window in the storm", "label": 0},
    {"text": "The stars fell softly as I waited for your call", "label": 0},

    {"text": "My sandwich ran away and joined a polka band", "label": 1},
    {"text": "I fell in love with a toaster wearing neon shoes", "label": 1},
    {"text": "The cafeteria spaghetti tried to steal my car", "label": 1},
    {"text": "My homework ate my dog and then blamed it on the cat", "label": 1},
    {"text": "The disco duck accountant audited my socks", "label": 1},
    {"text": "I wrote a love song to my broken microwave", "label": 1},
]

df = pd.DataFrame(data)
print(df, "\n")


# sec 3
print(df["label"].value_counts())
print("\nSerious examples:")
print(df[df["label"] == 0].head())
print("\nSilly examples:")
print(df[df["label"] == 1].head(), "\n")


# sec 4
train_df, temp_df = train_test_split(
    df, test_size=0.30, random_state=123, stratify=df["label"]
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=123, stratify=temp_df["label"]
)

print(f"Training rows: {len(train_df)}")
print(f"Validation rows: {len(val_df)}")
print(f"Test rows: {len(test_df)}\n")


# sec 5
from weird_ai.tokenizer import SimpleCharacterTokenizer

all_text = "\n".join(df["text"].tolist())
tokenizer = SimpleCharacterTokenizer(all_text)

print(f"Vocabulary size: {len(tokenizer.chars)}")
print(tokenizer.chars, "\n")

sample_text = df.iloc[0]["text"]
encoded = tokenizer.encode(sample_text)
decoded = tokenizer.decode(encoded)
print(sample_text)
print(encoded)
print(decoded, "\n")


# sec 6
class LyricsClassificationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=None, pad_token_id=0):
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id

        self.encoded_texts = [
            tokenizer.encode(text) for text in self.dataframe["text"]
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length

        self.encoded_texts = [enc[:self.max_length] for enc in self.encoded_texts]

        self.encoded_texts = [
            enc + [self.pad_token_id] * (self.max_length - len(enc))
            for enc in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.dataframe.iloc[index]["label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )

    def __len__(self):
        return len(self.dataframe)

    def _longest_encoded_length(self):
        return max(len(enc) for enc in self.encoded_texts)


# sec 7
train_dataset = LyricsClassificationDataset(train_df, tokenizer)
val_dataset = LyricsClassificationDataset(
    val_df, tokenizer, max_length=train_dataset.max_length
)
test_dataset = LyricsClassificationDataset(
    test_df, tokenizer, max_length=train_dataset.max_length
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

input_batch, label_batch = next(iter(train_loader))
print(input_batch)
print(label_batch)
print(input_batch.shape)
print(label_batch.shape, "\n")


# sec 8
class TinyLyricsClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, input_ids):
        embeddings = self.embedding(input_ids)
        pooled = embeddings.mean(dim=1)     # (batch, emb_dim)
        logits = self.classifier(pooled)
        return logits


# sec 9
vocab_size = len(tokenizer.chars)
emb_dim = 32
num_classes = 2

model = TinyLyricsClassifier(vocab_size, emb_dim, num_classes).to(device)

input_batch, label_batch = next(iter(train_loader))
input_batch = input_batch.to(device)
logits = model(input_batch)
print(logits)
print(logits.shape, "\n")


# sec 10
input_batch, label_batch = next(iter(train_loader))
input_batch = input_batch.to(device)
label_batch = label_batch.to(device)
logits = model(input_batch)
loss = F.cross_entropy(logits, label_batch)
print("Initial loss:", loss.item(), "\n")


# sec 11
def calculate_accuracy(data_loader, model, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for input_batch, label_batch in data_loader:
            input_batch = input_batch.to(device)
            label_batch = label_batch.to(device)
            logits = model(input_batch)
            predicted_labels = torch.argmax(logits, dim=-1)
            correct += (predicted_labels == label_batch).sum().item()
            total += label_batch.shape[0]
    return correct / total


print("Val accuracy before training:", calculate_accuracy(val_loader, model, device), "\n")


# sec 12
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for input_batch, label_batch in train_loader:
        input_batch = input_batch.to(device)
        label_batch = label_batch.to(device)

        optimizer.zero_grad()
        logits = model(input_batch)
        loss = F.cross_entropy(logits, label_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_acc = calculate_accuracy(train_loader, model, device)
    val_acc = calculate_accuracy(val_loader, model, device)

    print(f"Epoch {epoch + 1}: loss = {avg_loss:.4f} | "
          f"train acc = {train_acc:.2%} | val acc = {val_acc:.2%}")


# sec 13
test_accuracy = calculate_accuracy(test_loader, model, device)
print(f"\nTest accuracy: {test_accuracy:.2%}\n")


# sec 14
def classify_text(text, model, tokenizer, max_length, device):
    model.eval()
    encoded = tokenizer.encode(text)
    encoded = encoded[:max_length]
    encoded = encoded + [0] * (max_length - len(encoded))
    input_tensor = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_tensor)
    predicted_label = torch.argmax(logits, dim=-1).item()
    return predicted_label


examples = [
    "My sandwich started singing opera in the shower",
    "I still remember the night you walked away",
]

for example in examples:
    label = classify_text(example, model, tokenizer, train_dataset.max_length, device)
    name = "silly" if label == 1 else "serious"
    print(f"{example} => {label} ({name})")