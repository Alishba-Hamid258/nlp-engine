
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

# Load dataset
df = pd.read_csv("data/aspect_sentiment_data.csv")
df['label'] = df['sentiment'].map({"Positive": 2, "Neutral": 1, "Negative": 0})

# Custom Dataset
class ABSDataset(Dataset):
    def __init__(self, texts, aspects, labels, tokenizer, max_length=128):
        self.texts = texts
        self.aspects = aspects
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        aspect = str(self.aspects[idx])
        label = self.labels[idx]
        encoded = self.tokenizer(text, aspect, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        return {
            'input_ids': encoded['input_ids'].flatten(),
            'attention_mask': encoded['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Prepare data
texts = df['text'].values
aspects = df['aspect'].values
labels = df['label'].values
train_texts, val_texts, train_aspects, val_aspects, train_labels, val_labels = train_test_split(texts, aspects, labels, test_size=0.2)

train_dataset = ABSDataset(train_texts, train_aspects, train_labels, tokenizer)
val_dataset = ABSDataset(val_texts, val_aspects, val_labels, tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("models/absa_model")
tokenizer.save_pretrained("models/absa_model")
