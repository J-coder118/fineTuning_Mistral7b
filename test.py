import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW

# Set device to CPU
device = torch.device("cpu")

# Load tokenizer and model on CPU
model_name = "EleutherAI/qm-Mistral-7B-v0.1-grader-last"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.to(device)

# Load and preprocess the CSV data
df = pd.read_csv("test.csv")
texts = df["text_column"].tolist()
labels = df["label_column"].tolist()

# Tokenize and encode the texts
encoded_inputs = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
encoded_inputs.to(device)

# Convert labels to tensors
labels_tensor = torch.tensor(labels).to(device)

# Configure the optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

# Fine-tuning loop
num_epochs = 5
batch_size = 8

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Perform forward pass
    outputs = model(**encoded_inputs, labels=labels_tensor)

    # Compute loss and perform backward pass
    loss = outputs.loss
    loss.backward()
    optimizer.step()

    print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Save the fine-tuned model
model.save_pretrained("fine_tuned_model")
tokenizer.save_pretrained("fine_tuned_model")