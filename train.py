import torch
from transformers import BertTokenizer, BertForSequenceClassification
print(torch.backends.mps.is_available())
print(torch.backends.mps.is_built())
torch.device("mps")

# Define the training data
product_descriptions = [
    "Signal White Now Erkekler İçin Leke Karşıtı Diş Macunu 75 ml",
    "Signal White Now Men Erkeklere Özel Leke Karşıtı Diş Macunu 75 ml",
    "Signal White Now Lekesiz Beyazlık Anında Beyazlatıcı Leke Karşıtı Diş Macunu 75 ml",
    "Tat Katkısız Tatlı Ketçap 630 G",
    "Tat Ketçap 630gr Tatlı Edt",
    "Tat 630 gr Tatlı Ketçap",
    "CALVE MAYONEZ 540 GR",
    "Calve Mayonez 540 Gr",
    "calve mayonez 540cc",
    "Duracell 10'lu Aaa İnce Kalem Pil",
    "Duracell Simply Aaa 10 İnce Kalem Pil 10'lu",
    "Duracell 1,5v Aaa İnce Pil Alkalin (10'lu Paket)",
    "Duracell Pil Kalem Aa 10'lu",
    "Duracell 10'lu Aa Kalem Pil",
    "Duracell Alkalin Aa 10'lu Kalem Pil",
    "Pril Excellence Bulaşık Makinesi Kapsülü 40'lı",
    "Pril Excellence 40 Kapsül",
    "Pril Excellence 4'ü 1 Arada Bulaşık Makinesi Tableti 40'lı",
    "Pril Gold Tablet 45 Li",
    "Pril Gold 45 Yıkama Bulaşık Makinesi Deterjanı Tableti",
    "Pril Gold Yanmış Kirler için 45 Tablet Bulaşık Makinesi Deterjanı Tableti (1 x 45 tablet)"
]

# Install the tokenizer
tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-uncased')

# Merge the tokens into a single tensor
tokenized_inputs = [tokenizer.encode(product, return_tensors='pt').long() for product in product_descriptions]

# Pad the tokens to the same size
max_size = max(token.shape[1] for token in tokenized_inputs)
tokenized_inputs = [torch.cat([token, torch.zeros((1, max_size - token.shape[1]))], dim=1).long() for token in tokenized_inputs]

# Concatenate the tokenized inputs into a single tensor
X_train = torch.cat(tokenized_inputs, dim=0)

# Define labels for each product
labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6])  # 0: Signal White Now, 1: Tat Ketçap 630gr, 2: Calve Mayonez

# Define the model
model = BertForSequenceClassification.from_pretrained('dbmdz/bert-base-turkish-uncased', num_labels=7)

# Define the optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss()

# Train the model
num_epochs = 100000

prev_losses = []
patience = 10  # Number of epochs with no improvement to wait
threshold = 0.0001  # Minimum change in the monitored quantity to qualify as an improvement
  

for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train)

    # Calculate loss
    loss = criterion(outputs.logits, labels)
    prev_losses.append(loss)
    
    # Check if the last 'patience' epochs had similar loss values
    if len(prev_losses) > patience and all(abs(x - prev_losses[-1]) < threshold for x in prev_losses[-patience:]):
        print(f"Early stopping at epoch {epoch} due to no improvement in the last {patience} epochs.")
        break

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss every epoch
    print("Epoch {} - Loss: {:.4f}".format(epoch + 1, loss.item()))

# Save the model
model.save_pretrained('product_similarity_model')

# Save the tokenizer
tokenizer.save_pretrained('product_similarity_tokenizer')