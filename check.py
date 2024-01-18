import torch
from transformers import BertTokenizer, BertForSequenceClassification
torch.device("mps")

# Load the trained model and tokenizer from the correct path
model = BertForSequenceClassification.from_pretrained('product_similarity_model')
tokenizer = BertTokenizer.from_pretrained('product_similarity_tokenizer')

# Test data
test_product_descriptions = [
    "Pril Gold 12 Etki Bulaşık Makinesi Deterjanı",
    "Duracell 10'lu paket Aa pil",
    "Duracell 10'lu aaa pil",
    "Pril Excellence 4U 1 Arada Bulasik Makinesi Tableti"
]

# Tokenize and format the test data
tokenized_test_inputs = [tokenizer.encode(product, return_tensors='pt').squeeze() for product in test_product_descriptions]
max_size = max(token.shape[0] for token in tokenized_test_inputs)
tokenized_test_inputs = [torch.cat([token, torch.zeros((max_size - token.shape[0]))], dim=0).long() for token in tokenized_test_inputs]
X_test = torch.stack(tokenized_test_inputs)

# Ensure the model is in evaluation mode
model.eval()

# Pass the test data through the model
with torch.no_grad():
    outputs = model(X_test)

# Get predictions
_, predicted_labels = torch.max(outputs.logits, 1)

# Print predictions
for i, product_description in enumerate(test_product_descriptions):
    print(f"Product: {product_description}, Predicted Label: {predicted_labels[i].item()}")
