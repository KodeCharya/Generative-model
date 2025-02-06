import torch
import torch.nn as nn
import torch.optim as optim

class BertModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_heads, max_length):
        super(BertModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_length = max_length
        self.build_model()

    def build_model(self):
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=self.num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.fc = nn.Linear(self.embedding_dim, self.vocab_size)

    def forward(self, input_ids, attention_mask=None):
        embeddings = self.embedding(input_ids)
        transformer_output = self.transformer_encoder(embeddings)
        logits = self.fc(transformer_output)
        return logits

    def train(self, train_data, epochs, learning_rate):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            self.train()
            total_loss = 0
            for batch in train_data:
                inputs, labels = batch
                optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = criterion(outputs.view(-1, self.vocab_size), labels.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_data)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")