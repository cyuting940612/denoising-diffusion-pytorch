import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import math


# Prepare dataset
class SentenceDataset(Dataset):
    def __init__(self, sentences, tokenizer, max_length):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        inputs = self.tokenizer.encode_plus(
            sentence,
            None,
            add_special_tokens=True,
            max_length=self.max_length,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long)
        }


# Transformer model definition
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward,
                 max_seq_length):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = self.create_positional_encoding(max_seq_length, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def create_positional_encoding(self, max_seq_length, d_model):
        positional_encoding = torch.zeros(max_seq_length, d_model)
        for pos in range(max_seq_length):
            for i in range(0, d_model, 2):
                positional_encoding[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                positional_encoding[pos, i + 1] = math.cos(pos / (10000 ** ((2 * i + 1) / d_model)))
        return positional_encoding

    def forward(self, src, tgt, src_mask, tgt_mask):
        src_embedding = self.embedding(src) + self.positional_encoding[:src.size(1), :]
        tgt_embedding = self.embedding(tgt) + self.positional_encoding[:tgt.size(1), :]
        transformer_output = self.transformer(src_embedding, tgt_embedding, src_key_padding_mask=src_mask,
                                              tgt_key_padding_mask=tgt_mask)
        output = self.fc_out(transformer_output)
        return output


# Load data
sentences = ["Your dataset of sentences goes here"]
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_length = 50
vocab_size = tokenizer.vocab_size

dataset = SentenceDataset(sentences, tokenizer, max_length)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Define model
model = TransformerModel(vocab_size, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                         dim_feedforward=2048, max_seq_length=max_length)

# Training setup
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Training loop
model.train()
for epoch in range(10):  # Number of epochs
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        src = input_ids[:, :-1]
        tgt = input_ids[:, 1:]

        src_mask = attention_mask[:, :-1]
        tgt_mask = attention_mask[:, 1:]

        outputs = model(src, tgt, src_mask, tgt_mask)
        loss = criterion(outputs.view(-1, vocab_size), tgt.view(-1))
        loss.backward()
        optimizer.step()

# Save model
torch.save(model.state_dict(), './trained_transformer_model.pth')
torch.save(tokenizer, './tokenizer.pth')


# Inference
def complete_sentence(starting_word):
    model.eval()
    input_ids = tokenizer.encode(starting_word, return_tensors='pt')
    for _ in range(max_length - len(input_ids[0])):
        src = input_ids
        tgt = input_ids
        src_mask = torch.ones(src.shape, dtype=torch.bool)
        tgt_mask = torch.ones(tgt.shape, dtype=torch.bool)
        with torch.no_grad():
            output = model(src, tgt, src_mask, tgt_mask)
            next_token_logits = output[0, -1, :]
            next_token = torch.argmax(next_token_logits)
            input_ids = torch.cat((input_ids, next_token.unsqueeze(0).unsqueeze(0)), dim=1)
            if next_token.item() == tokenizer.sep_token_id:
                break
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


print(complete_sentence("The"))