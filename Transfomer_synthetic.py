from transformers import AdamW, get_linear_schedule_with_warmup

import torch
from torch.utils.data import Dataset, DataLoader
# from Transformer import TransformerModel
import DataProcess_Transformer as dpt
import numpy as np

# def make_model(
#     src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1
# ):
#     "Helper: Construct a model from hyperparameters."
#     c = copy.deepcopy
#     attn = MultiHeadedAttention(h, d_model)
#     ff = PositionwiseFeedForward(d_model, d_ff, dropout)
#     position = PositionalEncoding(d_model, dropout)
#     model = EncoderDecoder(
#         Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
#         Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
#         nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
#         nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
#         Generator(d_model, tgt_vocab),
#     )
#
#     # This was important from their code.
#     # Initialize parameters with Glorot / fan_avg.
#     for p in model.parameters():
#         if p.dim() > 1:
#             nn.init.xavier_uniform_(p)
#     return model
#
# def subsequent_mask(size):
#     "Mask out subsequent positions."
#     attn_shape = (1, size, size)
#     subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
#         torch.uint8
#     )
#     return subsequent_mask == 0


test_model = make_model(11, 11, 2)
test_model.eval()
src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
src_mask = torch.ones(1, 1, 10)

memory = test_model.encode(src, src_mask)
ys = torch.zeros(1, 1).type_as(src)

for i in range(9):
    out = test_model.decode(
        memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
    )
    prob = test_model.generator(out[:, -1])
    _, next_word = torch.max(prob, dim=1)
    next_word = next_word.data[0]
    ys = torch.cat(
        [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
    )

print("Example Untrained Model Prediction:", ys)


class Config:
    vocab_size = 1111  # Adjust according to your dataset if needed
    n_positions = 97
    n_ctx = 97
    n_embd = 128  # Adjust embedding size based on your data
    n_layer = 4  # Fewer layers for a smaller dataset
    n_head = 4


config = Config()
model = TransformerModel(config)
model_checkpoint_path = 'transformer_model.pth'
loaded_checkpoint = torch.load(model_checkpoint_path)
model.load_state_dict(loaded_checkpoint['model_state_dict'])
seq_length = 97
def generate_sequence(model, label):
    model.eval()
    current_seq = torch.full((1, 1), fill_value=1, dtype=torch.long)
    # current_seq[0,0]=1
    # current_seq = torch.tensor([label], dtype=torch.long) # Encode the label
    generated_seq = []

    # Initialize the sequence with the label embedding
    # current_seq = torch.zeros((seq_length, 1, 1))  # (seq_length, batch_size, input_dim)
    for i in range(seq_length):
        with torch.no_grad():
            output = model(current_seq)
            monitor = output.numpy()
            next_value = output[0:1,i:i+1,:]
            next_value_index = torch.argmax(next_value, dim=2)
            # reshaped_next_value = next_value.view(-1)# Forward pass
            # # next_value = output[0, 0, ].item()  # Get the predicted next value
            generated_seq.append(next_value_index)
            current_seq = torch.cat((current_seq,next_value_index),dim=1)  # Update the current sequence

    return generated_seq

label = 1  # Desired label to condition the sequence
generated_sequence = generate_sequence(model, 0)
concatenated_tensor = torch.cat(generated_sequence, dim=1)
final_data = concatenated_tensor[0:96].numpy()



# final_data = generated_sequence.reshape(96*10, 10)

# final_data = stacked_sample.reshape(96*synthetic_n, feature_n)
file_path = 'transformer_weekend.csv'
np.savetxt(file_path, final_data, delimiter=',', fmt='%.2f')


