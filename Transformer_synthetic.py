import torch


def generate_sequence(model, start_sequence, length, config):
    model.eval()
    generated = torch.tensor(start_sequence, dtype=torch.long).unsqueeze(0).to(config.device)
    with torch.no_grad():
        for _ in range(length - len(start_sequence)):
            outputs = model(generated)
            next_token_logits = outputs[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
    return generated.squeeze().tolist()

# Assuming model and config are already defined and trained as above

start_sequence = [1]  # Starting token
generation_length = 96  # Length of the sequence you want to generate

# Generate sequence
generated_sequence = generate_sequence(model, start_sequence, generation_length, config)

print("Generated Sequence:", generated_sequence)