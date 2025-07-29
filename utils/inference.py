import torch
from utils.vocabulary import vocab_size, stoi, itos,chars

def get_lm_probs(context, lm_model, device="cpu", seq_len=128):
    if not context:
        return {char: 1.0 / vocab_size for char in chars}
    context = context[-seq_len:]
    context = context.lower()

    input_indices = [stoi.get(ch, 0) for ch in context]
    input_tensor = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0).to(device)  # (1, len)
    lm_model.eval()
    with torch.no_grad():
        logits = lm_model(input_tensor)  # (1, seq_len, vocab_size)
        last_logits = logits[0, -1, :]     # (vocab_size,)
        probs = torch.softmax(last_logits, dim=0)  # (vocab_size,)
    lm_probs = {itos[i]: probs[i].item() for i in range(vocab_size)}

    sorted_probs = sorted(lm_probs.items(), key=lambda x: x[1], reverse=True)
    print(f"Predicted LM Top3: {sorted_probs[:3]}")

    return lm_probs