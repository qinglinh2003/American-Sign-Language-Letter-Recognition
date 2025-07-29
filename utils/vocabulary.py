chars = list("abcdefghijklmnopqrstuvwxyz ")
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def get_vocab():
    return chars, vocab_size, stoi, itos
