import os
import tqdm
import torch

def train_model(model, dataloader, optimizer, criterion, device, num_epochs, checkpoint_path=None, resume_checkpoint=None):
    start_epoch = 0
    if resume_checkpoint is not None and os.path.exists(resume_checkpoint):
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resumed training from epoch {start_epoch}")
    model.train()
    for epoch in range(num_epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)
        epoch_loss = 0.0
        for batch_idx, (x, y) in enumerate(progress_bar):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)  # (batch, seq_len, vocab_size)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch + 1} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")
        avg_loss = epoch_loss / len(dataloader)
        tqdm.write(f"Epoch {epoch+1} finished, Average Loss: {avg_loss:.4f}")
        # print(f"Epoch {epoch + 1} finished, Average Loss: {avg_loss:.4f}")
        if checkpoint_path:
            checkpoint_dict = {
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }
            torch.save(checkpoint_dict, checkpoint_path)