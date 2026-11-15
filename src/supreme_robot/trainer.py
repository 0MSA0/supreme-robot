import torch
import wandb


def train_model(model, optimizer, scheduler, criterion, train_loader, num_epochs, wandb_project_name, wandb_config, save_path):

    # Get device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        # Use PyTorch 2.9 or newer for stable mps results
        major, minor = map(int, torch.__version__.split(".")[:2])
        if (major, minor) >= (2, 9):
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    print(f"Using {device} device.")

    model.to(device)

    with wandb.init(project=wandb_project_name, config=wandb_config) as run:
        global_step = 0
        for epoch in range(num_epochs):
            model.train()
            for step, (in_batch, out_batch) in enumerate(train_loader):
                # Send data to device
                in_batch = in_batch.to(device)
                out_batch = out_batch.to(device)

                optimizer.zero_grad()

                # Forward pass
                out_predicted = model(in_batch)

                logits = out_predicted.view(-1, out_predicted.size(-1))

                # Targets: [batch_size * seq_length]
                targets = out_batch.view(-1)

                # Compute loss
                loss = criterion(logits, targets)

                # Back propagation
                loss.backward()

                optimizer.step()

                # Don't use grad for accuracy
                with torch.no_grad():
                    # Compute accuracy
                    predictions = torch.argmax(logits, dim=-1)
                    accuracy = (predictions == targets).float().mean().item()

                # Get VRAM
                vram_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)

                # LOG
                run.log({
                    "train/loss": loss.item(),
                    "train/accuracy": accuracy,
                    "system/vram_allocated_gb": vram_gb,
                    "epoch": epoch,
                })

                global_step += 1
            
            scheduler.step()

        run.finish()

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
            save_path
        )