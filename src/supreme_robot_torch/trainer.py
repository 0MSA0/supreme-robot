import torch
import wandb


def train_model(
    model,
    optimizer,
    scheduler,
    criterion,
    train_loader,
    num_epochs,
    wandb_project_name,
    wandb_config,
    save_path,
    use_amp=True,         # bf16 autocast -> Tensor Cores + FlashAttention
    compile_model=False,  # optional torch.compile
):
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

    amp_enabled = use_amp and device.type == "cuda"
    if device.type == "cuda":
        # TF32 Tensor Cores for any remaining fp32 matmuls
        torch.set_float32_matmul_precision("high")

    # Keep `model` uncompiled so state_dict keys have no "_orig_mod." prefix
    fwd_model = torch.compile(model) if compile_model else model

    with wandb.init(project=wandb_project_name, config=wandb_config) as run:
        global_step = 0
        for epoch in range(num_epochs):
            model.train()
            for step, (in_batch, out_batch) in enumerate(train_loader):
                in_batch = in_batch.to(device, non_blocking=True)
                out_batch = out_batch.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                # bf16 needs no GradScaler (same exponent range as fp32)
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled
                ):
                    out_predicted = fwd_model(in_batch)
                    logits = out_predicted.view(-1, out_predicted.size(-1))
                    targets = out_batch.view(-1)
                    loss = criterion(logits, targets)  # autocast runs CE in fp32

                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    predictions = torch.argmax(logits, dim=-1)
                    accuracy = (predictions == targets).float().mean().item()

                vram_gb = (
                    torch.cuda.memory_allocated(device) / (1024**3)
                    if device.type == "cuda"
                    else 0.0
                )

                run.log({
                    "train/loss": loss.item(),
                    "train/accuracy": accuracy,
                    "system/vram_allocated_gb": vram_gb,
                    "epoch": epoch,
                })

                global_step += 1

            scheduler.step()

        run.finish()

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            save_path,
        )
