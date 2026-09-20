"""
Training utilities for Weird AI.

This module contains the training loop used to pretrain the Weird AI model
on unlabeled lyric data.
"""

import torch

from weird_ai.losses import calc_loss_batch, calc_loss_loader
from weird_ai.generation import generate_and_print_sample


def evaluate_model(
    model,
    train_loader,
    val_loader,
    device,
    eval_iter
):
    """
    Evaluate the model on a limited number of training and validation batches.

    Args:
        model: The Weird AI model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        device: CPU or CUDA device.
        eval_iter: Number of batches to use for evaluation.

    Returns:
        A tuple containing:
            train_loss: Average training loss.
            val_loss: Average validation loss.
    """

    model.eval()

    # technically not necessary to wrap this but makes sure gradient tracking is off
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )

    return train_loss, val_loss


def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
    context_size
):
    """
    Train the Weird AI model using a basic PyTorch training loop.

    Args:
        model: The Weird AI model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        optimizer: PyTorch optimizer.
        device: CPU or CUDA device.
        num_epochs: Number of epochs to train.
        eval_freq: How often to evaluate, in training steps.
        eval_iter: Number of batches to use during evaluation.
        start_context: Prompt used to generate sample text.
        tokenizer: Tokenizer used for text generation.
        context_size: Maximum context size for generation.

    Returns:
        A tuple containing:
            train_losses: List of recorded training losses.
            val_losses: List of recorded validation losses.
            track_tokens_seen: List of token counts seen at each evaluation point.
    """

    train_losses = []
    val_losses = []
    track_tokens_seen = []

    tokens_seen = 0
    global_step = -1

    model.to(device)

    for epoch in range(num_epochs):

        model.train()

        for input_batch, target_batch in train_loader:

            optimizer.zero_grad()

            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )

            loss.backward()
            optimizer.step()

            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter,
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

            model.train() # switch back eval mode

        generate_and_print_sample(
            model, tokenizer, device, start_context, context_size,
        )

    return train_losses, val_losses, track_tokens_seen


def save_checkpoint(
    model,
    optimizer,
    epoch,
    train_losses,
    val_losses,
    track_tokens_seen,
    checkpoint_path
):
    """
    Save model and optimizer state so training can continue later.

    Args:
        model: The Weird AI model.
        optimizer: PyTorch optimizer.
        epoch: Current epoch.
        train_losses: Recorded training losses.
        val_losses: Recorded validation losses.
        track_tokens_seen: Recorded token counts.
        checkpoint_path: Path where checkpoint should be saved.
    """

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "track_tokens_seen": track_tokens_seen,
        },
        checkpoint_path,
    )


def load_checkpoint(
    model,
    optimizer,
    checkpoint_path,
    device
):
    """
    Load model and optimizer state from a checkpoint.

    Args:
        model: The Weird AI model.
        optimizer: PyTorch optimizer.
        checkpoint_path: Path to checkpoint file.
        device: CPU or CUDA device.

    Returns:
        A dictionary containing checkpoint metadata.
    """

    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )

    model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "train_losses": checkpoint.get("train_losses", []),
        "val_losses": checkpoint.get("val_losses", []),
        "track_tokens_seen": checkpoint.get("track_tokens_seen", []),
    }