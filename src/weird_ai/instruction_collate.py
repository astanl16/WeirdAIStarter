"""
Custom collate function for Weird AI instruction fine-tuning.
"""

import torch


def custom_collate_fn(
    batch,
    pad_token_id=0,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    """
    Pad variable-length token ID lists, create shifted targets, and mask extra padding.
    """
    batch_max_length = max(len(item) + 1 for item in batch)

    inputs_lst = []
    targets_lst = []

    for item in batch:
        new_item = item.copy()

        # append one pad token to new_item
        new_item.append(pad_token_id)

        # pad new_item to batch_max_length
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))

        # inputs are padded[:-1], targets are padded[1:]
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        # replace all but the first pad_token_id in targets with ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # if allowed_max_length is not None, truncate inputs and targets
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # stack input and target tensors, then move them to device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor
