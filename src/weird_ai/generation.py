import torch


def text_to_token_ids(text, tokenizer):
    ids = tokenizer.encode(text)
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    if token_ids.dim() == 2:
        token_ids = token_ids.squeeze(0)
    return tokenizer.decode(token_ids.tolist())


def generate_text_simple(model, input_ids, max_new_tokens, context_size):
    model.eval()
    for _ in range(max_new_tokens):
        cropped = input_ids[:, -context_size:]
        with torch.no_grad():
            logits = model(cropped)
        next_logits = logits[:, -1, :]
        next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_id], dim=1)
    return input_ids


def generate_and_print_sample(model, tokenizer, device, start_context,
                              context_size, max_new_tokens=50):
    model.eval()
    ids = text_to_token_ids(start_context, tokenizer).to(device)
    out = generate_text_simple(model, ids, max_new_tokens, context_size)
    print(token_ids_to_text(out, tokenizer))