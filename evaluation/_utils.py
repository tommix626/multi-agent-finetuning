import torch
def evaluate_model(model, batch, device):
    """
    Evaluate a PyTorch Model
    :param torch.nn.Module model: the model to be evaluated
    :param torch.utils.data.DataLoader test_dataloader: DataLoader containing testing examples
    :param torch.device device: the device that we'll be training on
    :return accuracy
    """
    # load metrics
    total_loss = 0.0
    total_tokens = 0
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')
    total_correct = 0

    # turn model into evaluation mode
    model.eval()

    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch["labels"].to(device)

    # forward pass
    output = model(input_ids=input_ids, attention_mask=attention_mask, labels = labels)
    logits = output.logits

    # Shift for next token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    num_tokens = (shift_labels != -100).sum()

    total_loss += loss.item()
    total_tokens += num_tokens.item()

    # Compute accuracy 
    predictions = torch.argmax(shift_logits, dim=-1)
    mask = shift_labels != -100
    correct = (predictions == shift_labels) & mask
    total_correct += correct.sum().item()

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    accuracy = total_correct / total_tokens


    # compute and return metrics
    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "accuracy": accuracy
    }
