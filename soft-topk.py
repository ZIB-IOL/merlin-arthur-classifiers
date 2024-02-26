import torch
import torch.nn.functional as F


def top_k_gumbel(logits, k, temperature=0.5):
    top_k_logits = torch.zeros_like(logits)

    for i in range(k):
        top1_gumbel = F.gumbel_softmax(logits, tau=temperature, hard=True)
        top_k_logits += top1_gumbel
        logits = logits - top1_gumbel * 1e6  # mask the selected entry

    return top_k_logits


x = torch.randn(5, 10, requires_grad=True)

# Find the highest 5 entries using Gumbel-softmax trick
y = top_k_gumbel(x, 5, temperature=0.5)

# Example of using the gradients
loss = y.sum()
loss.backward()
