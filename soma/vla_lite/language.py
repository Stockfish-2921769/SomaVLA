"""Language slot without an LLM.

The current task uses one template instruction, so the main implementation is
a learned task embedding.  A frozen sentence encoder can be plugged in later
when open-vocabulary instructions are introduced; it is deliberately not an
autoregressive LLM.
"""
import torch
import torch.nn as nn


class LanguageSlot(nn.Module):
    def __init__(self, d=256, task_dim=8, n_tasks=1):
        super().__init__()
        self.task_dim = task_dim
        self.emb = nn.Embedding(n_tasks, task_dim)
        self.token = nn.Linear(task_dim, d)

    def forward(self, task_ids=None, batch_size=None, device=None):
        if task_ids is None:
            if batch_size is None:
                raise ValueError("task_ids or batch_size required")
            task_ids = torch.zeros(batch_size, dtype=torch.long,
                                   device=device)
        e = self.emb(task_ids)
        return e, self.token(e)
