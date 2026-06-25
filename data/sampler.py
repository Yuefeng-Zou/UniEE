"""Domain-balanced batch sampler.

Two hard constraints from MD-DAPA:
  1. Every batch must contain windows from a **single domain** — the
     DomainPromptPool selects exactly one prompt per forward pass, and the
     collate function asserts this.
  2. Across an epoch, domains should be sampled with sqrt(N_windows)
     frequency so the rare ones (mpii: 6 sessions) don't get drowned out
     by the populous ones (noxi: 76 (sess,role) → many more windows).

This sampler yields lists of indices, where each list is one batch worth
of indices all drawn from the same domain. Designed to be passed as
``batch_sampler=`` to DataLoader (NOT sampler=).

Eval mode uses a *deterministic* per-domain pass — emits every window in
manifest order, batched by domain. This keeps overlap-add reconstruction
in eval/ensemble code simple.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
from torch.utils.data import Sampler


class DomainBalancedBatchSampler(Sampler[list[int]]):
    def __init__(self, dataset, batch_size: int, n_batches: int,
                 alpha: float = 0.5, seed: int = 0) -> None:
        """
        Parameters
        ----------
        dataset : SessionDataset
            Needs ``_windows`` with ``.domain`` attribute on each entry.
        batch_size : int
            All items in one yielded batch come from the same domain.
        n_batches : int
            Total batches per epoch. The trainer's `len(dataloader)` becomes this.
        alpha : float
            Frequency exponent. 0 = uniform over domains, 1 = proportional
            to N_windows, 0.5 = sqrt (recommended).
        seed : int
            Reproducibility seed for the per-epoch shuffle.
        """
        self.batch_size = int(batch_size)
        self.n_batches = int(n_batches)
        self.alpha = float(alpha)
        self._rng = np.random.RandomState(seed)

        # Group window indices by domain.
        self._by_domain: dict[str, list[int]] = defaultdict(list)
        for i, w in enumerate(dataset._windows):
            self._by_domain[w.domain].append(i)
        self.domains = sorted(self._by_domain)
        counts = np.array([len(self._by_domain[d]) for d in self.domains], dtype=np.float64)
        weights = counts ** alpha
        self.probs = weights / weights.sum()

    def __iter__(self) -> Iterable[list[int]]:
        for _ in range(self.n_batches):
            d = self.domains[self._rng.choice(len(self.domains), p=self.probs)]
            pool = self._by_domain[d]
            # If domain has fewer windows than batch_size, sample with replacement.
            replace = len(pool) < self.batch_size
            idxs = self._rng.choice(pool, size=self.batch_size, replace=replace).tolist()
            yield idxs

    def __len__(self) -> int:
        return self.n_batches


class DomainGroupedEvalSampler(Sampler[list[int]]):
    """Deterministic eval: every window once, batched by contiguous domain runs."""
    def __init__(self, dataset, batch_size: int) -> None:
        self.batch_size = int(batch_size)
        by_domain: dict[str, list[int]] = defaultdict(list)
        for i, w in enumerate(dataset._windows):
            by_domain[w.domain].append(i)
        self._batches: list[list[int]] = []
        for d in sorted(by_domain):
            pool = by_domain[d]
            for i in range(0, len(pool), self.batch_size):
                self._batches.append(pool[i: i + self.batch_size])

    def __iter__(self) -> Iterable[list[int]]:
        yield from self._batches

    def __len__(self) -> int:
        return len(self._batches)
