"""Cross-layer expert speculative prefetching.

Adjacent MoE layers have >88% cosine similarity in gate inputs (Fate, arxiv
2502.12224). This means the current layer's input can predict which experts
the *next* layer will need with ~97% recall. By running a cheap CPU matmul
during the current layer's GPU compute, we can begin async H2D transfers
for predicted experts — hiding the transfer latency behind compute.

Usage:
    from wm_infra.ops.speculative_prefetch import link_moe_layers

    layers = [MoELayer(config).cuda().half().eval() for _ in range(N)]
    link_moe_layers(layers)

    # Each layer now speculatively prefetches for the next during forward()
"""

import dataclasses
import torch
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from wm_infra.ops.expert_cache import ExpertCache


@dataclasses.dataclass
class SpeculatorStats:
    """Tracks prediction accuracy of speculative prefetching."""

    total_predictions: int = 0  # number of unique experts predicted
    total_hits: int = 0         # predicted AND actually needed
    total_actual: int = 0       # total unique experts actually needed

    @property
    def recall(self) -> float:
        """What fraction of actually-needed experts were predicted."""
        return self.total_hits / max(self.total_actual, 1)

    @property
    def precision(self) -> float:
        """What fraction of predicted experts were actually needed."""
        return self.total_hits / max(self.total_predictions, 1)

    def reset(self):
        self.total_predictions = 0
        self.total_hits = 0
        self.total_actual = 0


class ExpertSpeculator:
    """Predicts next layer's experts and begins async prefetch.

    Uses a CPU copy of the next layer's gate weight to run a cheap matmul
    on the current layer's hidden states. The predicted expert IDs are
    passed to ExpertCache.speculative_prefetch(), which loads them into
    free GPU slots without evicting anything.

    Args:
        gate_weight_cpu: [num_experts, hidden_dim] pinned CPU tensor —
            copy of the next layer's gate weight.
        top_k: The next layer's top_k routing parameter.
        cache: The next layer's ExpertCache instance.
        speculative_top_k: How many experts to predict per token.
            Default: disabled; set >0 to enable explicitly.
    """

    def __init__(
        self,
        gate_weight_cpu: torch.Tensor,
        top_k: int,
        cache: "ExpertCache",
        speculative_top_k: Optional[int] = None,
    ):
        self.gate_weight_cpu = gate_weight_cpu  # [E, H] pinned CPU
        self.top_k = top_k
        self.speculative_top_k = 0 if speculative_top_k is None else speculative_top_k
        self.cache = cache
        self.stats = SpeculatorStats()
        self._last_predicted: Optional[set] = None

    def predict_and_prefetch(self, hidden_states: torch.Tensor):
        """Predict next layer's experts and begin async prefetch.

        Runs a CPU matmul (hidden_states @ gate_weight.T) to predict
        which experts will be selected, then issues non-blocking H2D
        copies for predicted experts using only free cache slots.

        Skips prediction for large batches (M > 256) where nearly all
        experts are needed anyway.

        Args:
            hidden_states: [M, H] current layer's input on GPU.
        """
        if self.speculative_top_k <= 0:
            self._last_predicted = None
            return

        if hidden_states.shape[0] > 256:
            self._last_predicted = None
            return

        # CPU matmul for prediction — doesn't block GPU
        h_cpu = hidden_states.detach().float().cpu()  # [M, H]
        logits = h_cpu @ self.gate_weight_cpu.T       # [M, E]
        _, predicted_ids = torch.topk(logits, self.speculative_top_k, dim=-1)
        unique_predicted = set(predicted_ids.unique().tolist())

        self._last_predicted = unique_predicted
        self.stats.total_predictions += len(unique_predicted)

        # Conservative prefetch: only use free slots, never evict
        self.cache.speculative_prefetch(list(unique_predicted))

    def record_actual(self, actual_expert_ids: List[int]):
        """Record which experts were actually needed for stats tracking.

        Call this after the next layer's routing to measure prediction quality.

        Args:
            actual_expert_ids: List of expert IDs actually used by the next layer.
        """
        actual_set = set(actual_expert_ids)
        self.stats.total_actual += len(actual_set)

        if self._last_predicted is not None:
            hits = self._last_predicted & actual_set
            self.stats.total_hits += len(hits)


def link_moe_layers(layers) -> None:
    """Chain MoE layers so each speculatively prefetches for the next.

    For each consecutive pair (layer_i, layer_{i+1}), creates an
    ExpertSpeculator on layer_i that predicts layer_{i+1}'s expert
    needs using a CPU copy of layer_{i+1}'s gate weight.

    Only links layers that use expert offloading (max_experts_in_gpu is set).
    The last layer gets no speculator (nothing to prefetch for).

    Args:
        layers: List of MoELayer instances, in forward execution order.
    """
    for i in range(len(layers) - 1):
        current = layers[i]
        next_layer = layers[i + 1]

        # Only link if next layer uses offloading
        if not next_layer._use_offloading:
            continue

        # Get or create the next layer's cache (need device info)
        device = next(next_layer.parameters()).device
        next_cache = next_layer._get_expert_cache(device)

        # Create a pinned CPU copy of next layer's gate weight
        gate_weight_cpu = next_layer.gate.weight.detach().float().cpu()
        if not gate_weight_cpu.is_pinned():
            gate_weight_cpu = gate_weight_cpu.pin_memory()

        speculator = ExpertSpeculator(
            gate_weight_cpu=gate_weight_cpu,
            top_k=next_layer.config.top_k,
            cache=next_cache,
            speculative_top_k=next_layer.config.speculative_top_k,
        )
        current._speculator = speculator
