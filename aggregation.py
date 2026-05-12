"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).

Converts per-token, per-layer hidden states from the extraction loop in
``solution.py`` into flat feature vectors for the probe classifier.

Two stages can be customised independently:

  1. ``aggregate`` — select layers and token positions, pool into a vector.
  2. ``extract_geometric_features`` — optional hand-crafted features
     (enabled by setting ``USE_GEOMETRIC = True`` in ``solution.py``).

Both stages are combined by ``aggregation_and_feature_extraction``, the
single entry point called from the notebook.
"""

from __future__ import annotations

import torch


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    # Берём последний слой
    last_layer = hidden_states[-1]  # (seq_len, hidden_dim)
    
    # Усредняем только по реальным токенам (где attention_mask == 1)
    mask = attention_mask.bool()  # (seq_len,)
    masked = last_layer[mask]      # (n_real_tokens, hidden_dim)
    
    if masked.shape[0] == 0:
        # Если нет реальных токенов (не должно случиться), возвращаем нули
        return torch.zeros(last_layer.shape[1])
    
    feature = masked.mean(dim=0)    # (hidden_dim,)
    return feature
    # ------------------------------------------------------------------


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    # Берём последний слой и последний токен
    last_layer = hidden_states[-1]  # (seq_len, hidden_dim)
    real_positions = attention_mask.nonzero(as_tuple=False)
    last_pos = int(real_positions[-1].item())
    last_token_emb = last_layer[last_pos]  # (hidden_dim,)
    
    # Вычисляем норму (L2) как один дополнительный признак
    norm = torch.norm(last_token_emb, p=2).unsqueeze(0)  # (1,)
    
    return norm


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric features.

    Main entry point called from ``solution.ipynb`` for each sample.
    Concatenates the output of ``aggregate`` with that of
    ``extract_geometric_features`` when ``use_geometric=True``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``
                        for a single sample.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.
        use_geometric:  Whether to append geometric features.  Controlled by
                        the ``USE_GEOMETRIC`` flag in ``solution.ipynb``.

    Returns:
        A 1-D float tensor of shape ``(feature_dim,)`` where
        ``feature_dim = hidden_dim`` (or larger for multi-layer or geometric
        concatenations).
    """
    agg_features = aggregate(hidden_states, attention_mask)  # (feature_dim,)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
