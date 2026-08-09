import copy

import pytest
import torch
import torch.nn as nn

from src.encoding import relative_position_bucket
from src.models.attention import PositionAwareSparseAttention
from src.models.feature_composition import compose_legacy_variant_features_torch
from src.models.position_runtime import (
    LegacyT5RelativePositionRuntime,
    ObservedAbsolutePositionRuntime,
)
from src.models.sieve import SIEVE, load_state_dict_with_legacy_upgrade


def _make_features(input_dim: int, content_dim: int):
    torch.manual_seed(700 + input_dim + content_dim)
    features = torch.randn(2, 4, input_dim)
    if input_dim == content_dim:
        return features, features.clone(), features[..., :0].clone()
    content = torch.cat([features[..., :1], features[..., 65:]], dim=-1)
    absolute_position = features[..., 1:65].clone()
    return features, content, absolute_position


def _make_metadata():
    positions = torch.tensor(
        [
            [100, 120, 500, 900],
            [200, 240, 260, 1000],
        ],
        dtype=torch.long,
    )
    gene_ids = torch.tensor(
        [
            [0, 1, 2, 3],
            [1, 2, 0, 3],
        ],
        dtype=torch.long,
    )
    mask = torch.tensor(
        [
            [True, True, True, True],
            [True, False, True, True],
        ],
        dtype=torch.bool,
    )
    chrom_ids = torch.tensor(
        [
            [0, 1, 1, 0],
            [0, 0, 1, 1],
        ],
        dtype=torch.long,
    )
    return positions, gene_ids, mask, chrom_ids


def _set_nonzero_chrom_embedding(module):
    if module.chrom_embedding is None:
        return
    with torch.no_grad():
        values = torch.arange(
            module.chrom_embedding.weight.numel(),
            dtype=module.chrom_embedding.weight.dtype,
        ).reshape_as(module.chrom_embedding.weight)
        module.chrom_embedding.weight.copy_(values / 50.0)


def _make_attention(*, num_chromosomes: int = 0):
    torch.manual_seed(1100 + num_chromosomes)
    attention = PositionAwareSparseAttention(
        latent_dim=8,
        num_heads=2,
        dropout=0.0,
        num_position_buckets=32,
        max_distance=100000,
        num_chromosomes=num_chromosomes,
    )
    attention.eval()
    with torch.no_grad():
        attention.position_bias.weight.copy_(
            torch.arange(33 * 2, dtype=torch.float32).reshape(33, 2) / 100.0
        )
    _set_nonzero_chrom_embedding(attention)
    return attention


def _make_sieve(input_dim: int, *, num_chromosomes: int = 0):
    torch.manual_seed(2100 + input_dim + num_chromosomes)
    model = SIEVE(
        input_dim=input_dim,
        num_genes=4,
        latent_dim=8,
        hidden_dim=10,
        num_heads=2,
        num_attention_layers=2,
        classifier_hidden_dim=12,
        dropout=0.0,
        num_chromosomes=num_chromosomes,
    )
    model.eval()
    for layer in model.attention.attention_layers:
        _set_nonzero_chrom_embedding(layer)
    return model


def _legacy_position_buckets(
    query_positions,
    key_positions,
    *,
    num_position_buckets,
    max_distance,
    query_chroms=None,
    key_chroms=None,
):
    buckets = []
    for batch_idx in range(query_positions.shape[0]):
        buckets.append(
            relative_position_bucket(
                query_positions[batch_idx],
                key_positions[batch_idx],
                num_buckets=num_position_buckets,
                max_distance=max_distance,
                query_chroms=query_chroms[batch_idx] if query_chroms is not None else None,
                key_chroms=key_chroms[batch_idx] if key_chroms is not None else None,
            )
        )
    return torch.stack(buckets, dim=0)


def _legacy_position_bias(
    layer, query_positions, key_positions, query_chroms=None, key_chroms=None
):
    buckets = _legacy_position_buckets(
        query_positions,
        key_positions,
        num_position_buckets=layer.num_position_buckets,
        max_distance=layer.max_distance,
        query_chroms=query_chroms,
        key_chroms=key_chroms,
    )
    return layer.position_bias(buckets).permute(0, 3, 1, 2)


def _legacy_attention_reference(layer, x, positions, mask=None, chrom_ids=None):
    batch_size, num_variants, _ = x.shape
    if chrom_ids is not None and layer.chrom_embedding is not None:
        x = x + layer.chrom_embedding(chrom_ids)

    query = layer.query(x)
    key = layer.key(x)
    value = layer.value(x)

    query = query.view(batch_size, num_variants, layer.num_heads, layer.head_dim).transpose(1, 2)
    key = key.view(batch_size, num_variants, layer.num_heads, layer.head_dim).transpose(1, 2)
    value = value.view(batch_size, num_variants, layer.num_heads, layer.head_dim).transpose(1, 2)

    scores = torch.matmul(query, key.transpose(-2, -1)) / (layer.head_dim**0.5)
    scores = scores + _legacy_position_bias(
        layer,
        positions,
        positions,
        query_chroms=chrom_ids,
        key_chroms=chrom_ids,
    )

    if mask is not None:
        scores = scores.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0)
    weights_dropout = layer.dropout(weights)
    attended = torch.matmul(weights_dropout, value)
    attended = attended.transpose(1, 2)
    attended = attended.contiguous().view(batch_size, num_variants, layer.latent_dim)
    return layer.output_proj(attended), weights


def _legacy_sieve_reference(model, features, positions, gene_ids, mask, chrom_ids=None):
    variant_embeddings = model.variant_encoder(features)
    attended = variant_embeddings
    attention_weights = []
    for layer, layer_norm in zip(
        model.attention.attention_layers,
        model.attention.layer_norms,
        strict=True,
    ):
        attention_output, weights = _legacy_attention_reference(
            layer,
            attended,
            positions,
            mask,
            chrom_ids=chrom_ids,
        )
        attended = layer_norm(attended + attention_output)
        attention_weights.append(weights)

    gene_embeddings = model.gene_aggregator(attended, gene_ids, mask)
    logits = model.classifier(gene_embeddings)
    return {
        "logits": logits,
        "variant_embeddings": variant_embeddings,
        "attended_embeddings": attended,
        "gene_embeddings": gene_embeddings,
        "attention_weights": attention_weights,
    }


@pytest.mark.parametrize("position_width", [0, 64])
def test_observed_absolute_runtime_returns_exact_observed_tensor(position_width):
    runtime = ObservedAbsolutePositionRuntime()
    observed = torch.randn(2, 3, position_width)
    reference = torch.randn(2, 3, 5)

    resolved = runtime.resolve(
        observed,
        positions=torch.ones(2, 3, dtype=torch.long),
        chrom_ids=None,
        mask=torch.ones(2, 3, dtype=torch.bool),
        reference=reference,
    )

    assert resolved is observed
    assert torch.equal(resolved, observed)


def test_observed_absolute_runtime_validates_batch_variant_alignment():
    runtime = ObservedAbsolutePositionRuntime()
    with pytest.raises(ValueError, match="same rank"):
        runtime.resolve(
            torch.ones(2, 3),
            positions=torch.ones(2, 3, dtype=torch.long),
            chrom_ids=None,
            mask=None,
            reference=torch.ones(2, 3, 1),
        )
    with pytest.raises(ValueError, match="leading batch/variant"):
        runtime.resolve(
            torch.ones(2, 4, 64),
            positions=torch.ones(2, 4, dtype=torch.long),
            chrom_ids=None,
            mask=None,
            reference=torch.ones(2, 3, 5),
        )


def test_absolute_runtime_is_parameterless_and_not_registered_in_sieve_state_dict():
    runtime = ObservedAbsolutePositionRuntime()
    model = _make_sieve(69)

    assert not isinstance(runtime, nn.Module)
    assert not hasattr(runtime, "parameters")
    assert not hasattr(runtime, "buffers")
    assert not any("_absolute_position_runtime" in key for key in model.state_dict())


def test_sieve_split_composition_uses_runtime_returned_tensor():
    model = _make_sieve(69)
    features, content, absolute_position = _make_features(69, 5)
    positions, gene_ids, mask, _ = _make_metadata()
    replacement = absolute_position + 10.0
    captured = {}

    class RecordingAbsoluteRuntime:
        def __init__(self):
            self.observed = None

        def resolve(self, observed, positions, chrom_ids, mask, reference):
            self.observed = observed
            return replacement

    runtime = RecordingAbsoluteRuntime()
    model._absolute_position_runtime = runtime

    def capture_input(_module, args):
        captured["encoder_input"] = args[0].detach().clone()

    handle = model.variant_encoder.register_forward_pre_hook(capture_input)
    try:
        model(
            features,
            positions,
            gene_ids,
            mask,
            content_features=content,
            absolute_position_features=absolute_position,
        )
    finally:
        handle.remove()

    assert runtime.observed is absolute_position
    assert torch.equal(
        captured["encoder_input"],
        compose_legacy_variant_features_torch(content, replacement),
    )


def test_sieve_historical_fallback_does_not_call_absolute_runtime():
    model = _make_sieve(69)
    features, _, _ = _make_features(69, 5)
    positions, gene_ids, mask, _ = _make_metadata()

    class FailingAbsoluteRuntime:
        def resolve(self, *args, **kwargs):
            raise AssertionError("split-only absolute runtime should not be called")

    model._absolute_position_runtime = FailingAbsoluteRuntime()

    model(features, positions, gene_ids, mask)


@pytest.mark.parametrize("with_chroms", [False, True])
def test_legacy_t5_runtime_bias_matches_independent_reference(with_chroms):
    layer = _make_attention(num_chromosomes=2 if with_chroms else 0)
    positions, _, _, chrom_ids = _make_metadata()
    runtime = LegacyT5RelativePositionRuntime(
        num_position_buckets=layer.num_position_buckets,
        max_distance=layer.max_distance,
    )
    chrom_arg = chrom_ids if with_chroms else None

    bias = runtime.compute_bias(
        positions,
        positions,
        layer.position_bias,
        query_chroms=chrom_arg,
        key_chroms=chrom_arg,
    )
    reference = _legacy_position_bias(
        layer,
        positions,
        positions,
        query_chroms=chrom_arg,
        key_chroms=chrom_arg,
    )

    assert torch.equal(bias, reference)
    assert bias.shape == (2, 2, 4, 4)


def test_legacy_t5_runtime_buckets_and_cross_chromosome_row_match_reference():
    layer = _make_attention(num_chromosomes=2)
    positions, _, _, chrom_ids = _make_metadata()
    buckets = _legacy_position_buckets(
        positions,
        positions,
        num_position_buckets=layer.num_position_buckets,
        max_distance=layer.max_distance,
        query_chroms=chrom_ids,
        key_chroms=chrom_ids,
    )

    assert torch.equal(
        buckets,
        _legacy_position_buckets(
            positions,
            positions,
            num_position_buckets=32,
            max_distance=100000,
            query_chroms=chrom_ids,
            key_chroms=chrom_ids,
        ),
    )
    assert buckets[0, 0, 1].item() == layer.num_position_buckets
    assert buckets[1, 0, 1].item() != layer.num_position_buckets


def test_legacy_t5_runtime_adjusts_scores_by_independent_bias_only():
    layer = _make_attention(num_chromosomes=2)
    positions, _, _, chrom_ids = _make_metadata()
    runtime = layer._relative_position_runtime
    base_scores = torch.randn(2, 2, 4, 4)
    query = torch.randn(2, 2, 4, 4)
    key = torch.randn(2, 2, 4, 4)
    independent_bias = _legacy_position_bias(
        layer,
        positions,
        positions,
        query_chroms=chrom_ids,
        key_chroms=chrom_ids,
    )

    adjusted = runtime.adjust_attention_scores(
        base_scores,
        query=query,
        key=key,
        positions=positions,
        chrom_ids=chrom_ids,
        position_bias=layer.position_bias,
    )

    assert torch.equal(adjusted, base_scores + independent_bias)


def test_legacy_t5_runtime_is_parameterless_and_not_a_module():
    runtime = LegacyT5RelativePositionRuntime(num_position_buckets=32, max_distance=100000)

    assert not isinstance(runtime, nn.Module)
    assert not hasattr(runtime, "parameters")
    assert not hasattr(runtime, "buffers")


@pytest.mark.parametrize("with_chroms", [False, True])
def test_attention_forward_matches_independent_historical_reference(with_chroms):
    layer = _make_attention(num_chromosomes=2 if with_chroms else 0)
    torch.manual_seed(3100 + int(with_chroms))
    x = torch.randn(2, 4, 8, requires_grad=True)
    positions, _, mask, chrom_ids = _make_metadata()
    chrom_arg = chrom_ids if with_chroms else None

    output, weights = layer(x, positions, mask, return_attention=True, chrom_ids=chrom_arg)
    reference_output, reference_weights = _legacy_attention_reference(
        layer,
        x,
        positions,
        mask,
        chrom_ids=chrom_arg,
    )

    assert torch.equal(output, reference_output)
    assert torch.equal(weights, reference_weights)
    assert torch.equal(
        layer._compute_position_bias(
            positions,
            positions,
            query_chroms=chrom_arg,
            key_chroms=chrom_arg,
        ),
        _legacy_position_bias(
            layer,
            positions,
            positions,
            query_chroms=chrom_arg,
            key_chroms=chrom_arg,
        ),
    )


def test_attention_padding_and_cross_chromosome_behavior_remain_legacy():
    layer = _make_attention(num_chromosomes=2)
    x = torch.randn(2, 4, 8)
    positions, _, mask, chrom_ids = _make_metadata()
    real_chrom_ids = chrom_ids[mask]

    _, weights = layer(x, positions, mask, return_attention=True, chrom_ids=chrom_ids)

    assert torch.all((0 <= real_chrom_ids) & (real_chrom_ids < layer.num_chromosomes))
    assert torch.all(weights[1, :, :, 1] == 0)
    assert torch.all(weights[0, :, 0, 1] > 0)
    assert chrom_ids[0, 0].item() == 0
    assert mask[0, 0].item() is True
    assert chrom_ids[1, 1].item() == 0
    assert mask[1, 1].item() is False


def test_attention_gradient_matches_independent_historical_reference():
    layer = _make_attention(num_chromosomes=2)
    positions, _, mask, chrom_ids = _make_metadata()
    x_reference = torch.randn(2, 4, 8, requires_grad=True)
    x_runtime = x_reference.detach().clone().requires_grad_(True)

    reference_output, _ = _legacy_attention_reference(
        layer,
        x_reference,
        positions,
        mask,
        chrom_ids=chrom_ids,
    )
    reference_output.sum().backward()

    output, _ = layer(x_runtime, positions, mask, chrom_ids=chrom_ids)
    output.sum().backward()

    assert torch.equal(x_runtime.grad, x_reference.grad)


@pytest.mark.parametrize(
    ("level_name", "input_dim", "content_dim", "num_chromosomes"),
    [
        ("L0", 1, 1, 0),
        ("L1", 65, 1, 0),
        ("L2", 69, 5, 2),
        ("L3", 71, 7, 0),
        ("L4", 71, 7, 2),
    ],
)
def test_sieve_l0_to_l4_split_and_historical_paths_match_legacy_reference(
    level_name,
    input_dim,
    content_dim,
    num_chromosomes,
):
    model = _make_sieve(input_dim, num_chromosomes=num_chromosomes)
    features, content, absolute_position = _make_features(input_dim, content_dim)
    positions, gene_ids, mask, chrom_ids = _make_metadata()
    chrom_arg = chrom_ids if num_chromosomes else None
    captured = {}

    def capture_input(_module, args):
        captured["encoder_input"] = args[0].detach().clone()

    reference = _legacy_sieve_reference(
        model,
        features,
        positions,
        gene_ids,
        mask,
        chrom_ids=chrom_arg,
    )

    historical_logits, historical_mid = model(
        features,
        positions,
        gene_ids,
        mask,
        return_attention=True,
        return_intermediate=True,
        chrom_ids=chrom_arg,
    )
    handle = model.variant_encoder.register_forward_pre_hook(capture_input)
    try:
        split_logits, split_mid = model(
            None,
            positions,
            gene_ids,
            mask,
            return_attention=True,
            return_intermediate=True,
            chrom_ids=chrom_arg,
            content_features=content,
            absolute_position_features=absolute_position,
        )
    finally:
        handle.remove()
    historical_patterns = model.get_attention_patterns(
        features,
        positions,
        gene_ids,
        mask,
        chrom_ids=chrom_arg,
    )
    split_patterns = model.get_attention_patterns(
        None,
        positions,
        gene_ids,
        mask,
        chrom_ids=chrom_arg,
        content_features=content,
        absolute_position_features=absolute_position,
    )

    assert level_name in {"L0", "L1", "L2", "L3", "L4"}
    assert torch.equal(historical_logits, reference["logits"])
    assert torch.equal(split_logits, historical_logits)
    assert torch.equal(captured["encoder_input"], features)
    for key in ("variant_embeddings", "attended_embeddings", "gene_embeddings"):
        assert torch.equal(historical_mid[key], reference[key])
        assert torch.equal(split_mid[key], historical_mid[key])
    for actual, expected in zip(
        historical_mid["attention_weights"], reference["attention_weights"], strict=True
    ):
        assert torch.equal(actual, expected)
    for split_attention, historical_attention in zip(
        split_mid["attention_weights"], historical_mid["attention_weights"], strict=True
    ):
        assert torch.equal(split_attention, historical_attention)
    for split_attention, historical_attention in zip(
        split_patterns, historical_patterns, strict=True
    ):
        assert torch.equal(split_attention, historical_attention)


def test_sieve_historical_and_split_gradients_match_legacy_columns():
    model = _make_sieve(69, num_chromosomes=2)
    features, content, absolute_position = _make_features(69, 5)
    positions, gene_ids, mask, chrom_ids = _make_metadata()
    historical_features = features.detach().clone().requires_grad_(True)
    reference_features = features.detach().clone().requires_grad_(True)
    content = content.detach().clone().requires_grad_(True)
    absolute_position = absolute_position.detach().clone().requires_grad_(True)

    reference = _legacy_sieve_reference(
        model,
        reference_features,
        positions,
        gene_ids,
        mask,
        chrom_ids=chrom_ids,
    )
    reference["logits"].sum().backward()
    reference_grad = reference_features.grad.detach().clone()

    model.zero_grad(set_to_none=True)
    historical_logits, _ = model(
        historical_features,
        positions,
        gene_ids,
        mask,
        chrom_ids=chrom_ids,
    )
    historical_logits.sum().backward()
    historical_grad = historical_features.grad.detach().clone()

    model.zero_grad(set_to_none=True)
    split_logits, _ = model(
        None,
        positions,
        gene_ids,
        mask,
        chrom_ids=chrom_ids,
        content_features=content,
        absolute_position_features=absolute_position,
    )
    split_logits.sum().backward()

    assert torch.equal(historical_grad, reference_grad)
    assert torch.equal(
        content.grad, torch.cat([historical_grad[..., :1], historical_grad[..., 65:]], dim=-1)
    )
    assert torch.equal(absolute_position.grad, historical_grad[..., 1:65])


def test_runtime_objects_do_not_change_state_dict_keys_shapes_or_parameter_count():
    model = _make_sieve(69, num_chromosomes=2)
    state_keys_before = set(model.state_dict())
    state_shapes = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    submodule_parameter_count = sum(
        parameter.numel()
        for module in (
            model.variant_encoder,
            model.attention,
            model.gene_aggregator,
            model.classifier,
        )
        for parameter in module.parameters()
    )

    assert parameter_count == submodule_parameter_count
    assert model.variant_encoder.encoder[0].weight.shape == (10, 69)
    assert "attention.attention_layers.0.position_bias.weight" in state_shapes
    assert "attention.attention_layers.1.position_bias.weight" in state_shapes
    assert "attention.attention_layers.0.chrom_embedding.weight" in state_shapes
    assert "attention.attention_layers.1.chrom_embedding.weight" in state_shapes
    assert state_shapes["attention.attention_layers.0.position_bias.weight"] == (33, 2)
    assert state_shapes["attention.attention_layers.0.chrom_embedding.weight"] == (3, 8)
    assert not any("position_runtime" in key for key in state_shapes)
    assert not any("_relative_position_runtime" in key for key in state_shapes)
    assert not any("_absolute_position_runtime" in key for key in state_shapes)
    assert not isinstance(model._absolute_position_runtime, nn.Module)
    for layer in model.attention.attention_layers:
        assert not isinstance(layer._relative_position_runtime, nn.Module)

    del model._absolute_position_runtime
    for layer in model.attention.attention_layers:
        del layer._relative_position_runtime

    state_keys_after = set(model.state_dict())
    state_shapes_after = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    parameter_count_after = sum(parameter.numel() for parameter in model.parameters())

    assert state_keys_after == state_keys_before
    assert state_shapes_after == state_shapes
    assert parameter_count_after == parameter_count


def test_legacy_checkpoint_upgrade_preserves_overlapping_position_bias_rows():
    source_model = _make_sieve(69, num_chromosomes=2)
    old_state = copy.deepcopy(source_model.state_dict())
    for key, tensor in list(old_state.items()):
        if key.endswith("position_bias.weight"):
            old_state[key] = tensor[:32].clone()

    destination = _make_sieve(69, num_chromosomes=2)
    destination_before = copy.deepcopy(destination.state_dict())

    load_state_dict_with_legacy_upgrade(destination, old_state)

    loaded_state = destination.state_dict()
    for key, tensor in old_state.items():
        if key.endswith("position_bias.weight"):
            assert torch.equal(loaded_state[key][:32], tensor)
            assert torch.equal(loaded_state[key][32], destination_before[key][32])
