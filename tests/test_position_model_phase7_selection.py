"""Tests for Phase 7B2 model-side positional runtime selection."""

import copy

import pytest
import torch
import torch.nn as nn

from src.encoding.levels import AnnotationLevel
from src.encoding.position_config import (
    AbsolutePositionEncoding,
    ChromosomeEncoding,
    CrossChromosomePolicy,
    PositionEncodingRequest,
    PositionPreset,
    RelativePositionEncoding,
    resolve_position_encoding_config,
)
from src.models.attention import PositionAwareSparseAttention
from src.models.position_runtime import (
    LegacyT5RelativePositionRuntime,
    NoAbsolutePositionRuntime,
    NoRelativePositionRuntime,
    ObservedAbsolutePositionRuntime,
    SinusoidalAbsolutePositionRuntime,
    T5RelativePositionRuntime,
)
from src.models.sieve import SIEVE


def _resolve_custom(
    *,
    absolute: AbsolutePositionEncoding = AbsolutePositionEncoding.NONE,
    relative: RelativePositionEncoding = RelativePositionEncoding.NONE,
    chromosome: ChromosomeEncoding = ChromosomeEncoding.NONE,
    cross_policy: CrossChromosomePolicy = CrossChromosomePolicy.SEPARATE,
    level: AnnotationLevel = AnnotationLevel.L3,
    latent_dim: int = 8,
    num_heads: int = 2,
    num_chromosomes: int = 3,
    **kwargs,
):
    request = PositionEncodingRequest(
        preset=PositionPreset.CUSTOM,
        absolute_position_encoding=absolute,
        relative_position_encoding=relative,
        chromosome_encoding=chromosome,
        cross_chromosome_policy=cross_policy,
        **kwargs,
    )
    return resolve_position_encoding_config(
        request,
        level,
        latent_dim=latent_dim,
        num_heads=num_heads,
        num_chromosomes=num_chromosomes,
    )


def _resolve_legacy(level: AnnotationLevel = AnnotationLevel.L3, *, num_chromosomes: int = 3):
    return resolve_position_encoding_config(
        PositionEncodingRequest(),
        level,
        latent_dim=8,
        num_heads=2,
        num_chromosomes=num_chromosomes,
    )


def _make_model(config, *, num_chromosomes: int = 0, layers: int = 1) -> SIEVE:
    model = SIEVE(
        input_dim=config.input_dim,
        num_genes=4,
        latent_dim=8,
        hidden_dim=10,
        num_heads=2,
        num_attention_layers=layers,
        classifier_hidden_dim=12,
        dropout=0.0,
        num_chromosomes=num_chromosomes,
        position_encoding=config,
    )
    model.eval()
    return model


def _batch(content_dim: int, position_dim: int = 64):
    content = torch.arange(1, 1 + 3 * content_dim, dtype=torch.float32).reshape(1, 3, content_dim)
    observed_absolute = torch.randn(1, 3, position_dim)
    positions = torch.tensor([[0, 100, 250]], dtype=torch.long)
    gene_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
    mask = torch.tensor([[True, True, False]])
    chrom_ids = torch.tensor([[0, 1, 0]], dtype=torch.long)
    return content, observed_absolute, positions, gene_ids, mask, chrom_ids


def _capture_encoder_input(model: SIEVE, call):
    captured = {}

    def capture(_module, args):
        captured["encoder_input"] = args[0].detach().clone()

    handle = model.variant_encoder.register_forward_pre_hook(capture)
    try:
        output = call()
    finally:
        handle.remove()
    return output, captured["encoder_input"]


def _assert_no_runtime_state_names(model: nn.Module) -> None:
    forbidden = (
        "position_encoding",
        "position_runtime",
        "_relative_position_runtime",
        "_absolute_position_runtime",
    )
    assert not any(any(token in key for token in forbidden) for key in model.state_dict())


@pytest.mark.parametrize(
    "absolute",
    [AbsolutePositionEncoding.NONE, AbsolutePositionEncoding.SINUSOIDAL],
)
@pytest.mark.parametrize(
    "relative",
    [RelativePositionEncoding.NONE, RelativePositionEncoding.T5_BUCKET],
)
@pytest.mark.parametrize(
    "chromosome",
    [ChromosomeEncoding.NONE, ChromosomeEncoding.LEARNED],
)
@pytest.mark.parametrize(
    "cross_policy",
    [CrossChromosomePolicy.SEPARATE, CrossChromosomePolicy.MASK],
)
def test_explicit_custom_combination_state_surfaces(
    absolute,
    relative,
    chromosome,
    cross_policy,
):
    config = _resolve_custom(
        absolute=absolute,
        relative=relative,
        chromosome=chromosome,
        cross_policy=cross_policy,
        num_chromosomes=3,
    )
    model = _make_model(config)
    layer = model.attention.attention_layers[0]

    assert model.input_dim == config.input_dim
    assert isinstance(
        model._absolute_position_runtime,
        (
            NoAbsolutePositionRuntime
            if absolute is AbsolutePositionEncoding.NONE
            else SinusoidalAbsolutePositionRuntime
        ),
    )
    assert isinstance(
        layer._relative_position_runtime,
        (
            NoRelativePositionRuntime
            if relative is RelativePositionEncoding.NONE
            else T5RelativePositionRuntime
        ),
    )
    if relative is RelativePositionEncoding.NONE:
        assert layer.position_bias is None
        assert not any("position_bias.weight" in key for key in model.state_dict())
    else:
        assert layer.position_bias.weight.shape == (config.relative.total_bias_rows, 2)
    if chromosome is ChromosomeEncoding.NONE:
        assert layer.chrom_embedding is None
        assert not any("chrom_embedding.weight" in key for key in model.state_dict())
    else:
        assert layer.chrom_embedding.weight.shape == (4, 8)
        assert torch.equal(
            layer.chrom_embedding.weight, torch.zeros_like(layer.chrom_embedding.weight)
        )
    assert (
        layer.position_encoding.chromosome.requires_chrom_ids
        is config.chromosome.requires_chrom_ids
    )
    _assert_no_runtime_state_names(model)


def test_no_config_state_surface_remains_legacy():
    model = SIEVE(
        input_dim=71,
        num_genes=4,
        latent_dim=8,
        hidden_dim=10,
        num_heads=2,
        num_attention_layers=1,
        dropout=0.0,
    )
    layer = model.attention.attention_layers[0]
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    assert model.position_encoding is None
    assert isinstance(model._absolute_position_runtime, ObservedAbsolutePositionRuntime)
    assert isinstance(layer._relative_position_runtime, LegacyT5RelativePositionRuntime)
    assert layer.position_bias.weight.shape == (33, 2)
    assert layer.chrom_embedding is None
    assert parameter_count == sum(parameter.numel() for parameter in model.parameters())
    _assert_no_runtime_state_names(model)


def test_explicit_resolved_legacy_uses_new_schema_authority():
    config = _resolve_legacy(num_chromosomes=3)
    model = _make_model(config, num_chromosomes=0)
    layer = model.attention.attention_layers[0]

    assert isinstance(model._absolute_position_runtime, ObservedAbsolutePositionRuntime)
    assert isinstance(layer._relative_position_runtime, LegacyT5RelativePositionRuntime)
    assert layer.position_bias.weight.shape == (33, 2)
    assert layer.chrom_embedding.weight.shape == (4, 8)
    assert config.chromosome.requires_chrom_ids is True


@pytest.mark.parametrize("preset", [PositionPreset.LEGACY, PositionPreset.CUSTOM])
def test_explicit_config_rejects_input_dim_mismatch(preset):
    config = _resolve_legacy() if preset is PositionPreset.LEGACY else _resolve_custom()

    with pytest.raises(ValueError, match="input_dim"):
        SIEVE(
            input_dim=config.input_dim + 1,
            num_genes=4,
            latent_dim=8,
            num_heads=2,
            position_encoding=config,
        )


def test_explicit_config_rejects_conflicting_nonzero_num_chromosomes():
    config = _resolve_custom(num_chromosomes=3)

    _make_model(config, num_chromosomes=0)
    _make_model(config, num_chromosomes=3)
    with pytest.raises(ValueError, match="num_chromosomes"):
        _make_model(config, num_chromosomes=2)
    with pytest.raises(ValueError, match="num_chromosomes"):
        PositionAwareSparseAttention(
            latent_dim=8, num_heads=2, num_chromosomes=2, position_encoding=config
        )


def test_custom_model_requires_split_primary_features_even_when_width_matches():
    config = _resolve_custom(absolute=AbsolutePositionEncoding.NONE, num_chromosomes=0)
    model = _make_model(config)
    content, observed_absolute, positions, gene_ids, mask, _ = _batch(config.content_dim)

    with pytest.raises(ValueError, match="custom positional execution requires"):
        model(content, positions, gene_ids, mask)

    logits, encoder_input = _capture_encoder_input(
        model,
        lambda: model(
            None,
            positions,
            gene_ids,
            mask,
            content_features=content,
            absolute_position_features=observed_absolute,
        ),
    )

    assert logits[0].shape == (1, 1)
    assert torch.equal(encoder_input, content)


def test_custom_absolute_none_ignores_observed_absolute_tensor_values():
    config = _resolve_custom(absolute=AbsolutePositionEncoding.NONE, num_chromosomes=0)
    model = _make_model(config)
    content, observed_absolute, positions, gene_ids, mask, _ = _batch(config.content_dim)

    logits_a, encoder_a = _capture_encoder_input(
        model,
        lambda: model(
            None,
            positions,
            gene_ids,
            mask,
            content_features=content,
            absolute_position_features=observed_absolute,
        ),
    )
    logits_b, encoder_b = _capture_encoder_input(
        model,
        lambda: model(
            None,
            positions,
            gene_ids,
            mask,
            content_features=content,
            absolute_position_features=observed_absolute + 1000.0,
        ),
    )

    assert torch.equal(encoder_a, content)
    assert torch.equal(encoder_b, content)
    assert torch.equal(logits_a[0], logits_b[0])


def test_custom_sinusoidal_uses_positions_and_masks_padding_rows():
    config = _resolve_custom(
        absolute=AbsolutePositionEncoding.SINUSOIDAL,
        position_dim=8,
        sinusoidal_coordinate_scale=25.0,
        sinusoidal_max_wavelength=50000.0,
        num_chromosomes=0,
    )
    model = _make_model(config)
    content, observed_absolute, positions, gene_ids, mask, _ = _batch(
        config.content_dim,
        position_dim=8,
    )

    _, encoder_a = _capture_encoder_input(
        model,
        lambda: model(
            None,
            positions,
            gene_ids,
            mask,
            content_features=content,
            absolute_position_features=observed_absolute,
        ),
    )
    _, encoder_b = _capture_encoder_input(
        model,
        lambda: model(
            None,
            positions + torch.tensor([[0, 50, 0]]),
            gene_ids,
            mask,
            content_features=content,
            absolute_position_features=observed_absolute + 1000.0,
        ),
    )

    assert encoder_a.shape[-1] == config.input_dim
    assert not torch.equal(encoder_a[..., 1:9], observed_absolute)
    assert not torch.equal(encoder_a[:, 1, 1:9], encoder_b[:, 1, 1:9])
    assert torch.equal(encoder_a[0, 2, 1:9], torch.zeros(8))


def test_relative_none_separate_has_no_bias_and_allows_cross_chromosome_attention():
    config = _resolve_custom(relative=RelativePositionEncoding.NONE, num_chromosomes=0)
    model = _make_model(config)
    content, observed_absolute, positions, gene_ids, mask, _ = _batch(config.content_dim)
    patterns = model.get_attention_patterns(
        None,
        positions,
        gene_ids,
        mask,
        content_features=content,
        absolute_position_features=observed_absolute,
    )

    layer = model.attention.attention_layers[0]
    assert layer.position_bias is None
    assert not any("position_bias.weight" in key for key in model.state_dict())
    assert torch.all(patterns[0][0, :, 0, 1] > 0)
    with pytest.raises(ValueError, match="No position bias"):
        layer._compute_position_bias(positions, positions)


def test_t5_separate_requires_chrom_ids_and_allows_cross_chromosome_attention():
    config = _resolve_custom(relative=RelativePositionEncoding.T5_BUCKET)
    model = _make_model(config)
    content, observed_absolute, positions, gene_ids, mask, chrom_ids = _batch(config.content_dim)
    layer = model.attention.attention_layers[0]

    with pytest.raises(ValueError, match="chrom_ids"):
        model(
            None,
            positions,
            gene_ids,
            mask,
            content_features=content,
            absolute_position_features=observed_absolute,
        )

    bias = layer._compute_position_bias(positions, positions, chrom_ids, chrom_ids)
    patterns = model.get_attention_patterns(
        None,
        positions,
        gene_ids,
        mask,
        chrom_ids=chrom_ids,
        content_features=content,
        absolute_position_features=observed_absolute,
    )

    assert layer.position_bias.weight.shape == (9 if config.relative.num_buckets == 8 else 33, 2)
    assert torch.equal(bias[:, :, 0, 1], layer.position_bias.weight[-1].reshape(1, 2))
    assert torch.all(patterns[0][0, :, 0, 1] > 0)


@pytest.mark.parametrize(
    "relative", [RelativePositionEncoding.NONE, RelativePositionEncoding.T5_BUCKET]
)
def test_mask_policy_masks_cross_chromosome_attention_and_keeps_self(relative):
    config = _resolve_custom(
        relative=relative,
        cross_policy=CrossChromosomePolicy.MASK,
        num_position_buckets=8 if relative is RelativePositionEncoding.T5_BUCKET else None,
        max_position_distance=1000 if relative is RelativePositionEncoding.T5_BUCKET else None,
    )
    model = _make_model(config)
    content, observed_absolute, positions, gene_ids, mask, chrom_ids = _batch(config.content_dim)

    patterns = model.get_attention_patterns(
        None,
        positions,
        gene_ids,
        mask,
        chrom_ids=chrom_ids,
        content_features=content,
        absolute_position_features=observed_absolute,
    )

    layer = model.attention.attention_layers[0]
    if relative is RelativePositionEncoding.NONE:
        assert layer.position_bias is None
    else:
        assert layer.position_bias.weight.shape == (8, 2)
    assert torch.all(patterns[0][0, :, 0, 1] == 0)
    assert torch.all(patterns[0][0, :, 0, 0] > 0)


def test_chromosome_none_does_not_disable_t5_routing():
    config = _resolve_custom(
        relative=RelativePositionEncoding.T5_BUCKET,
        chromosome=ChromosomeEncoding.NONE,
    )
    model = _make_model(config)
    layer = model.attention.attention_layers[0]

    assert layer.chrom_embedding is None
    assert layer.position_encoding.chromosome.requires_chrom_ids is True


def test_chromosome_learned_embedding_is_direct_zero_init_and_affects_outputs():
    config = _resolve_custom(chromosome=ChromosomeEncoding.LEARNED)
    model = _make_model(config)
    content, observed_absolute, positions, gene_ids, mask, chrom_ids = _batch(config.content_dim)
    layer = model.attention.attention_layers[0]

    assert layer.chrom_embedding.weight.shape == (4, 8)
    assert torch.equal(layer.chrom_embedding.weight, torch.zeros_like(layer.chrom_embedding.weight))
    logits_before, _ = model(
        None,
        positions,
        gene_ids,
        mask,
        chrom_ids=chrom_ids,
        content_features=content,
        absolute_position_features=observed_absolute,
    )
    with torch.no_grad():
        layer.chrom_embedding.weight[1].fill_(3.0)
    logits_after, _ = model(
        None,
        positions,
        gene_ids,
        mask,
        chrom_ids=chrom_ids,
        content_features=content,
        absolute_position_features=observed_absolute,
    )

    assert not torch.equal(logits_before, logits_after)


@pytest.mark.parametrize(
    ("chrom_ids", "mask", "message"),
    [
        (None, torch.tensor([[True, True, False]]), "required"),
        (torch.tensor([0, 1, 0]), torch.tensor([[True, True, False]]), "shape"),
        (torch.tensor([[0, 1]]), torch.tensor([[True, True, False]]), "shape"),
        (torch.tensor([[0, -1, 0]]), torch.tensor([[True, True, False]]), "0 <= chrom_id"),
        (torch.tensor([[0, 3, 0]]), torch.tensor([[True, True, False]]), "0 <= chrom_id"),
    ],
)
def test_explicit_chrom_id_validation(chrom_ids, mask, message):
    config = _resolve_custom(cross_policy=CrossChromosomePolicy.MASK, num_chromosomes=3)
    model = _make_model(config)
    content, observed_absolute, positions, gene_ids, _, _ = _batch(config.content_dim)

    with pytest.raises(ValueError, match=message):
        model(
            None,
            positions,
            gene_ids,
            mask,
            chrom_ids=chrom_ids,
            content_features=content,
            absolute_position_features=observed_absolute,
        )


def test_explicit_chrom_id_validation_accepts_real_zero_and_padded_zero():
    config = _resolve_custom(cross_policy=CrossChromosomePolicy.MASK, num_chromosomes=3)
    model = _make_model(config)
    content, observed_absolute, positions, gene_ids, mask, _ = _batch(config.content_dim)

    model(
        None,
        positions,
        gene_ids,
        mask,
        chrom_ids=torch.tensor([[0, 1, 0]]),
        content_features=content,
        absolute_position_features=observed_absolute,
    )


def test_explicit_config_uses_query_key_padding_safety():
    config = _resolve_custom(relative=RelativePositionEncoding.NONE, num_chromosomes=0)
    model = _make_model(config)
    content, observed_absolute, positions, gene_ids, mask, _ = _batch(config.content_dim)

    patterns = model.get_attention_patterns(
        None,
        positions,
        gene_ids,
        mask,
        content_features=content,
        absolute_position_features=observed_absolute,
    )

    assert torch.all(patterns[0][0, :, 2, :] == 0)
    assert torch.all(patterns[0][0, :, :, 2] == 0)


def test_each_attention_layer_owns_independent_custom_parameters():
    config = _resolve_custom(
        relative=RelativePositionEncoding.T5_BUCKET,
        chromosome=ChromosomeEncoding.LEARNED,
    )
    model = _make_model(config, layers=2)
    first, second = model.attention.attention_layers

    assert first.position_bias is not second.position_bias
    assert first.chrom_embedding is not second.chrom_embedding
    assert first.position_bias.weight.shape == second.position_bias.weight.shape
    assert first.chrom_embedding.weight.shape == second.chrom_embedding.weight.shape


def test_get_attention_patterns_rejects_custom_variant_feature_fallback():
    config = _resolve_custom(relative=RelativePositionEncoding.NONE, num_chromosomes=0)
    model = _make_model(config)
    content, _, positions, gene_ids, mask, _ = _batch(config.content_dim)

    with pytest.raises(ValueError, match="custom positional execution requires"):
        model.get_attention_patterns(content, positions, gene_ids, mask)


def test_no_config_outputs_attention_and_gradients_match_after_constructor_extension():
    torch.manual_seed(7200)
    before = SIEVE(
        input_dim=71,
        num_genes=4,
        latent_dim=8,
        hidden_dim=10,
        num_heads=2,
        num_attention_layers=1,
        dropout=0.0,
    )
    torch.manual_seed(7200)
    after = SIEVE(
        input_dim=71,
        num_genes=4,
        latent_dim=8,
        hidden_dim=10,
        num_heads=2,
        num_attention_layers=1,
        dropout=0.0,
        position_encoding=None,
    )
    after.load_state_dict(copy.deepcopy(before.state_dict()))
    before.eval()
    after.eval()
    features = torch.randn(1, 3, 71, requires_grad=True)
    features_after = features.detach().clone().requires_grad_(True)
    _, _, positions, gene_ids, mask, chrom_ids = _batch(7)

    logits_before, mid_before = before(
        features,
        positions,
        gene_ids,
        mask,
        chrom_ids=chrom_ids,
        return_attention=True,
    )
    logits_after, mid_after = after(
        features_after,
        positions,
        gene_ids,
        mask,
        chrom_ids=chrom_ids,
        return_attention=True,
    )
    logits_before.sum().backward()
    logits_after.sum().backward()

    assert torch.equal(logits_before, logits_after)
    assert torch.equal(mid_before["attention_weights"][0], mid_after["attention_weights"][0])
    assert torch.equal(features.grad, features_after.grad)
