"""Tests for Phase 8B2 learned-binned absolute-position runtime execution."""

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
from src.encoding.position_layout import LearnedBinnedAbsolutePositionLayout
from src.models.position_runtime import (
    LearnedBinnedAbsolutePositionRuntime,
    NoAbsolutePositionRuntime,
    ObservedAbsolutePositionRuntime,
    SinusoidalAbsolutePositionRuntime,
    validate_attention_runtime_support,
    validate_learned_binned_layout_settings,
    validate_phase7_runtime_support,
)
from src.models.sieve import SIEVE


def _layout() -> LearnedBinnedAbsolutePositionLayout:
    return LearnedBinnedAbsolutePositionLayout(
        schema_version=1,
        coordinate_origin=1,
        layout="chromosome_local_contiguous",
        chromosome_lengths_bp=(20, 15),
        bins_per_chromosome=(2, 2),
        num_embeddings=4,
    )


def _resolve_learned(
    *,
    relative: RelativePositionEncoding = RelativePositionEncoding.NONE,
    chromosome: ChromosomeEncoding = ChromosomeEncoding.NONE,
    cross_policy: CrossChromosomePolicy = CrossChromosomePolicy.SEPARATE,
):
    return resolve_position_encoding_config(
        PositionEncodingRequest(
            preset=PositionPreset.CUSTOM,
            absolute_position_encoding=AbsolutePositionEncoding.LEARNED_BINNED,
            relative_position_encoding=relative,
            chromosome_encoding=chromosome,
            cross_chromosome_policy=cross_policy,
            position_dim=4,
            position_bin_size=10,
        ),
        AnnotationLevel.L3,
        latent_dim=8,
        num_heads=2,
        num_chromosomes=2,
    )


def _resolve_custom(
    absolute: AbsolutePositionEncoding,
):
    return resolve_position_encoding_config(
        PositionEncodingRequest(
            preset=PositionPreset.CUSTOM,
            absolute_position_encoding=absolute,
            relative_position_encoding=RelativePositionEncoding.NONE,
            chromosome_encoding=ChromosomeEncoding.NONE,
            cross_chromosome_policy=CrossChromosomePolicy.SEPARATE,
            position_dim=4 if absolute is AbsolutePositionEncoding.SINUSOIDAL else None,
        ),
        AnnotationLevel.L3,
        latent_dim=8,
        num_heads=2,
        num_chromosomes=0,
    )


def _resolve_legacy():
    return resolve_position_encoding_config(
        PositionEncodingRequest(),
        AnnotationLevel.L3,
        latent_dim=8,
        num_heads=2,
        num_chromosomes=2,
    )


def _embedding() -> nn.Embedding:
    embedding = nn.Embedding(4, 4)
    with torch.no_grad():
        embedding.weight.copy_(
            torch.tensor(
                [
                    [0.0, 0.1, 0.2, 0.3],
                    [1.0, 1.1, 1.2, 1.3],
                    [2.0, 2.1, 2.2, 2.3],
                    [3.0, 3.1, 3.2, 3.3],
                ]
            )
        )
    return embedding


def _runtime() -> LearnedBinnedAbsolutePositionRuntime:
    return LearnedBinnedAbsolutePositionRuntime(
        embedding=_embedding(),
        layout=_layout(),
        bin_size_bp=10,
        position_dim=4,
    )


def _model(config=None, layout=None) -> SIEVE:
    return SIEVE(
        input_dim=71 if config is None else config.input_dim,
        num_genes=4,
        latent_dim=8,
        hidden_dim=10,
        num_heads=2,
        num_attention_layers=1,
        classifier_hidden_dim=12,
        dropout=0.0,
        num_chromosomes=0 if config is None else config.chromosome.num_chromosomes,
        position_encoding=config,
        learned_binned_position_layout=layout,
    )


def _batch(config):
    content = torch.arange(1, 1 + 4 * config.content_dim, dtype=torch.float32).reshape(
        1,
        4,
        config.content_dim,
    )
    observed = torch.full((1, 4, 99), 123.0)
    positions = torch.tensor([[1, 10, 11, 0]], dtype=torch.long)
    chrom_ids = torch.tensor([[0, 0, 1, 999]], dtype=torch.long)
    gene_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    mask = torch.tensor([[True, True, True, False]])
    return content, observed, positions, chrom_ids, gene_ids, mask


@pytest.mark.parametrize(
    "relative",
    [
        RelativePositionEncoding.NONE,
        RelativePositionEncoding.T5_BUCKET,
        RelativePositionEncoding.ALIBI_FIXED,
        RelativePositionEncoding.ALIBI_LEARNED,
    ],
)
def test_attention_validation_accepts_learned_binned_absolute_for_supported_attention(relative):
    config = _resolve_learned(relative=relative)

    validate_attention_runtime_support(config)


def test_attention_validation_accepts_learned_binned_with_rope_relative():
    config = resolve_position_encoding_config(
        PositionEncodingRequest(
            preset=PositionPreset.CUSTOM,
            absolute_position_encoding=AbsolutePositionEncoding.LEARNED_BINNED,
            relative_position_encoding=RelativePositionEncoding.ROPE,
            chromosome_encoding=ChromosomeEncoding.NONE,
            cross_chromosome_policy=CrossChromosomePolicy.SEPARATE,
            position_dim=4,
            position_bin_size=10,
            rope_coordinate_scale=1.0,
            rope_base=10000.0,
        ),
        AnnotationLevel.L3,
        latent_dim=8,
        num_heads=2,
        num_chromosomes=2,
    )

    validate_attention_runtime_support(config)


def test_phase7_external_gate_still_rejects_learned_binned():
    with pytest.raises(NotImplementedError, match="learned_binned"):
        validate_phase7_runtime_support(_resolve_learned())


def test_runtime_row_identity_padding_safety_and_observed_absolute_non_authority():
    runtime = _runtime()
    reference = torch.zeros(1, 6, 7)
    observed_a = torch.randn(1, 6, 64)
    observed_b = observed_a + 1000.0
    positions = torch.tensor([[1, 10, 11, 1, 10, 0]], dtype=torch.long)
    chrom_ids = torch.tensor([[0, 0, 0, 1, 1, 999]], dtype=torch.long)
    mask = torch.tensor([[True, True, True, True, True, False]])

    resolved_a = runtime.resolve(observed_a, positions, chrom_ids, mask, reference)
    resolved_b = runtime.resolve(observed_b, positions, chrom_ids, mask, reference)

    assert torch.equal(resolved_a, resolved_b)
    assert torch.equal(resolved_a[0, 0], runtime.embedding.weight[0])
    assert torch.equal(resolved_a[0, 1], runtime.embedding.weight[0])
    assert torch.equal(resolved_a[0, 2], runtime.embedding.weight[1])
    assert torch.equal(resolved_a[0, 3], runtime.embedding.weight[2])
    assert torch.equal(resolved_a[0, 4], runtime.embedding.weight[2])
    assert torch.equal(resolved_a[0, 5], torch.zeros(4))


@pytest.mark.parametrize("integer_dtype", [torch.int16, torch.uint8])
def test_runtime_canonicalizes_narrow_integer_inputs_to_long_for_lookup(integer_dtype):
    runtime = _runtime()
    reference = torch.zeros(1, 4, 7)
    observed = torch.randn(1, 4, 64)
    positions = torch.tensor([[1, 11, 15, 0]], dtype=torch.long)
    chrom_ids = torch.tensor([[0, 0, 1, 0]], dtype=torch.long)
    mask = torch.tensor([[True, True, True, False]])

    expected = runtime.resolve(observed, positions, chrom_ids, mask, reference)
    actual = runtime.resolve(
        observed,
        positions.to(dtype=integer_dtype),
        chrom_ids.to(dtype=integer_dtype),
        mask,
        reference,
    )

    assert torch.equal(actual, expected)


def test_runtime_accepts_exact_chromosome_end_coordinates_and_rejects_past_end():
    runtime = _runtime()
    reference = torch.zeros(1, 2, 7)
    observed = torch.zeros(1, 2, 64)
    mask = torch.tensor([[True, True]])

    resolved = runtime.resolve(
        observed,
        torch.tensor([[20, 15]], dtype=torch.long),
        torch.tensor([[0, 1]], dtype=torch.long),
        mask,
        reference,
    )

    assert torch.equal(resolved[0, 0], runtime.embedding.weight[1])
    assert torch.equal(resolved[0, 1], runtime.embedding.weight[3])

    with pytest.raises(ValueError, match="chromosome length"):
        runtime.resolve(
            observed[:, :1],
            torch.tensor([[16]], dtype=torch.long),
            torch.tensor([[1]], dtype=torch.long),
            mask[:, :1],
            reference[:, :1],
        )


@pytest.mark.parametrize(
    ("positions", "chrom_ids", "mask", "message"),
    [
        (torch.tensor([[0]]), torch.tensor([[0]]), torch.tensor([[True]]), ">= 1"),
        (torch.tensor([[21]]), torch.tensor([[0]]), torch.tensor([[True]]), "chromosome length"),
        (torch.tensor([[1]]), torch.tensor([[-1]]), torch.tensor([[True]]), "0 <= chrom_id"),
        (torch.tensor([[1]]), torch.tensor([[2]]), torch.tensor([[True]]), "0 <= chrom_id"),
        (torch.tensor([[1.0]]), torch.tensor([[0]]), torch.tensor([[True]]), "integer"),
        (torch.tensor([[1]]), torch.tensor([[0.0]]), torch.tensor([[True]]), "integer"),
        (torch.tensor([[1]]), torch.tensor([[0]]), torch.tensor([[1]]), "boolean"),
    ],
)
def test_runtime_rejects_malformed_real_inputs(positions, chrom_ids, mask, message):
    with pytest.raises(ValueError, match=message):
        _runtime().resolve(
            torch.zeros(1, 1, 64),
            positions,
            chrom_ids,
            mask,
            torch.zeros(1, 1, 7),
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"positions": None}, "positions"),
        ({"chrom_ids": None}, "chrom_ids"),
        ({"mask": None}, "mask"),
        ({"reference": torch.zeros(7)}, "rank"),
        ({"positions": torch.ones(1, 2, dtype=torch.long)}, "positions shape"),
        ({"chrom_ids": torch.ones(1, 2, dtype=torch.long)}, "chrom_ids shape"),
        ({"mask": torch.ones(1, 2, dtype=torch.bool)}, "mask shape"),
    ],
)
def test_runtime_rejects_missing_or_mismatched_inputs(kwargs, message):
    values = {
        "observed_absolute_position_features": torch.zeros(1, 1, 64),
        "positions": torch.ones(1, 1, dtype=torch.long),
        "chrom_ids": torch.zeros(1, 1, dtype=torch.long),
        "mask": torch.ones(1, 1, dtype=torch.bool),
        "reference": torch.zeros(1, 1, 7),
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        _runtime().resolve(**values)


def test_runtime_gradients_reach_selected_embedding_rows_only():
    runtime = _runtime()
    positions = torch.tensor([[1, 11]], dtype=torch.long)
    chrom_ids = torch.tensor([[0, 1]], dtype=torch.long)
    mask = torch.tensor([[True, True]])

    output = runtime.resolve(
        torch.zeros(1, 2, 64),
        positions,
        chrom_ids,
        mask,
        torch.zeros(1, 2, 7),
    )
    output.sum().backward()

    grad = runtime.embedding.weight.grad
    assert torch.all(grad[0] != 0)
    assert torch.equal(grad[1], torch.zeros_like(grad[1]))
    assert torch.equal(grad[2], torch.zeros_like(grad[2]))
    assert torch.all(grad[3] != 0)


def test_direct_sieve_learned_binned_executes_and_passes_authoritative_config_to_attention():
    config = _resolve_learned()
    model = _model(config, _layout())
    model.eval()
    content, observed, positions, chrom_ids, gene_ids, mask = _batch(config)

    logits, intermediates = model(
        None,
        positions,
        gene_ids,
        mask,
        chrom_ids=chrom_ids,
        content_features=content,
        absolute_position_features=observed,
        return_intermediate=True,
    )

    assert logits.shape == (1, 1)
    assert intermediates["variant_embeddings"].shape[-1] == 8
    assert model.position_encoding is config
    assert model.attention.attention_layers[0].position_encoding is config
    assert (
        model.attention.attention_layers[0].position_encoding.absolute.encoding
        is AbsolutePositionEncoding.LEARNED_BINNED
    )


def test_direct_sieve_learned_binned_coexists_with_t5_and_learned_chromosome():
    config = _resolve_learned(
        relative=RelativePositionEncoding.T5_BUCKET,
        chromosome=ChromosomeEncoding.LEARNED,
    )
    model = _model(config, _layout())
    content, observed, positions, chrom_ids, gene_ids, mask = _batch(config)
    chrom_ids = chrom_ids.masked_fill(~mask, 0)

    logits, intermediates = model(
        None,
        positions,
        gene_ids,
        mask,
        chrom_ids=chrom_ids,
        content_features=content,
        absolute_position_features=observed,
        return_attention=True,
    )

    layer = model.attention.attention_layers[0]
    assert logits.shape == (1, 1)
    assert layer.position_bias.weight.shape == (33, 2)
    assert layer.chrom_embedding.weight.shape == (3, 8)
    assert intermediates["attention_weights"][0].shape == (1, 2, 4, 4)


def test_sieve_constructor_validates_learned_binned_layout_rules():
    config = _resolve_learned()
    with pytest.raises(ValueError, match="learned_binned_position_layout"):
        _model(config, None)
    with pytest.raises(ValueError, match="only valid"):
        _model(_resolve_custom(AbsolutePositionEncoding.NONE), _layout())

    bad_layout = LearnedBinnedAbsolutePositionLayout(
        schema_version=1,
        coordinate_origin=1,
        layout="chromosome_local_contiguous",
        chromosome_lengths_bp=(20,),
        bins_per_chromosome=(2,),
        num_embeddings=2,
    )
    with pytest.raises(ValueError, match="length"):
        _model(config, bad_layout)


@pytest.mark.parametrize(
    ("layout_kwargs", "message"),
    [
        ({"schema_version": True}, "schema_version"),
        ({"coordinate_origin": True}, "coordinate_origin"),
        ({"num_embeddings": 4.0}, "num_embeddings"),
    ],
)
def test_direct_layout_validation_rejects_non_strict_scalar_types(layout_kwargs, message):
    values = {
        "schema_version": 1,
        "coordinate_origin": 1,
        "layout": "chromosome_local_contiguous",
        "chromosome_lengths_bp": (20, 15),
        "bins_per_chromosome": (2, 2),
        "num_embeddings": 4,
    }
    values.update(layout_kwargs)
    layout = LearnedBinnedAbsolutePositionLayout(**values)

    with pytest.raises(ValueError, match=message):
        validate_learned_binned_layout_settings(
            layout,
            bin_size_bp=10,
            position_dim=4,
            num_chromosomes=2,
        )


def test_direct_layout_validation_accepts_valid_layout():
    validate_learned_binned_layout_settings(
        _layout(),
        bin_size_bp=10,
        position_dim=4,
        num_chromosomes=2,
    )


def test_state_dict_surfaces_for_absolute_position_strategies():
    none_model = _model(_resolve_custom(AbsolutePositionEncoding.NONE), None)
    sinusoidal_model = _model(_resolve_custom(AbsolutePositionEncoding.SINUSOIDAL), None)
    legacy_model = _model(_resolve_legacy(), None)
    learned_model = _model(_resolve_learned(), _layout())

    assert isinstance(none_model._absolute_position_runtime, NoAbsolutePositionRuntime)
    assert isinstance(
        sinusoidal_model._absolute_position_runtime,
        SinusoidalAbsolutePositionRuntime,
    )
    assert isinstance(legacy_model._absolute_position_runtime, ObservedAbsolutePositionRuntime)
    assert "absolute_position_embedding.weight" not in none_model.state_dict()
    assert "absolute_position_embedding.weight" not in sinusoidal_model.state_dict()
    assert "absolute_position_embedding.weight" not in legacy_model.state_dict()
    assert [
        key for key in learned_model.state_dict() if key == "absolute_position_embedding.weight"
    ] == ["absolute_position_embedding.weight"]
    assert learned_model.state_dict()["absolute_position_embedding.weight"].shape == (4, 4)
    assert torch.equal(
        learned_model.absolute_position_embedding.weight,
        torch.zeros_like(learned_model.absolute_position_embedding.weight),
    )
