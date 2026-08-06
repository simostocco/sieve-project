import numpy as np
import pytest
import torch

from src.data import SampleVariants, VariantRecord
from src.encoding import (
    AnnotationLevel,
    ChunkedVariantDataset,
    VariantDataset,
    build_variant_tensor,
    collate_chunks,
    collate_samples,
    compose_legacy_variant_features,
    encode_variants,
    get_content_feature_dimension,
    get_feature_dimension,
    get_legacy_absolute_position_dimension,
    sinusoidal_position_encoding,
    split_legacy_variant_features,
)
from src.models import SIEVE

LEVELS = (
    AnnotationLevel.L0,
    AnnotationLevel.L1,
    AnnotationLevel.L2,
    AnnotationLevel.L3,
    AnnotationLevel.L4,
)


def _historical_features(
    level: AnnotationLevel,
    *,
    rows: int = 2,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    width = get_feature_dimension(level)
    values = np.arange(rows * width, dtype=dtype).reshape(rows, width)
    return np.ascontiguousarray(values)


def _variant(
    pos: int,
    *,
    genotype: int = 1,
    gene: str = "GENE1",
    consequence: str = "missense_variant",
    chrom: str = "1",
    annotations: dict | None = None,
) -> VariantRecord:
    return VariantRecord(
        chrom=chrom,
        pos=pos,
        ref="A",
        alt="T",
        gene=gene,
        consequence=consequence,
        genotype=genotype,
        annotations=annotations or {},
    )


def _sample(
    sample_id: str = "sample1",
    *,
    label: int = 1,
    variants: list[VariantRecord] | None = None,
    sex: str | None = None,
) -> SampleVariants:
    if variants is None:
        variants = [
            _variant(
                100,
                genotype=2,
                gene="GENE1",
                consequence="missense_variant",
                annotations={"sift": 0.05, "polyphen": 0.9},
            ),
            _variant(
                200,
                genotype=1,
                gene="GENE2",
                consequence="synonymous_variant",
                chrom="2",
            ),
            _variant(
                300,
                genotype=0,
                gene="GENE3",
                consequence="stop_gained",
            ),
        ]
    return SampleVariants(sample_id=sample_id, label=label, variants=variants, sex=sex)


def test_content_and_legacy_absolute_position_dimensions_are_authoritative():
    assert [get_content_feature_dimension(level) for level in LEVELS] == [1, 1, 5, 7, 7]
    assert [get_legacy_absolute_position_dimension(level) for level in LEVELS] == [
        0,
        64,
        64,
        64,
        64,
    ]


@pytest.mark.parametrize("level", LEVELS)
def test_split_accepts_historical_widths_without_mutating_inputs(level):
    features = _historical_features(level)
    original = features.copy()

    content, absolute_position = split_legacy_variant_features(features, level)

    assert np.array_equal(features, original)
    assert content.flags.c_contiguous
    assert absolute_position.flags.c_contiguous
    assert content.dtype == features.dtype
    assert absolute_position.dtype == features.dtype
    assert content.shape == (2, get_content_feature_dimension(level))
    assert absolute_position.shape == (
        2,
        get_legacy_absolute_position_dimension(level),
    )


def test_split_rejects_wrong_rank_and_historical_width():
    with pytest.raises(ValueError, match="two-dimensional"):
        split_legacy_variant_features(np.zeros(5, dtype=np.float32), AnnotationLevel.L0)

    with pytest.raises(ValueError, match="historical input dimension"):
        split_legacy_variant_features(
            np.zeros((2, get_feature_dimension(AnnotationLevel.L2) - 1), dtype=np.float32),
            AnnotationLevel.L2,
        )


def test_split_rejects_invalid_annotation_level():
    with pytest.raises(ValueError, match="Unknown annotation level"):
        split_legacy_variant_features(np.zeros((2, 1), dtype=np.float32), "L0")


def test_split_preserves_l0_zero_width_absolute_position_block():
    features = _historical_features(AnnotationLevel.L0)

    content, absolute_position = split_legacy_variant_features(features, AnnotationLevel.L0)

    assert np.array_equal(content, features)
    assert absolute_position.shape == (2, 0)


def test_split_preserves_canonical_content_order_for_l1_to_l4():
    l1 = _historical_features(AnnotationLevel.L1)
    l2 = _historical_features(AnnotationLevel.L2)
    l3 = _historical_features(AnnotationLevel.L3)
    l4 = _historical_features(AnnotationLevel.L4)

    content_l1, _ = split_legacy_variant_features(l1, AnnotationLevel.L1)
    content_l2, _ = split_legacy_variant_features(l2, AnnotationLevel.L2)
    content_l3, _ = split_legacy_variant_features(l3, AnnotationLevel.L3)
    content_l4, _ = split_legacy_variant_features(l4, AnnotationLevel.L4)

    assert np.array_equal(content_l1, l1[:, :1])
    assert np.array_equal(content_l2, np.concatenate([l2[:, :1], l2[:, 65:]], axis=1))
    assert np.array_equal(content_l3, np.concatenate([l3[:, :1], l3[:, 65:]], axis=1))
    assert np.array_equal(content_l4, np.concatenate([l4[:, :1], l4[:, 65:]], axis=1))


@pytest.mark.parametrize("level", LEVELS)
def test_composition_reconstructs_historical_features_exactly(level):
    features = _historical_features(level)
    content, absolute_position = split_legacy_variant_features(features, level)

    recomposed = compose_legacy_variant_features(content, absolute_position, level)

    assert np.array_equal(recomposed, features)
    assert torch.equal(torch.from_numpy(recomposed), torch.from_numpy(features))
    assert recomposed.flags.c_contiguous


def test_composition_validates_rows_widths_dtype_and_annotation_level():
    content = np.zeros((2, 5), dtype=np.float32)
    position = np.zeros((2, 64), dtype=np.float32)

    with pytest.raises(ValueError, match="same number of rows"):
        compose_legacy_variant_features(content, position[:1], AnnotationLevel.L2)

    with pytest.raises(ValueError, match="content_features width"):
        compose_legacy_variant_features(content[:, :4], position, AnnotationLevel.L2)

    with pytest.raises(ValueError, match="absolute_position_features width"):
        compose_legacy_variant_features(content, position[:, :63], AnnotationLevel.L2)

    with pytest.raises(ValueError, match="same dtype"):
        compose_legacy_variant_features(content, position.astype(np.float64), AnnotationLevel.L2)

    with pytest.raises(ValueError, match="two-dimensional"):
        compose_legacy_variant_features(content[0], position, AnnotationLevel.L2)

    with pytest.raises(ValueError, match="Unknown annotation level"):
        compose_legacy_variant_features(
            np.zeros((2, 1), dtype=np.float32),
            np.zeros((2, 0), dtype=np.float32),
            "L0",
        )


@pytest.mark.parametrize("level", LEVELS)
def test_zero_row_arrays_split_and_compose(level):
    features = np.empty((0, get_feature_dimension(level)), dtype=np.float32)

    content, absolute_position = split_legacy_variant_features(features, level)
    recomposed = compose_legacy_variant_features(content, absolute_position, level)

    assert content.shape == (0, get_content_feature_dimension(level))
    assert absolute_position.shape == (
        0,
        get_legacy_absolute_position_dimension(level),
    )
    assert np.array_equal(recomposed, features)


@pytest.mark.parametrize("level", LEVELS)
def test_build_variant_tensor_features_match_independent_historical_encoding(level):
    sample = _sample()
    gene_index = {"GENE1": 0, "GENE2": 1, "GENE3": 2}
    positions_np = np.array([variant.pos for variant in sample.variants], dtype=np.int64)
    position_encodings = (
        None
        if level == AnnotationLevel.L0
        else sinusoidal_position_encoding(positions_np, d_model=64)
    )

    expected_features, expected_positions, _ = encode_variants(
        sample.variants,
        level,
        position_encodings,
    )
    tensor = build_variant_tensor(sample, level, gene_index)

    assert np.array_equal(tensor["features"].numpy(), expected_features)
    assert np.array_equal(tensor["positions"].numpy(), expected_positions)


@pytest.mark.parametrize("level", LEVELS)
def test_build_variant_tensor_returns_split_keys_and_preserves_features(level):
    sample = _sample()
    gene_index = {"GENE1": 0, "GENE2": 1, "GENE3": 2}
    chrom_index = {"1": 0, "2": 1}

    tensor = build_variant_tensor(sample, level, gene_index, chrom_index=chrom_index)

    assert {"features", "content_features", "absolute_position_features"}.issubset(tensor)
    assert tensor["features"].dtype == torch.float32
    assert tensor["content_features"].dtype == torch.float32
    assert tensor["absolute_position_features"].dtype == torch.float32
    recomposed = compose_legacy_variant_features(
        tensor["content_features"].numpy(),
        tensor["absolute_position_features"].numpy(),
        level,
    )
    assert np.array_equal(recomposed, tensor["features"].numpy())


def test_build_variant_tensor_empty_samples_have_correct_split_shapes():
    sample = _sample(variants=[])
    gene_index = {}

    l0 = build_variant_tensor(sample, AnnotationLevel.L0, gene_index)
    l3 = build_variant_tensor(sample, AnnotationLevel.L3, gene_index)

    assert l0["features"].shape == (0, 1)
    assert l0["content_features"].shape == (0, 1)
    assert l0["absolute_position_features"].shape == (0, 0)
    assert l3["features"].shape == (0, 71)
    assert l3["content_features"].shape == (0, 7)
    assert l3["absolute_position_features"].shape == (0, 64)
    for tensor in (l0, l3):
        assert tensor["positions"].shape == (0,)
        assert tensor["gene_ids"].shape == (0,)
        assert tensor["mask"].shape == (0,)
        assert tensor["sample_id"] == "sample1"


def test_variant_and_chunked_dataset_items_contain_split_keys():
    samples = [_sample()]

    variant_item = VariantDataset(samples, AnnotationLevel.L3)[0]
    chunked_item = ChunkedVariantDataset(samples, AnnotationLevel.L3, chunk_size=2)[0]

    for item in (variant_item, chunked_item):
        assert "content_features" in item
        assert "absolute_position_features" in item
        recomposed = compose_legacy_variant_features(
            item["content_features"].numpy(),
            item["absolute_position_features"].numpy(),
            AnnotationLevel.L3,
        )
        assert np.array_equal(recomposed, item["features"].numpy())


def test_collate_samples_pads_truncates_and_preserves_split_rows():
    gene_index = {"GENE1": 0, "GENE2": 1, "GENE3": 2}
    first = build_variant_tensor(_sample("s1"), AnnotationLevel.L3, gene_index)
    second = build_variant_tensor(
        _sample("s2", label=0, variants=[_variant(400, gene="GENE1")]),
        AnnotationLevel.L3,
        gene_index,
    )

    batch = collate_samples([first, second], max_variants_per_batch=2)

    assert batch["features"].shape == (2, 2, 71)
    assert batch["content_features"].shape == (2, 2, 7)
    assert batch["absolute_position_features"].shape == (2, 2, 64)
    assert torch.equal(batch["features"][0], first["features"][:2])
    assert torch.equal(batch["content_features"][0], first["content_features"][:2])
    assert torch.equal(
        batch["absolute_position_features"][0],
        first["absolute_position_features"][:2],
    )
    assert torch.equal(batch["mask"], torch.tensor([[True, True], [True, False]]))
    assert torch.equal(batch["content_features"][1, 1], torch.zeros(7))
    assert torch.equal(batch["absolute_position_features"][1, 1], torch.zeros(64))
    assert torch.equal(batch["positions"], torch.tensor([[100, 200], [400, 0]]))
    assert torch.equal(batch["gene_ids"], torch.tensor([[0, 1], [0, 0]]))
    assert torch.equal(batch["labels"], torch.tensor([1, 0]))
    assert batch["sample_ids"] == ["s1", "s2"]


def test_collate_chunks_pads_split_rows_and_preserves_metadata():
    samples = [
        _sample("s1", sex="M"),
        _sample("s2", label=0, variants=[_variant(400, gene="GENE1")], sex="F"),
    ]
    dataset = ChunkedVariantDataset(samples, AnnotationLevel.L3, chunk_size=3)
    items = [dataset[0], dataset[-1]]

    batch = collate_chunks(items)

    assert batch["features"].shape == (2, 3, 71)
    assert batch["content_features"].shape == (2, 3, 7)
    assert batch["absolute_position_features"].shape == (2, 3, 64)
    assert torch.equal(batch["features"][0], items[0]["features"])
    assert torch.equal(batch["content_features"][0], items[0]["content_features"])
    assert torch.equal(
        batch["absolute_position_features"][0],
        items[0]["absolute_position_features"],
    )
    assert torch.equal(batch["mask"], torch.tensor([[True, True, True], [True, False, False]]))
    assert torch.equal(batch["content_features"][1, 1:], torch.zeros((2, 7)))
    assert torch.equal(batch["absolute_position_features"][1, 1:], torch.zeros((2, 64)))
    assert torch.equal(batch["positions"], torch.tensor([[100, 200, 300], [400, 0, 0]]))
    assert torch.equal(batch["gene_ids"], torch.tensor([[0, 1, 2], [0, 0, 0]]))
    assert "chrom_ids" in batch
    assert torch.equal(batch["labels"], torch.tensor([1, 0]))
    assert torch.equal(batch["sex"], torch.tensor([1.0, 0.0]))
    assert batch["sample_ids"] == ["s1", "s2"]
    assert torch.equal(batch["chunk_indices"], torch.tensor([0, 0]))
    assert torch.equal(batch["total_chunks"], torch.tensor([1, 1]))
    assert torch.equal(batch["original_sample_indices"], torch.tensor([0, 1]))


def test_l0_collation_supports_zero_width_absolute_position_tensor():
    samples = [_sample("s1"), _sample("s2", label=0, variants=[_variant(400)])]
    dataset = VariantDataset(samples, AnnotationLevel.L0)

    batch = collate_samples([dataset[0], dataset[1]])

    assert batch["features"].shape == (2, 3, 1)
    assert batch["content_features"].shape == (2, 3, 1)
    assert batch["absolute_position_features"].shape == (2, 3, 0)


def test_all_empty_sample_and_chunk_batches_preserve_split_dimensions():
    empty_samples = [
        _sample("empty1", variants=[]),
        _sample("empty2", label=0, variants=[]),
    ]

    sample_batch = collate_samples(
        [VariantDataset(empty_samples, AnnotationLevel.L3)[i] for i in range(2)]
    )
    chunk_dataset = ChunkedVariantDataset(empty_samples, AnnotationLevel.L3)
    chunk_batch = collate_chunks([chunk_dataset[0], chunk_dataset[1]])

    for batch in (sample_batch, chunk_batch):
        assert batch["features"].shape == (2, 0, 71)
        assert batch["content_features"].shape == (2, 0, 7)
        assert batch["absolute_position_features"].shape == (2, 0, 64)
        assert batch["positions"].shape == (2, 0)
        assert batch["gene_ids"].shape == (2, 0)
        assert batch["mask"].shape == (2, 0)


def test_legacy_manual_sample_and_chunk_dictionaries_keep_historical_schema():
    legacy_sample = {
        "features": torch.ones((1, 3)),
        "positions": torch.tensor([100]),
        "gene_ids": torch.tensor([0]),
        "mask": torch.tensor([True]),
        "label": torch.tensor(1),
        "sample_id": "manual-sample",
    }
    legacy_chunk = {
        **legacy_sample,
        "chunk_idx": 0,
        "total_chunks": 1,
        "original_sample_idx": 0,
        "sex": 1.0,
    }

    sample_batch = collate_samples([legacy_sample])
    chunk_batch = collate_chunks([legacy_chunk])

    assert "content_features" not in sample_batch
    assert "absolute_position_features" not in sample_batch
    assert "content_features" not in chunk_batch
    assert "absolute_position_features" not in chunk_batch


def test_mixed_or_partial_split_schemas_are_rejected():
    split_sample = {
        "features": torch.ones((1, 3)),
        "content_features": torch.ones((1, 1)),
        "absolute_position_features": torch.ones((1, 2)),
        "positions": torch.tensor([100]),
        "gene_ids": torch.tensor([0]),
        "mask": torch.tensor([True]),
        "label": torch.tensor(1),
        "sample_id": "split",
    }
    legacy_sample = {
        key: value
        for key, value in split_sample.items()
        if key not in {"content_features", "absolute_position_features"}
    }
    partial_sample = {
        key: value for key, value in split_sample.items() if key != "absolute_position_features"
    }
    wrong_content_width = {**split_sample, "content_features": torch.ones((1, 2))}
    wrong_position_width = {
        **split_sample,
        "absolute_position_features": torch.ones((1, 3)),
    }

    with pytest.raises(ValueError, match="include both"):
        collate_samples([split_sample, legacy_sample])
    with pytest.raises(ValueError, match="include both"):
        collate_samples([partial_sample])
    with pytest.raises(ValueError, match="Inconsistent content feature width"):
        collate_samples([split_sample, wrong_content_width])
    with pytest.raises(ValueError, match="Inconsistent absolute-position feature width"):
        collate_samples([split_sample, wrong_position_width])

    split_chunk = {
        **split_sample,
        "chunk_idx": 0,
        "total_chunks": 1,
        "original_sample_idx": 0,
        "sex": 1.0,
    }
    legacy_chunk = {
        key: value
        for key, value in split_chunk.items()
        if key not in {"content_features", "absolute_position_features"}
    }
    with pytest.raises(ValueError, match="include both"):
        collate_chunks([split_chunk, legacy_chunk])


@pytest.mark.parametrize(
    ("bad_key", "bad_value", "message"),
    [
        ("content_features", torch.ones(1), "content_features.*two-dimensional"),
        (
            "absolute_position_features",
            torch.ones(2),
            "absolute_position_features.*two-dimensional",
        ),
        (
            "content_features",
            torch.ones((0, 1)),
            "content_features.*row count must match",
        ),
        (
            "content_features",
            torch.ones((2, 1)),
            "content_features.*row count must match",
        ),
        (
            "absolute_position_features",
            torch.ones((0, 2)),
            "absolute_position_features.*row count must match",
        ),
    ],
)
def test_collate_samples_rejects_malformed_split_tensor_rank_and_rows(
    bad_key,
    bad_value,
    message,
):
    sample = {
        "features": torch.ones((1, 3)),
        "content_features": torch.ones((1, 1)),
        "absolute_position_features": torch.ones((1, 2)),
        "positions": torch.tensor([100]),
        "gene_ids": torch.tensor([0]),
        "mask": torch.tensor([True]),
        "label": torch.tensor(1),
        "sample_id": "split",
    }
    sample[bad_key] = bad_value

    with pytest.raises(ValueError, match=message):
        collate_samples([sample])


@pytest.mark.parametrize(
    ("bad_features", "message"),
    [
        (object(), "features.*torch.Tensor"),
        (torch.tensor(1.0), "features.*two-dimensional"),
        (torch.ones(1), "features.*two-dimensional"),
    ],
)
def test_collate_samples_rejects_malformed_split_aware_features(
    bad_features,
    message,
):
    sample = {
        "features": bad_features,
        "content_features": torch.ones((1, 1)),
        "absolute_position_features": torch.ones((1, 2)),
        "positions": torch.tensor([100]),
        "gene_ids": torch.tensor([0]),
        "mask": torch.tensor([True]),
        "label": torch.tensor(1),
        "sample_id": "split",
    }

    with pytest.raises(ValueError, match=message):
        collate_samples([sample])


def test_collate_chunks_applies_split_row_alignment_validation():
    chunk = {
        "features": torch.ones((1, 3)),
        "content_features": torch.ones((2, 1)),
        "absolute_position_features": torch.ones((1, 2)),
        "positions": torch.tensor([100]),
        "gene_ids": torch.tensor([0]),
        "mask": torch.tensor([True]),
        "label": torch.tensor(1),
        "sample_id": "split",
        "chunk_idx": 0,
        "total_chunks": 1,
        "original_sample_idx": 0,
        "sex": 1.0,
    }

    with pytest.raises(ValueError, match="content_features.*row count must match"):
        collate_chunks([chunk])


def test_model_still_consumes_historical_features_and_keeps_state_dict_shapes():
    for level in (AnnotationLevel.L0, AnnotationLevel.L3):
        input_dim = get_feature_dimension(level)
        model = SIEVE(input_dim=input_dim, num_genes=3, hidden_dim=8, latent_dim=4)
        first_linear = model.variant_encoder.encoder[0]

        assert first_linear.in_features == input_dim
        assert model.state_dict()["variant_encoder.encoder.0.weight"].shape == (
            8,
            input_dim,
        )
