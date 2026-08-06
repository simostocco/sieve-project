import copy

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.chunked_sieve import ChunkedSIEVEModel
from src.models.feature_composition import compose_legacy_variant_features_torch
from src.models.sieve import SIEVE, load_state_dict_with_legacy_upgrade
from src.training.loss import SIEVELoss
from src.training.trainer import Trainer

RTOL = 1e-6
ATOL = 1e-7


def _split_from_historical(features: torch.Tensor, content_dim: int):
    if features.shape[-1] == content_dim:
        return features.clone(), features[..., :0].clone()
    content = torch.cat([features[..., :1], features[..., 65:]], dim=-1)
    position = features[..., 1:65].clone()
    return content, position


def _compose_reference(content: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
    if position.shape[-1] == 0:
        return content
    return torch.cat([content[..., :1], position, content[..., 1:]], dim=-1)


def _make_inputs(input_dim: int, content_dim: int, *, batch: int = 2, variants: int = 4):
    torch.manual_seed(input_dim + content_dim)
    features = torch.randn(batch, variants, input_dim)
    content, position = _split_from_historical(features, content_dim)
    positions = torch.arange(100, 100 + variants).repeat(batch, 1)
    gene_ids = torch.arange(variants).repeat(batch, 1) % 3
    mask = torch.ones(batch, variants, dtype=torch.bool)
    chrom_ids = torch.arange(variants).repeat(batch, 1) % 2
    return features, content, position, positions, gene_ids, mask, chrom_ids


def _make_model(input_dim: int, *, num_chromosomes: int = 0) -> SIEVE:
    torch.manual_seed(1234 + input_dim + num_chromosomes)
    model = SIEVE(
        input_dim=input_dim,
        num_genes=3,
        latent_dim=8,
        hidden_dim=10,
        num_heads=2,
        num_attention_layers=1,
        classifier_hidden_dim=12,
        dropout=0.0,
        num_chromosomes=num_chromosomes,
    )
    model.eval()
    return model


@pytest.mark.parametrize(
    ("input_dim", "content_dim"),
    [(1, 1), (65, 1), (69, 5), (71, 7)],
)
def test_torch_composer_matches_historical_ordering(input_dim, content_dim):
    features, content, position, *_ = _make_inputs(input_dim, content_dim)

    composed = compose_legacy_variant_features_torch(content, position)

    assert torch.equal(composed, features)
    assert composed.dtype == content.dtype
    assert composed.device == content.device
    if position.shape[-1] == 0:
        assert composed is content


def test_torch_composer_l1_to_l4_ordering_is_dosage_position_remaining_content():
    content = torch.tensor([[[1.0, 2.0, 3.0]]])
    position = torch.tensor([[[10.0, 11.0]]])

    composed = compose_legacy_variant_features_torch(content, position)

    assert torch.equal(composed, torch.tensor([[[1.0, 10.0, 11.0, 2.0, 3.0]]]))


def test_torch_composer_preserves_autograd_and_does_not_mutate_inputs():
    content = torch.tensor([[[1.0, 2.0]]], requires_grad=True)
    position = torch.tensor([[[3.0, 4.0]]], requires_grad=True)
    original_content = content.detach().clone()
    original_position = position.detach().clone()

    composed = compose_legacy_variant_features_torch(content, position)
    composed.square().sum().backward()

    assert content.grad is not None
    assert position.grad is not None
    assert torch.equal(content.detach(), original_content)
    assert torch.equal(position.detach(), original_position)


@pytest.mark.parametrize(
    ("content", "position", "message"),
    [
        (object(), torch.ones(1, 1), "content_features.*torch.Tensor"),
        (torch.ones(1, 1), object(), "absolute_position_features.*torch.Tensor"),
        (torch.ones(1, 1), torch.ones(1, 1, 1), "same rank"),
        (torch.ones(1), torch.ones(1), "rank at least 2"),
        (torch.ones(2, 1), torch.ones(3, 1), "leading dimensions"),
        (torch.ones(1, 1), torch.ones(1, 1, dtype=torch.float64), "same dtype"),
        (torch.ones(1, 0), torch.ones(1, 0), "at least one"),
    ],
)
def test_torch_composer_rejects_malformed_inputs(content, position, message):
    with pytest.raises(ValueError, match=message):
        compose_legacy_variant_features_torch(content, position)


def test_torch_composer_rejects_device_mismatch_when_cuda_is_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    with pytest.raises(ValueError, match="same device"):
        compose_legacy_variant_features_torch(
            torch.ones(1, 1, device="cpu"),
            torch.ones(1, 1, device="cuda"),
        )


def test_sieve_historical_split_only_and_priority_paths_are_compatible():
    model = _make_model(69)
    features, content, position, positions, gene_ids, mask, _ = _make_inputs(69, 5)
    different_features = features + 1000.0

    historical_logits, _ = model(features, positions, gene_ids, mask)
    split_logits, _ = model(
        None,
        positions,
        gene_ids,
        mask,
        content_features=content,
        absolute_position_features=position,
    )
    priority_logits, _ = model(
        different_features,
        positions,
        gene_ids,
        mask,
        content_features=content,
        absolute_position_features=position,
    )
    different_logits, _ = model(different_features, positions, gene_ids, mask)

    torch.testing.assert_close(split_logits, historical_logits, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(priority_logits, split_logits, rtol=RTOL, atol=ATOL)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(priority_logits, different_logits, rtol=RTOL, atol=ATOL)


def test_sieve_rejects_partial_missing_and_wrong_width_inputs():
    model = _make_model(69)
    features, content, position, positions, gene_ids, mask, _ = _make_inputs(69, 5)

    with pytest.raises(ValueError, match="supplied together"):
        model(features, positions, gene_ids, mask, content_features=content)
    with pytest.raises(ValueError, match="variant_features is required"):
        model(None, positions, gene_ids, mask)
    with pytest.raises(ValueError, match="input width"):
        model(
            features,
            positions,
            gene_ids,
            mask,
            content_features=content[..., :4],
            absolute_position_features=position,
        )


def test_variant_encoder_receives_exact_composed_tensor_from_split_path():
    model = _make_model(71)
    features, content, position, positions, gene_ids, mask, _ = _make_inputs(71, 7)
    captured = {}

    def capture_input(_module, args):
        captured["encoder_input"] = args[0].detach().clone()

    handle = model.variant_encoder.register_forward_pre_hook(capture_input)
    try:
        model(
            features + 9.0,
            positions,
            gene_ids,
            mask,
            content_features=content,
            absolute_position_features=position,
        )
    finally:
        handle.remove()

    assert torch.equal(captured["encoder_input"], features)


@pytest.mark.parametrize(
    ("input_dim", "content_dim", "num_chromosomes"),
    [(1, 1, 0), (65, 1, 0), (69, 5, 2), (71, 7, 0)],
)
def test_l0_to_l4_model_outputs_intermediates_and_attention_match(
    input_dim,
    content_dim,
    num_chromosomes,
):
    model = _make_model(input_dim, num_chromosomes=num_chromosomes)
    features, content, position, positions, gene_ids, mask, chrom_ids = _make_inputs(
        input_dim,
        content_dim,
    )
    chrom_arg = chrom_ids if num_chromosomes > 0 else None

    historical_logits, historical_mid = model(
        features,
        positions,
        gene_ids,
        mask,
        return_attention=True,
        return_intermediate=True,
        chrom_ids=chrom_arg,
    )
    split_logits, split_mid = model(
        None,
        positions,
        gene_ids,
        mask,
        return_attention=True,
        return_intermediate=True,
        chrom_ids=chrom_arg,
        content_features=content,
        absolute_position_features=position,
    )

    torch.testing.assert_close(split_logits, historical_logits, rtol=RTOL, atol=ATOL)
    for key in ("variant_embeddings", "attended_embeddings", "gene_embeddings"):
        torch.testing.assert_close(split_mid[key], historical_mid[key], rtol=RTOL, atol=ATOL)
    for split_attention, historical_attention in zip(
        split_mid["attention_weights"],
        historical_mid["attention_weights"],
        strict=True,
    ):
        torch.testing.assert_close(split_attention, historical_attention, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize(
    ("input_dim", "content_dim"),
    [(1, 1), (65, 1), (69, 5), (71, 7)],
)
def test_split_gradients_match_historical_feature_gradient_slices(input_dim, content_dim):
    model = _make_model(input_dim)
    features, content, position, positions, gene_ids, mask, _ = _make_inputs(input_dim, content_dim)
    historical_features = features.clone().detach().requires_grad_(True)
    content = content.clone().detach().requires_grad_(True)
    position = position.clone().detach().requires_grad_(True)

    historical_logits, _ = model(historical_features, positions, gene_ids, mask)
    historical_logits.sum().backward()
    historical_grad = historical_features.grad.detach().clone()

    model.zero_grad(set_to_none=True)
    split_logits, _ = model(
        None,
        positions,
        gene_ids,
        mask,
        content_features=content,
        absolute_position_features=position,
    )
    split_logits.sum().backward()

    if position.shape[-1] == 0:
        torch.testing.assert_close(content.grad, historical_grad, rtol=RTOL, atol=ATOL)
        assert position.grad is None
    else:
        expected_content_grad = torch.cat(
            [historical_grad[..., :1], historical_grad[..., 65:]],
            dim=-1,
        )
        torch.testing.assert_close(content.grad, expected_content_grad, rtol=RTOL, atol=ATOL)
        torch.testing.assert_close(position.grad, historical_grad[..., 1:65], rtol=RTOL, atol=ATOL)


def test_get_attention_patterns_accepts_split_path():
    model = _make_model(69)
    features, content, position, positions, gene_ids, mask, _ = _make_inputs(69, 5)

    historical = model.get_attention_patterns(features, positions, gene_ids, mask)
    split = model.get_attention_patterns(
        None,
        positions,
        gene_ids,
        mask,
        content_features=content,
        absolute_position_features=position,
    )

    for split_attention, historical_attention in zip(split, historical, strict=True):
        torch.testing.assert_close(split_attention, historical_attention, rtol=RTOL, atol=ATOL)


def test_state_dict_has_no_composer_surface_and_loads_without_migration_key():
    model = _make_model(69)
    features, content, position, positions, gene_ids, mask, _ = _make_inputs(69, 5)
    before = {key: tuple(value.shape) for key, value in model.state_dict().items()}

    model(features, positions, gene_ids, mask)
    model(
        None,
        positions,
        gene_ids,
        mask,
        content_features=content,
        absolute_position_features=position,
    )
    after = {key: tuple(value.shape) for key, value in model.state_dict().items()}

    assert before == after
    assert not any("composer" in key or "composition" in key for key in before)
    assert model.variant_encoder.encoder[0].weight.shape == (10, 69)

    reloaded = _make_model(69)
    load_state_dict_with_legacy_upgrade(reloaded, copy.deepcopy(model.state_dict()))


def test_chunked_forward_split_path_matches_historical_outputs_and_intermediates():
    base = _make_model(69)
    chunked = ChunkedSIEVEModel(base, aggregation_method="mean")
    features, content, position, positions, gene_ids, mask, _ = _make_inputs(69, 5)
    original_indices = torch.tensor([0, 0])
    chunk_indices = torch.tensor([0, 1])
    total_chunks = torch.tensor([2, 2])

    historical_logits, historical_mid = chunked(
        features,
        positions,
        gene_ids,
        mask,
        chunk_indices,
        total_chunks,
        original_indices,
        return_attention=True,
        return_intermediate=True,
    )
    split_logits, split_mid = chunked(
        None,
        positions,
        gene_ids,
        mask,
        chunk_indices,
        total_chunks,
        original_indices,
        return_attention=True,
        return_intermediate=True,
        content_features=content,
        absolute_position_features=position,
    )

    torch.testing.assert_close(split_logits, historical_logits, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(
        split_mid["gene_embeddings"],
        historical_mid["gene_embeddings"],
        rtol=RTOL,
        atol=ATOL,
    )
    for split_attention, historical_attention in zip(
        split_mid["attention_weights"],
        historical_mid["attention_weights"],
        strict=True,
    ):
        torch.testing.assert_close(split_attention, historical_attention, rtol=RTOL, atol=ATOL)


def test_chunked_helpers_accept_split_path_and_partial_pairs_raise():
    base = _make_model(69)
    chunked = ChunkedSIEVEModel(base, aggregation_method="mean")
    features, content, position, positions, gene_ids, mask, _ = _make_inputs(69, 5)
    original_indices = torch.tensor([0, 0])
    chunk_indices = torch.tensor([0, 1])
    total_chunks = torch.tensor([2, 2])

    embeddings = chunked.get_gene_embeddings(
        None,
        positions,
        gene_ids,
        mask,
        chunk_indices,
        total_chunks,
        original_indices,
        content_features=content,
        absolute_position_features=position,
    )
    attention = chunked.get_attention_patterns(
        None,
        positions,
        gene_ids,
        mask,
        chunk_indices,
        total_chunks,
        original_indices,
        content_features=content,
        absolute_position_features=position,
    )

    assert embeddings.shape == (1, 3, 8)
    assert len(attention) == 1
    with pytest.raises(ValueError, match="supplied together"):
        chunked(
            features,
            positions,
            gene_ids,
            mask,
            chunk_indices,
            total_chunks,
            original_indices,
            content_features=content,
        )


class RecordingChunkBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dim = 3
        self.num_genes = 2
        self.latent_dim = 3
        self.num_covariates = 0
        self.classifier = nn.Sequential(nn.Flatten(start_dim=1), nn.Linear(6, 1))
        self.calls = []

    def forward(
        self,
        features,
        positions,
        gene_ids,
        mask,
        return_intermediate=False,
        return_attention=False,
        return_embeddings=False,
        content_features=None,
        absolute_position_features=None,
    ):
        self.calls.append(
            {
                "features": features,
                "content_features": content_features,
                "absolute_position_features": absolute_position_features,
            }
        )
        source = features
        if content_features is not None or absolute_position_features is not None:
            source = compose_legacy_variant_features_torch(
                content_features,
                absolute_position_features,
            )
        gene_embeddings = torch.zeros(source.shape[0], self.num_genes, self.latent_dim)
        gene_embeddings[:, 0, :] = source[:, 0, :]
        intermediates = {"gene_embeddings": gene_embeddings}
        if return_attention:
            intermediates["attention_weights"] = [torch.ones(source.shape[0], 1, 1, 1)]
        if return_embeddings:
            return gene_embeddings, intermediates
        return self.classifier(gene_embeddings), intermediates


def test_chunked_train_step_moves_and_forwards_split_tensors():
    base = RecordingChunkBase()
    chunked = ChunkedSIEVEModel(base, aggregation_method="mean")
    criterion = SIEVELoss(lambda_attr=0.0)
    batch = {
        "features": torch.zeros(2, 1, 3),
        "content_features": torch.tensor([[[1.0, 3.0]], [[2.0, 4.0]]]),
        "absolute_position_features": torch.tensor([[[5.0]], [[6.0]]]),
        "positions": torch.ones(2, 1, dtype=torch.long),
        "gene_ids": torch.zeros(2, 1, dtype=torch.long),
        "mask": torch.ones(2, 1, dtype=torch.bool),
        "labels": torch.tensor([0, 1]),
        "chunk_indices": torch.tensor([0, 0]),
        "total_chunks": torch.tensor([1, 1]),
        "original_sample_indices": torch.tensor([0, 1]),
    }

    loss_output, predictions = chunked.train_step(batch, criterion, torch.device("cpu"))

    assert loss_output["total"].requires_grad
    assert predictions.shape == (2,)
    assert base.calls[-1]["content_features"].device.type == "cpu"
    assert base.calls[-1]["absolute_position_features"].device.type == "cpu"


class RecordingStandardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_covariates = 0
        self.weight = nn.Parameter(torch.tensor(0.5))
        self.calls = []

    def forward(
        self,
        features,
        positions,
        gene_ids,
        mask,
        covariates=None,
        return_intermediate=False,
        chrom_ids=None,
        content_features=None,
        absolute_position_features=None,
    ):
        self.calls.append(
            {
                "has_split_kwargs": (
                    content_features is not None or absolute_position_features is not None
                ),
                "content_features": content_features,
                "absolute_position_features": absolute_position_features,
                "return_intermediate": return_intermediate,
            }
        )
        source = features
        if content_features is not None or absolute_position_features is not None:
            source = compose_legacy_variant_features_torch(
                content_features,
                absolute_position_features,
            )
        logits = source[:, :, 0].sum(dim=1, keepdim=True) * self.weight
        intermediates = None
        if return_intermediate:
            intermediates = {"variant_embeddings": source}
        return logits, intermediates


def _trainer_for_model(model, tmp_path, *, lambda_attr=0.0):
    return Trainer(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        loss_fn=SIEVELoss(lambda_attr=lambda_attr),
        device="cpu",
        checkpoint_dir=tmp_path,
    )


def _standard_batch(*, include_split=True):
    batch = {
        "features": torch.tensor(
            [[[1.0, 10.0, 2.0]], [[-1.0, 20.0, 3.0]]],
        ),
        "positions": torch.ones(2, 1, dtype=torch.long),
        "gene_ids": torch.zeros(2, 1, dtype=torch.long),
        "mask": torch.ones(2, 1, dtype=torch.bool),
        "labels": torch.tensor([0, 1]),
    }
    if include_split:
        batch["content_features"] = torch.tensor([[[1.0, 2.0]], [[-1.0, 3.0]]])
        batch["absolute_position_features"] = torch.tensor([[[10.0]], [[20.0]]])
    return batch


def test_standard_trainer_train_epoch_and_validate_forward_split_tensors(tmp_path):
    model = RecordingStandardModel()
    trainer = _trainer_for_model(model, tmp_path)
    loader = DataLoader([_standard_batch(include_split=True)], batch_size=None)

    trainer.train_epoch(loader)
    trainer.validate(loader)

    assert model.calls[0]["has_split_kwargs"] is True
    assert model.calls[1]["has_split_kwargs"] is True
    assert model.calls[0]["content_features"].device.type == "cpu"
    assert model.calls[1]["absolute_position_features"].device.type == "cpu"


def test_standard_trainer_legacy_batches_do_not_receive_split_kwargs(tmp_path):
    model = RecordingStandardModel()
    trainer = _trainer_for_model(model, tmp_path)
    loader = DataLoader([_standard_batch(include_split=False)], batch_size=None)

    trainer.train_epoch(loader)

    assert model.calls[-1]["has_split_kwargs"] is False


def test_standard_trainer_attribution_regularisation_threads_split_tensors(tmp_path):
    model = RecordingStandardModel()
    trainer = _trainer_for_model(model, tmp_path, lambda_attr=0.1)
    loader = DataLoader([_standard_batch(include_split=True)], batch_size=None)

    trainer.train_epoch(loader)

    assert model.calls[-1]["has_split_kwargs"] is True
    assert model.calls[-1]["return_intermediate"] is True
