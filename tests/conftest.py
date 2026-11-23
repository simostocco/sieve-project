"""Shared fixtures for the thesis verification tests on ``test_data/small``."""

import contextlib
import io
from pathlib import Path

import pytest
import torch

from src.encoding.chunked_dataset import ChunkedVariantDataset
from src.encoding.levels import AnnotationLevel

SMALL_DATA_DIR = Path(__file__).parent.parent / "test_data" / "small"


@pytest.fixture(scope="session")
def small_dataset() -> ChunkedVariantDataset:
    """L3 chunked dataset built exactly as ``train.py`` does (default 3000-variant chunks).

    Every sample has fewer than 3000 variants, so each sample is one chunk.
    """
    samples = torch.load(SMALL_DATA_DIR / "preprocessed_test.pt", weights_only=False)["samples"]
    with contextlib.redirect_stdout(io.StringIO()):
        return ChunkedVariantDataset(
            samples=samples, annotation_level=AnnotationLevel.L3, chunk_size=3000, overlap=0
        )


@pytest.fixture(scope="session")
def config_g_args() -> list[str]:
    """Configuration G as ``train.py`` CLI arguments."""
    return [
        "--latent-dim", "32",
        "--hidden-dim", "64",
        "--num-heads", "4",
        "--num-attention-layers", "1",
        "--lr", "1e-5",
        "--lambda-attr", "0.1",
    ]
