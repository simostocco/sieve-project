"""Tests for bootstrap rank-based null calibration."""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

import scripts.bootstrap_null_calibration as calibration


N_SAMPLES = 100
N_VARIANTS = 300


def _build_variant_catalog() -> pd.DataFrame:
    """Return a deterministic variant catalogue for synthetic tests."""
    chromosomes = np.array(
        ["1"] * 120 + ["2"] * 120 + ["X"] * 30 + ["Y"] * 30,
        dtype=object,
    )
    positions = np.arange(1, N_VARIANTS + 1) * 10

    gene_names: list[str] = []
    gene_ids: list[int] = []
    for idx in range(N_VARIANTS):
        if idx < 3:
            gene_names.append("UNDERPOWERED")
            gene_ids.append(0)
        elif idx < 15:
            gene_names.append("POWERED")
            gene_ids.append(1)
        else:
            gene_id = 2 + (idx - 15) // 10
            gene_names.append(f"GENE_{gene_id}")
            gene_ids.append(gene_id)

    return pd.DataFrame(
        {
            "chromosome": chromosomes,
            "position": positions,
            "gene_name": gene_names,
            "gene_id": gene_ids,
        }
    )


def _write_null_attributions(
    path: Path,
    catalog: pd.DataFrame,
    score_matrix: np.ndarray,
) -> None:
    """Write a mock explain.py-style attributions.npz file."""
    variant_scores = []
    metadata = []
    positions = catalog["position"].to_numpy(dtype=int)
    gene_ids = catalog["gene_id"].to_numpy(dtype=int)
    chromosomes = catalog["chromosome"].to_numpy(dtype=object)

    for sample_idx in range(score_matrix.shape[0]):
        variant_scores.append(score_matrix[sample_idx].astype(float))
        metadata.append(
            {
                "positions": positions.copy(),
                "gene_ids": gene_ids.copy(),
                "chromosomes": chromosomes.copy(),
                "sample_idx": sample_idx,
                "sample_id": f"S{sample_idx:03d}",
                "label": int(sample_idx % 2 == 0),
            }
        )

    np.savez(
        path,
        variant_scores=np.array(variant_scores, dtype=object),
        metadata=np.array(metadata, dtype=object),
    )


def _build_real_rankings(
    catalog: pd.DataFrame,
    null_means: np.ndarray,
    *,
    boost_indices: list[int] | None = None,
    prefix_chr: bool = False,
    add_missing_variant: bool = False,
    permute_null_means: bool = False,
) -> pd.DataFrame:
    """Build a synthetic real rankings dataframe."""
    real_df = catalog.copy()
    real_df["num_samples"] = N_SAMPLES
    real_df["mean_attribution"] = np.array(null_means, copy=True)

    if permute_null_means:
        rng = np.random.default_rng(2024)
        real_df["mean_attribution"] = rng.permutation(real_df["mean_attribution"].to_numpy())

    if boost_indices:
        boost_values = 20.0 + np.arange(len(boost_indices), 0, -1)
        real_df.loc[boost_indices, "mean_attribution"] = boost_values

    if prefix_chr:
        real_df["chromosome"] = real_df["chromosome"].map(lambda chrom: f"chr{chrom}")

    if add_missing_variant:
        real_df = pd.concat(
            [
                real_df,
                pd.DataFrame(
                    [
                        {
                            "chromosome": "chr9" if prefix_chr else "9",
                            "position": 9_999_999,
                            "gene_name": "MISSING_NULL",
                            "gene_id": 9_999,
                            "num_samples": N_SAMPLES,
                            "mean_attribution": 0.0,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    return real_df


def _run_bootstrap(
    tmp_path: Path,
    *,
    real_df: pd.DataFrame,
    null_path: Path,
    n_bootstrap: int = 100,
    seed: int = 42,
    exclude_sex_chroms: bool = False,
    genome_build: str | None = None,
    metadata_genome_build: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run the bootstrap script and return its three outputs."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    real_path = tmp_path / "real_rankings.csv"
    output_path = tmp_path / "rank_calibrated.csv"
    gene_path = tmp_path / "gene_stats.csv"
    summary_path = tmp_path / "summary.yaml"
    metadata_path = tmp_path / "analysis_metadata.yaml"

    real_df.to_csv(real_path, index=False)
    metadata_payload: dict[str, object] = {"n_samples": N_SAMPLES}
    if metadata_genome_build is not None:
        metadata_payload["genome_build"] = metadata_genome_build
    metadata_path.write_text(
        yaml.safe_dump(metadata_payload, sort_keys=False),
        encoding="utf-8",
    )

    argv = [
        "--real-rankings",
        str(real_path),
        "--null-attributions",
        str(null_path),
        "--output",
        str(output_path),
        "--output-gene-stats",
        str(gene_path),
        "--output-summary",
        str(summary_path),
        "--n-bootstrap",
        str(n_bootstrap),
        "--seed",
        str(seed),
        "--n-jobs",
        "1",
    ]
    if genome_build is not None:
        argv.extend(["--genome-build", genome_build])
    if exclude_sex_chroms:
        argv.append("--exclude-sex-chroms")

    exit_code = calibration.main(argv)
    assert exit_code == 0

    return (
        pd.read_csv(output_path),
        pd.read_csv(gene_path),
        yaml.safe_load(summary_path.read_text(encoding="utf-8")),
    )


@pytest.fixture(scope="module")
def synthetic_null_dataset(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """Create a shared synthetic null attributions file."""
    base_dir = tmp_path_factory.mktemp("bootstrap_null_fixture")
    catalog = _build_variant_catalog()

    rng = np.random.default_rng(123)
    base_effects = rng.normal(loc=0.0, scale=0.2, size=N_VARIANTS)
    base_effects[:20] = -2.0
    base_effects[20:40] = 2.0
    noise = rng.normal(loc=0.0, scale=0.15, size=(N_SAMPLES, N_VARIANTS))
    score_matrix = base_effects[None, :] + noise

    null_path = base_dir / "attributions.npz"
    _write_null_attributions(null_path, catalog, score_matrix)

    return {
        "catalog": catalog,
        "null_means": score_matrix.mean(axis=0),
        "null_path": null_path,
    }


def test_bootstrap_reproducibility(
    tmp_path: Path,
    synthetic_null_dataset: dict[str, object],
) -> None:
    """Running with the same seed should reproduce identical outputs."""
    real_df = _build_real_rankings(
        synthetic_null_dataset["catalog"],
        synthetic_null_dataset["null_means"],
        boost_indices=list(range(20)),
    )

    out_a, gene_a, summary_a = _run_bootstrap(
        tmp_path / "run_a",
        real_df=real_df,
        null_path=synthetic_null_dataset["null_path"],
        n_bootstrap=100,
        seed=7,
    )
    out_b, gene_b, summary_b = _run_bootstrap(
        tmp_path / "run_b",
        real_df=real_df,
        null_path=synthetic_null_dataset["null_path"],
        n_bootstrap=100,
        seed=7,
    )

    pd.testing.assert_frame_equal(out_a, out_b)
    pd.testing.assert_frame_equal(gene_a, gene_b)
    assert summary_a == summary_b


@pytest.mark.parametrize("n_bootstrap", [100, 200])
def test_resolution_floor(
    tmp_path: Path,
    synthetic_null_dataset: dict[str, object],
    n_bootstrap: int,
) -> None:
    """Empirical p-values must respect the Phipson-Smyth resolution floor."""
    real_df = _build_real_rankings(
        synthetic_null_dataset["catalog"],
        synthetic_null_dataset["null_means"],
        boost_indices=list(range(20)),
    )
    out_df, _, summary = _run_bootstrap(
        tmp_path / f"resolution_{n_bootstrap}",
        real_df=real_df,
        null_path=synthetic_null_dataset["null_path"],
        n_bootstrap=n_bootstrap,
    )

    expected_floor = 1.0 / (n_bootstrap + 1)
    assert out_df["p_rank_boot"].min() >= expected_floor - 1e-12
    assert np.isclose(
        summary["per_variant"]["p_rank_boot_resolution_floor"],
        expected_floor,
    )
    assert (
        out_df["at_resolution_floor"]
        == np.isclose(out_df["p_rank_boot"], expected_floor)
    ).all()


def test_fdr_monotonicity(
    tmp_path: Path,
    synthetic_null_dataset: dict[str, object],
) -> None:
    """BH-FDR must remain monotone when sorted by p-value."""
    real_df = _build_real_rankings(
        synthetic_null_dataset["catalog"],
        synthetic_null_dataset["null_means"],
        boost_indices=list(range(20)),
    )
    out_df, _, _ = _run_bootstrap(
        tmp_path / "fdr_monotonicity",
        real_df=real_df,
        null_path=synthetic_null_dataset["null_path"],
        n_bootstrap=100,
    )

    sorted_out = out_df.sort_values("p_rank_boot").reset_index(drop=True)
    diffs = np.diff(sorted_out["fdr_rank_boot"].to_numpy(dtype=float))
    assert np.all(diffs >= -1e-12)


def test_null_signal_case(
    tmp_path: Path,
    synthetic_null_dataset: dict[str, object],
) -> None:
    """A null-like real ranking should not produce a large discovery fraction."""
    real_df = _build_real_rankings(
        synthetic_null_dataset["catalog"],
        synthetic_null_dataset["null_means"],
    )
    out_df, _, _ = _run_bootstrap(
        tmp_path / "null_case",
        real_df=real_df,
        null_path=synthetic_null_dataset["null_path"],
        n_bootstrap=100,
        seed=11,
    )

    discovery_fraction = float((out_df["fdr_rank_boot"] < 0.05).mean())
    assert discovery_fraction <= 0.10


def test_strong_signal_case(
    tmp_path: Path,
    synthetic_null_dataset: dict[str, object],
) -> None:
    """Strongly boosted variants should sit at the empirical resolution floor."""
    signal_indices = list(range(20))
    signal_positions = set(
        synthetic_null_dataset["catalog"].iloc[signal_indices]["position"].tolist()
    )
    real_df = _build_real_rankings(
        synthetic_null_dataset["catalog"],
        synthetic_null_dataset["null_means"],
        boost_indices=signal_indices,
    )
    out_df, _, _ = _run_bootstrap(
        tmp_path / "strong_signal",
        real_df=real_df,
        null_path=synthetic_null_dataset["null_path"],
        n_bootstrap=500,
    )

    signal_rows = out_df[out_df["position"].isin(signal_positions)].copy()
    assert len(signal_rows) == 20
    assert signal_rows["at_resolution_floor"].all()
    assert (signal_rows["fdr_rank_boot"] < 0.05).all()


def test_gene_wilcoxon_underpowered_flag(
    tmp_path: Path,
    synthetic_null_dataset: dict[str, object],
) -> None:
    """Genes below the variant threshold should be flagged and skipped."""
    real_df = _build_real_rankings(
        synthetic_null_dataset["catalog"],
        synthetic_null_dataset["null_means"],
        boost_indices=list(range(20)),
    )
    _, gene_df, _ = _run_bootstrap(
        tmp_path / "gene_stats",
        real_df=real_df,
        null_path=synthetic_null_dataset["null_path"],
        n_bootstrap=100,
    )

    underpowered = gene_df.loc[gene_df["gene_name"] == "UNDERPOWERED"].iloc[0]
    powered = gene_df.loc[gene_df["gene_name"] == "POWERED"].iloc[0]
    assert bool(underpowered["underpowered"]) is True
    assert pd.isna(underpowered["wilcoxon_p"])
    assert bool(powered["underpowered"]) is False
    assert not pd.isna(powered["wilcoxon_p"])


def test_exclude_sex_chroms(
    tmp_path: Path,
    synthetic_null_dataset: dict[str, object],
) -> None:
    """Sex chromosomes should be removed from both real and null inputs."""
    real_df = _build_real_rankings(
        synthetic_null_dataset["catalog"],
        synthetic_null_dataset["null_means"],
        boost_indices=list(range(20)),
    )
    out_df, _, summary = _run_bootstrap(
        tmp_path / "exclude_sex",
        real_df=real_df,
        null_path=synthetic_null_dataset["null_path"],
        n_bootstrap=100,
        exclude_sex_chroms=True,
    )

    assert not out_df["chromosome"].isin(["X", "Y"]).any()
    assert summary["excluded_sex_chroms"] is True
    assert summary["n_real_variants_removed_sex_chroms"] > 0
    assert summary["n_null_rows_removed_sex_chroms"] > 0


# ---------------------------------------------------------------------------
# Sex-chromosome filtering index alignment
# ---------------------------------------------------------------------------


def _interleave_sex_rows(real_df: pd.DataFrame) -> pd.DataFrame:
    """Reorder rows so sex-chromosome rows come first and between autosomes.

    The catalogue normally places X/Y at the end, where filtering happens to
    leave a contiguous index. This layout guarantees non-contiguous labels
    after filtering, which is the case the index reset must handle.
    """
    is_sex = real_df["chromosome"].isin(["X", "Y"]).to_numpy()
    sex_rows = np.flatnonzero(is_sex)
    auto_rows = np.flatnonzero(~is_sex)
    half = len(auto_rows) // 2
    order = np.concatenate([sex_rows[:20], auto_rows[:half], sex_rows[20:], auto_rows[half:]])
    return real_df.iloc[order].reset_index(drop=True)


def _small_interleaved_df() -> pd.DataFrame:
    """Return a small frame with X/Y rows before and between two autosomal genes."""
    return pd.DataFrame(
        {
            "chromosome": ["X", "1", "1", "Y", "2", "X", "2", "2"],
            "position": [10, 20, 30, 40, 50, 60, 70, 80],
            "gene_name": ["GX", "GA", "GA", "GY", "GB", "GX", "GB", "GB"],
            "mean_attribution": [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            "delta_rank": [0.0, 5.0, -1.0, 0.0, 2.0, 0.0, 9.0, -3.0],
        }
    )


def test_sex_chrom_filter_resets_to_contiguous_index() -> None:
    """Filtered rows keep their order but get a contiguous positional index."""
    build = calibration.get_genome_build("GRCh37")
    real_df = _small_interleaved_df()

    filtered, removed = calibration._maybe_filter_real_df(real_df, True, build)

    assert removed == 3
    assert filtered.index.tolist() == list(range(5))
    assert filtered["position"].tolist() == [20, 30, 50, 70, 80]
    assert not filtered["chromosome"].isin(["X", "Y"]).any()


def test_sex_chrom_filter_is_identity_when_disabled() -> None:
    """exclude_sex_chroms=False returns the input frame untouched."""
    build = calibration.get_genome_build("GRCh37")
    real_df = _small_interleaved_df()

    filtered, removed = calibration._maybe_filter_real_df(real_df, False, build)

    assert filtered is real_df
    assert removed == 0


def test_gene_statistics_use_retained_rows_after_sex_chrom_filter() -> None:
    """Gene stats must read the ranks and delta_rank of the retained rows only.

    Before the index reset, the retained labels (1, 2, 4, 6, 7) were used as
    positions into five-element arrays, so gene GB raised IndexError.
    """
    build = calibration.get_genome_build("GRCh37")
    filtered, _ = calibration._maybe_filter_real_df(_small_interleaved_df(), True, build)
    rank_real = np.array([11.0, 12.0, 13.0, 14.0, 15.0])
    rank_null_full = np.array([21.0, 22.0, 23.0, 24.0, 25.0])

    def gene_stats(aggregation: str) -> pd.DataFrame:
        return calibration._compute_gene_statistics(
            real_df=filtered,
            rank_real=rank_real,
            rank_null_full=rank_null_full,
            real_to_null_index=np.arange(5, dtype=np.int64),
            min_variants_per_gene=1,
            delta_rank_aggregation=aggregation,
        ).set_index("gene_name")

    max_df = gene_stats("max")
    assert sorted(max_df.index) == ["GA", "GB"]
    assert max_df.loc["GA", "n_variants_real"] == 2
    assert max_df.loc["GB", "n_variants_real"] == 3
    assert max_df.loc["GA", "median_rank_real"] == pytest.approx(11.5)
    assert max_df.loc["GB", "median_rank_real"] == pytest.approx(14.0)
    assert max_df.loc["GA", "median_rank_null"] == pytest.approx(21.5)
    assert max_df.loc["GB", "median_rank_null"] == pytest.approx(24.0)
    assert max_df.loc["GA", "gene_delta_rank"] == pytest.approx(5.0)
    assert max_df.loc["GB", "gene_delta_rank"] == pytest.approx(9.0)

    mean_df = gene_stats("mean")
    assert mean_df.loc["GA", "gene_delta_rank"] == pytest.approx(2.0)
    assert mean_df.loc["GB", "gene_delta_rank"] == pytest.approx(8.0 / 3.0)


def test_exclude_sex_chroms_interleaved_matches_prefiltered_input(
    tmp_path: Path,
    synthetic_null_dataset: dict[str, object],
) -> None:
    """Interleaved sex rows filtered in-script equal a manually pre-filtered input."""
    interleaved = _interleave_sex_rows(
        _build_real_rankings(
            synthetic_null_dataset["catalog"],
            synthetic_null_dataset["null_means"],
            boost_indices=list(range(20)),
        )
    )
    is_autosomal = ~interleaved["chromosome"].isin(["X", "Y"])
    prefiltered = interleaved.loc[is_autosomal].reset_index(drop=True)

    out_df, gene_df, summary = _run_bootstrap(
        tmp_path / "interleaved",
        real_df=interleaved,
        null_path=synthetic_null_dataset["null_path"],
        n_bootstrap=50,
        exclude_sex_chroms=True,
    )
    ref_out, ref_gene, ref_summary = _run_bootstrap(
        tmp_path / "prefiltered",
        real_df=prefiltered,
        null_path=synthetic_null_dataset["null_path"],
        n_bootstrap=50,
        exclude_sex_chroms=True,
    )

    assert out_df["position"].tolist() == prefiltered["position"].tolist()
    assert not out_df["chromosome"].astype(str).isin(["X", "Y"]).any()
    # Variant-level calibration must be unchanged by the index reset.
    np.testing.assert_array_equal(out_df["delta_rank"], ref_out["delta_rank"])
    np.testing.assert_array_equal(out_df["rank_real"], ref_out["rank_real"])
    pd.testing.assert_frame_equal(out_df, ref_out)
    pd.testing.assert_frame_equal(gene_df, ref_gene)
    assert summary["n_real_variants_removed_sex_chroms"] == 60
    assert ref_summary["n_real_variants_removed_sex_chroms"] == 0
    summary.pop("n_real_variants_removed_sex_chroms")
    ref_summary.pop("n_real_variants_removed_sex_chroms")
    assert summary == ref_summary


def test_include_sex_chroms_interleaved_keeps_all_rows(
    tmp_path: Path,
    synthetic_null_dataset: dict[str, object],
) -> None:
    """Without exclusion, interleaved input keeps every row in its original order."""
    interleaved = _interleave_sex_rows(
        _build_real_rankings(
            synthetic_null_dataset["catalog"],
            synthetic_null_dataset["null_means"],
        )
    )

    out_df, gene_df, summary = _run_bootstrap(
        tmp_path / "include_sex",
        real_df=interleaved,
        null_path=synthetic_null_dataset["null_path"],
        n_bootstrap=50,
        exclude_sex_chroms=False,
    )

    assert out_df["position"].tolist() == interleaved["position"].tolist()
    assert summary["excluded_sex_chroms"] is False
    assert summary["n_real_variants_removed_sex_chroms"] == 0
    assert gene_df["n_variants_real"].sum() == len(interleaved)


def test_missing_variant_in_null(
    tmp_path: Path,
    synthetic_null_dataset: dict[str, object],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Variants absent from the null universe should be assigned worst null ranks."""
    real_df = _build_real_rankings(
        synthetic_null_dataset["catalog"],
        synthetic_null_dataset["null_means"],
        add_missing_variant=True,
    )

    with caplog.at_level(logging.WARNING):
        out_df, _, summary = _run_bootstrap(
            tmp_path / "missing_null",
            real_df=real_df,
            null_path=synthetic_null_dataset["null_path"],
            n_bootstrap=100,
        )

    missing_row = out_df.loc[out_df["gene_name"] == "MISSING_NULL"].iloc[0]
    assert summary["n_real_variants_missing_from_null"] == 1
    assert missing_row["median_rank_null_boot"] > len(synthetic_null_dataset["catalog"])
    assert missing_row["fdr_rank_boot"] >= 0.05
    assert "absent from the null universe" in caplog.text


def test_delta_rank_sign(
    tmp_path: Path,
    synthetic_null_dataset: dict[str, object],
) -> None:
    """Signal variants should have positive delta_rank while null-like ones stay near zero."""
    real_df = _build_real_rankings(
        synthetic_null_dataset["catalog"],
        synthetic_null_dataset["null_means"],
        boost_indices=[0],
    )
    out_df, _, _ = _run_bootstrap(
        tmp_path / "delta_rank",
        real_df=real_df,
        null_path=synthetic_null_dataset["null_path"],
        n_bootstrap=100,
    )

    signal_row = out_df.loc[out_df["position"] == 10].iloc[0]
    background_row = out_df.loc[out_df["position"] == 1010].iloc[0]
    assert signal_row["delta_rank"] > 0
    assert abs(background_row["delta_rank"]) < 10


def test_chromosome_normalisation(
    tmp_path: Path,
    synthetic_null_dataset: dict[str, object],
) -> None:
    """Real 'chrN' labels should still match null 'N' labels."""
    real_df = _build_real_rankings(
        synthetic_null_dataset["catalog"],
        synthetic_null_dataset["null_means"],
        boost_indices=[0],
        prefix_chr=True,
    )
    out_df, _, summary = _run_bootstrap(
        tmp_path / "chrom_norm",
        real_df=real_df,
        null_path=synthetic_null_dataset["null_path"],
        n_bootstrap=100,
    )

    assert summary["n_real_variants_missing_from_null"] == 0
    assert "1" in set(out_df["chromosome"].astype(str))
    assert "chr1" not in set(out_df["chromosome"].astype(str))


def test_worst_rank_for_absent_bootstrap_variants() -> None:
    """Variants absent from a bootstrap draw should receive a constant worst rank."""
    ranks, present_count = calibration._compute_rank_vector(
        flat_key_indices=np.array([0], dtype=np.int32),
        flat_scores=np.array([1.0], dtype=np.float64),
        sample_starts=np.array([0, 1], dtype=np.int64),
        sample_lengths=np.array([1, 0], dtype=np.int32),
        selected_sample_indices=np.array([0], dtype=np.int64),
        n_unique_variants=3,
    )

    assert present_count == 1
    assert ranks.tolist() == [1.0, 4.0, 4.0]


def test_hodges_lehmann_fallback_uses_median_difference() -> None:
    """Large genes should use the linear-memory median-difference fallback."""
    real_ranks = np.arange(1200, dtype=np.float64)
    null_ranks = np.arange(1200, dtype=np.float64) + 5.0

    shift = calibration._estimate_hodges_lehmann_shift(real_ranks, null_ranks)
    assert shift == pytest.approx(np.median(null_ranks) - np.median(real_ranks))


def test_genome_build_defaults_from_metadata(
    tmp_path: Path,
    synthetic_null_dataset: dict[str, object],
) -> None:
    """Genome build should default from nearby analysis metadata when present."""
    real_df = _build_real_rankings(
        synthetic_null_dataset["catalog"],
        synthetic_null_dataset["null_means"],
        prefix_chr=True,
    )
    out_df, _, summary = _run_bootstrap(
        tmp_path / "genome_build_metadata",
        real_df=real_df,
        null_path=synthetic_null_dataset["null_path"],
        n_bootstrap=100,
        metadata_genome_build="hg38",
    )

    assert summary["genome_build"] == "GRCh38"
    assert out_df["chromosome"].iloc[0] == "1"


# ---------------------------------------------------------------------------
# gene_delta_rank tests
# ---------------------------------------------------------------------------


def _make_minimal_real_df(delta_rank_values: list[float], gene_names: list[str]) -> pd.DataFrame:
    """Construct the minimal real_df required by _compute_gene_statistics."""
    n = len(delta_rank_values)
    return pd.DataFrame(
        {
            "chromosome": ["1"] * n,
            "position": list(range(1, n + 1)),
            "gene_name": gene_names,
            "mean_attribution": [float(i) for i in range(n, 0, -1)],
            "delta_rank": delta_rank_values,
        }
    )


def _call_compute_gene_statistics(
    real_df: pd.DataFrame,
    aggregation: str,
    min_variants: int = 1,
) -> pd.DataFrame:
    """Call _compute_gene_statistics with dummy rank arrays."""
    n = len(real_df)
    rank_real = np.arange(1, n + 1, dtype=float)
    rank_null_full = np.arange(1, n + 1, dtype=float)
    real_to_null_index = np.arange(n, dtype=np.int64)
    return calibration._compute_gene_statistics(
        real_df=real_df,
        rank_real=rank_real,
        rank_null_full=rank_null_full,
        real_to_null_index=real_to_null_index,
        min_variants_per_gene=min_variants,
        delta_rank_aggregation=aggregation,
    )


def test_gene_delta_rank_max_matches_manual_aggregation() -> None:
    """gene_delta_rank with aggregation='max' must equal manual per-gene max."""
    delta_values = [5.0, 3.0, 4.0, 2.0, 1.0, 6.0, 3.0, 4.0, 2.0]
    gene_names = ["A", "A", "B", "B", "B", "C", "C", "C", "C"]
    real_df = _make_minimal_real_df(delta_values, gene_names)
    gene_df = _call_compute_gene_statistics(real_df, aggregation="max")

    gene_df = gene_df.set_index("gene_name")
    assert math.isclose(gene_df.loc["A", "gene_delta_rank"], 5.0)
    assert math.isclose(gene_df.loc["B", "gene_delta_rank"], 4.0)
    assert math.isclose(gene_df.loc["C", "gene_delta_rank"], 6.0)


def test_gene_delta_rank_mean_matches_manual_aggregation() -> None:
    """gene_delta_rank with aggregation='mean' must equal manual per-gene mean."""
    delta_values = [5.0, 3.0, 4.0, 2.0, 1.0, 6.0, 3.0, 4.0, 2.0]
    gene_names = ["A", "A", "B", "B", "B", "C", "C", "C", "C"]
    real_df = _make_minimal_real_df(delta_values, gene_names)
    gene_df = _call_compute_gene_statistics(real_df, aggregation="mean")

    gene_df = gene_df.set_index("gene_name")
    assert math.isclose(gene_df.loc["A", "gene_delta_rank"], 4.0)
    assert math.isclose(gene_df.loc["B", "gene_delta_rank"], 7.0 / 3.0)
    assert math.isclose(gene_df.loc["C", "gene_delta_rank"], 15.0 / 4.0)


def test_gene_delta_rank_aggregation_column_is_constant() -> None:
    """gene_delta_rank_aggregation must be the same constant string on every row."""
    delta_values = [1.0, 2.0, 3.0, 4.0]
    gene_names = ["A", "A", "B", "B"]
    real_df = _make_minimal_real_df(delta_values, gene_names)

    for agg in ("max", "mean"):
        gene_df = _call_compute_gene_statistics(real_df, aggregation=agg)
        assert (gene_df["gene_delta_rank_aggregation"] == agg).all(), (
            f"Expected all rows to have aggregation='{agg}'"
        )


def test_gene_delta_rank_respects_underpowered_variants(
    tmp_path: Path,
    synthetic_null_dataset: dict[str, object],
) -> None:
    """Underpowered genes must still receive a gene_delta_rank value."""
    real_df = _build_real_rankings(
        synthetic_null_dataset["catalog"],
        synthetic_null_dataset["null_means"],
        boost_indices=list(range(20)),
    )
    _, gene_df, _ = _run_bootstrap(
        tmp_path / "gene_delta_rank_underpowered",
        real_df=real_df,
        null_path=synthetic_null_dataset["null_path"],
        n_bootstrap=100,
    )

    underpowered_rows = gene_df[gene_df["underpowered"]]
    assert len(underpowered_rows) > 0, "Expected at least one underpowered gene in fixture"
    assert not underpowered_rows["gene_delta_rank"].isna().any(), (
        "Underpowered genes must have a gene_delta_rank value, not NaN"
    )


def test_cli_default_is_max() -> None:
    """parse_args with no --gene-delta-rank-aggregation must default to 'max'."""
    args = calibration.parse_args([
        "--real-rankings", "r.csv",
        "--null-attributions", "n.npz",
        "--output", "o.csv",
    ])
    assert args.gene_delta_rank_aggregation == "max"


def test_cli_rejects_invalid_aggregation() -> None:
    """parse_args must reject --gene-delta-rank-aggregation sum with SystemExit."""
    with pytest.raises(SystemExit):
        calibration.parse_args([
            "--real-rankings", "r.csv",
            "--null-attributions", "n.npz",
            "--output", "o.csv",
            "--gene-delta-rank-aggregation", "sum",
        ])
