"""Tests for ablation comparison ranking semantics."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml
import pytest

import scripts.compare_ablation_rankings as ablation


def write_variant_rankings(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a small ranking CSV for testing."""
    headers = [
        "variant_id",
        "gene_name",
        "gene_id",
        "chromosome",
        "position",
        "empirical_p_variant",
        "fdr_variant",
        "z_attribution",
        "delta_rank",
        "corrected_rank",
        "rank",
    ]
    lines = [",".join(headers)]
    for row in rows:
        lines.append(",".join(str(row.get(header, "")) for header in headers))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_load_rankings_sorts_empirical_p_ascending(tmp_path: Path) -> None:
    """Lower empirical p-values should receive better ranks."""
    csv_path = tmp_path / "L0_sieve_variant_rankings.csv"
    write_variant_rankings(
        csv_path,
        [
            {
                "variant_id": "1:100_A",
                "gene_name": "GENE1",
                "gene_id": 1,
                "chromosome": "1",
                "position": 100,
                "empirical_p_variant": 0.20,
                "fdr_variant": 0.30,
                "z_attribution": 6.0,
            },
            {
                "variant_id": "1:200_B",
                "gene_name": "GENE2",
                "gene_id": 2,
                "chromosome": "1",
                "position": 200,
                "empirical_p_variant": 0.01,
                "fdr_variant": 0.02,
                "z_attribution": 1.0,
            },
            {
                "variant_id": "1:300_C",
                "gene_name": "GENE3",
                "gene_id": 3,
                "chromosome": "1",
                "position": 300,
                "empirical_p_variant": 0.10,
                "fdr_variant": 0.15,
                "z_attribution": 4.0,
            },
        ],
    )

    rows, resolved_col, was_explicit = ablation.load_rankings(
        csv_path,
        score_column="empirical_p_variant",
    )

    assert resolved_col == "empirical_p_variant"
    assert was_explicit is True
    assert [row["variant_id"] for row in rows] == [
        "1:200_B",
        "1:300_C",
        "1:100_A",
    ]
    assert [row["rank"] for row in rows] == [1, 2, 3]


def test_main_defaults_to_z_attribution(tmp_path: Path, monkeypatch) -> None:
    """The CLI should default to z_attribution for cross-level ranking."""
    ranking_dir = tmp_path / "rankings"
    ranking_dir.mkdir()

    write_variant_rankings(
        ranking_dir / "L0_sieve_variant_rankings.csv",
        [
            {
                "variant_id": "1:100_A",
                "gene_name": "GENE1",
                "gene_id": 1,
                "chromosome": "1",
                "position": 100,
                "empirical_p_variant": 0.30,
                "fdr_variant": 0.40,
                "z_attribution": 9.0,
            },
            {
                "variant_id": "1:200_B",
                "gene_name": "GENE2",
                "gene_id": 2,
                "chromosome": "1",
                "position": 200,
                "empirical_p_variant": 0.01,
                "fdr_variant": 0.02,
                "z_attribution": 1.0,
            },
        ],
    )
    write_variant_rankings(
        ranking_dir / "L1_sieve_variant_rankings.csv",
        [
            {
                "variant_id": "1:100_A",
                "gene_name": "GENE1",
                "gene_id": 1,
                "chromosome": "1",
                "position": 100,
                "empirical_p_variant": 0.02,
                "fdr_variant": 0.03,
                "z_attribution": 2.0,
            },
            {
                "variant_id": "1:300_C",
                "gene_name": "GENE3",
                "gene_id": 3,
                "chromosome": "1",
                "position": 300,
                "empirical_p_variant": 0.50,
                "fdr_variant": 0.60,
                "z_attribution": 10.0,
            },
        ],
    )

    out_comparison = tmp_path / "summary.yaml"
    out_jaccard = tmp_path / "jaccard.tsv"
    out_level_specific = tmp_path / "level_specific.tsv"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_ablation_rankings.py",
            "--ranking-dir",
            str(ranking_dir),
            "--top-k",
            "1",
            "--out-comparison",
            str(out_comparison),
            "--out-jaccard",
            str(out_jaccard),
            "--out-level-specific",
            str(out_level_specific),
        ],
    )

    exit_code = ablation.main()

    assert exit_code == 0
    summary = yaml.safe_load(out_comparison.read_text(encoding="utf-8"))
    assert summary["score_column"] == "z_attribution"
    assert summary["score_sort_order"] == "descending"


def test_load_rankings_pushes_invalid_empirical_p_to_the_bottom(tmp_path: Path) -> None:
    """Malformed p-values must not be promoted to the best ranks."""
    csv_path = tmp_path / "L0_sieve_variant_rankings.csv"
    csv_path.write_text(
        "\n".join(
            [
                "variant_id,gene_name,chromosome,position,empirical_p_variant,fdr_variant,z_attribution",
                "1:100_A,GENE1,1,100,0.02,0.05,5.0",
                "1:200_B,GENE2,1,200,NA,0.10,4.0",
                "1:300_C,GENE3,1,300,0.10,0.15,3.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows, _, _ = ablation.load_rankings(csv_path, score_column="empirical_p_variant")

    assert [row["variant_id"] for row in rows] == [
        "1:100_A",
        "1:300_C",
        "1:200_B",
    ]


def test_main_fails_when_z_attribution_column_is_missing(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """The default ablation comparison should fail when z_attribution is absent."""
    ranking_dir = tmp_path / "rankings"
    ranking_dir.mkdir()

    # CSVs that have empirical_p_variant but NOT z_attribution should fail
    # with the new default.
    raw_csv = "\n".join(
        [
            "variant_id,gene_name,chromosome,position,empirical_p_variant",
            "1:100_A,GENE1,1,100,0.05",
            "1:200_B,GENE2,1,200,0.10",
        ]
    )
    (ranking_dir / "L0_sieve_variant_rankings.csv").write_text(raw_csv + "\n", encoding="utf-8")
    (ranking_dir / "L1_sieve_variant_rankings.csv").write_text(raw_csv + "\n", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_ablation_rankings.py",
            "--ranking-dir",
            str(ranking_dir),
            "--out-comparison",
            str(tmp_path / "summary.yaml"),
            "--out-jaccard",
            str(tmp_path / "jaccard.tsv"),
            "--out-level-specific",
            str(tmp_path / "level_specific.tsv"),
        ],
    )

    exit_code = ablation.main()
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "z_attribution" in captured.err


def test_delta_rank_is_descending() -> None:
    """delta_rank must be treated as a descending score."""
    assert ablation._score_column_is_ascending("delta_rank") is False


@pytest.mark.parametrize(
    ("column_name", "expected"),
    [
        ("corrected_rank", True),
        ("rank", True),
        ("empirical_p_rank", True),
        ("p_rank_boot", True),
        ("rank_real", True),
        ("median_rank_null_boot", True),
    ],
)
def test_existing_rank_columns_still_ascending(
    column_name: str,
    expected: bool,
) -> None:
    """Legacy rank-like columns must keep the old ascending behavior."""
    assert ablation._score_column_is_ascending(column_name) is expected


def test_delta_rank_sort_produces_correct_top_k(tmp_path: Path) -> None:
    """The highest delta_rank should be ranked first."""
    csv_path = tmp_path / "L0_sieve_variant_rankings.csv"
    write_variant_rankings(
        csv_path,
        [
            {
                "variant_id": "1:100_1",
                "gene_name": "GENE1",
                "gene_id": 1,
                "chromosome": "1",
                "position": 100,
                "delta_rank": 100,
            },
            {
                "variant_id": "1:200_2",
                "gene_name": "GENE2",
                "gene_id": 2,
                "chromosome": "1",
                "position": 200,
                "delta_rank": 50,
            },
            {
                "variant_id": "1:300_3",
                "gene_name": "GENE3",
                "gene_id": 3,
                "chromosome": "1",
                "position": 300,
                "delta_rank": 0,
            },
            {
                "variant_id": "1:400_4",
                "gene_name": "GENE4",
                "gene_id": 4,
                "chromosome": "1",
                "position": 400,
                "delta_rank": -50,
            },
            {
                "variant_id": "1:500_5",
                "gene_name": "GENE5",
                "gene_id": 5,
                "chromosome": "1",
                "position": 500,
                "delta_rank": -100,
            },
        ],
    )

    rows, resolved_col, was_explicit = ablation.load_rankings(
        csv_path,
        score_column="delta_rank",
    )

    assert resolved_col == "delta_rank"
    assert was_explicit is True
    assert rows[0]["variant_id"] == "1:100_1"
    assert rows[-1]["variant_id"] == "1:500_5"


def test_z_attribution_sort_unchanged(tmp_path: Path) -> None:
    """Existing attribution-like columns must remain descending."""
    csv_path = tmp_path / "L0_sieve_variant_rankings.csv"
    write_variant_rankings(
        csv_path,
        [
            {
                "variant_id": "1:100_1",
                "gene_name": "GENE1",
                "gene_id": 1,
                "chromosome": "1",
                "position": 100,
                "z_attribution": 1.0,
            },
            {
                "variant_id": "1:200_2",
                "gene_name": "GENE2",
                "gene_id": 2,
                "chromosome": "1",
                "position": 200,
                "z_attribution": 10.0,
            },
        ],
    )

    rows, _, _ = ablation.load_rankings(csv_path, score_column="z_attribution")
    assert rows[0]["variant_id"] == "1:200_2"


def test_bootstrap_calibrated_file_integration(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The script should work end-to-end with both z_attribution and delta_rank."""
    ranking_dir = tmp_path / "rankings"
    ranking_dir.mkdir()

    write_variant_rankings(
        ranking_dir / "L0_sieve_variant_rankings.csv",
        [
            {
                "variant_id": "1:100_1",
                "gene_name": "GENE1",
                "gene_id": 1,
                "chromosome": "1",
                "position": 100,
                "z_attribution": 10.0,
                "delta_rank": 100.0,
            },
            {
                "variant_id": "1:200_2",
                "gene_name": "GENE2",
                "gene_id": 2,
                "chromosome": "1",
                "position": 200,
                "z_attribution": 9.0,
                "delta_rank": 1.0,
            },
        ],
    )
    write_variant_rankings(
        ranking_dir / "L1_sieve_variant_rankings.csv",
        [
            {
                "variant_id": "1:100_1",
                "gene_name": "GENE1",
                "gene_id": 1,
                "chromosome": "1",
                "position": 100,
                "z_attribution": 12.0,
                "delta_rank": 0.0,
            },
            {
                "variant_id": "1:300_3",
                "gene_name": "GENE3",
                "gene_id": 3,
                "chromosome": "1",
                "position": 300,
                "z_attribution": 11.0,
                "delta_rank": 100.0,
            },
        ],
    )

    z_summary = tmp_path / "z_summary.yaml"
    z_jaccard = tmp_path / "z_jaccard.tsv"
    z_specific = tmp_path / "z_specific.tsv"
    delta_summary = tmp_path / "delta_summary.yaml"
    delta_jaccard = tmp_path / "delta_jaccard.tsv"
    delta_specific = tmp_path / "delta_specific.tsv"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_ablation_rankings.py",
            "--ranking-dir",
            str(ranking_dir),
            "--score-column",
            "z_attribution",
            "--top-k",
            "1",
            "--out-comparison",
            str(z_summary),
            "--out-jaccard",
            str(z_jaccard),
            "--out-level-specific",
            str(z_specific),
        ],
    )
    assert ablation.main() == 0

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_ablation_rankings.py",
            "--ranking-dir",
            str(ranking_dir),
            "--score-column",
            "delta_rank",
            "--top-k",
            "1",
            "--out-comparison",
            str(delta_summary),
            "--out-jaccard",
            str(delta_jaccard),
            "--out-level-specific",
            str(delta_specific),
        ],
    )
    assert ablation.main() == 0

    z_payload = yaml.safe_load(z_summary.read_text(encoding="utf-8"))
    delta_payload = yaml.safe_load(delta_summary.read_text(encoding="utf-8"))

    assert z_payload["score_column"] == "z_attribution"
    assert delta_payload["score_column"] == "delta_rank"
    assert z_payload["score_sort_order"] == "descending"
    assert delta_payload["score_sort_order"] == "descending"
    assert z_jaccard.read_text(encoding="utf-8") != delta_jaccard.read_text(
        encoding="utf-8"
    )


def test_missing_delta_rank_column_errors_cleanly(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """Requesting delta_rank should fail clearly when the column is absent."""
    ranking_dir = tmp_path / "rankings"
    ranking_dir.mkdir()
    raw_csv = "\n".join(
        [
            "variant_id,gene_name,gene_id,chromosome,position,z_attribution",
            "1:100_1,GENE1,1,1,100,0.5",
            "1:200_2,GENE2,2,1,200,1.5",
        ]
    )
    (ranking_dir / "L0_sieve_variant_rankings.csv").write_text(
        raw_csv + "\n",
        encoding="utf-8",
    )
    (ranking_dir / "L1_sieve_variant_rankings.csv").write_text(
        raw_csv + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_ablation_rankings.py",
            "--ranking-dir",
            str(ranking_dir),
            "--score-column",
            "delta_rank",
            "--out-comparison",
            str(tmp_path / "summary.yaml"),
            "--out-jaccard",
            str(tmp_path / "jaccard.tsv"),
            "--out-level-specific",
            str(tmp_path / "specific.tsv"),
        ],
    )

    exit_code = ablation.main()
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "delta_rank" in captured.err


def test_explicit_rankings_level_mode_keeps_legacy_outputs_without_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Explicit LEVEL:PATH mode should not require positional metadata."""
    l0_path = tmp_path / "custom_l0.csv"
    l1_path = tmp_path / "custom_l1.csv"
    write_variant_rankings(
        l0_path,
        [
            {
                "variant_id": "1:100_A",
                "gene_name": "GENE1",
                "gene_id": 1,
                "chromosome": "1",
                "position": 100,
                "z_attribution": 10.0,
            },
            {
                "variant_id": "1:200_B",
                "gene_name": "GENE2",
                "gene_id": 2,
                "chromosome": "1",
                "position": 200,
                "z_attribution": 1.0,
            },
        ],
    )
    write_variant_rankings(
        l1_path,
        [
            {
                "variant_id": "1:100_A",
                "gene_name": "GENE1",
                "gene_id": 1,
                "chromosome": "1",
                "position": 100,
                "z_attribution": 1.0,
            },
            {
                "variant_id": "1:300_C",
                "gene_name": "GENE3",
                "gene_id": 3,
                "chromosome": "1",
                "position": 300,
                "z_attribution": 10.0,
            },
        ],
    )

    out_comparison = tmp_path / "summary.yaml"
    out_jaccard = tmp_path / "jaccard.tsv"
    out_level_specific = tmp_path / "level_specific.tsv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_ablation_rankings.py",
            "--rankings",
            f"L0:{l0_path}",
            f"L1:{l1_path}",
            "--top-k",
            "1",
            "--high-rank-threshold",
            "1",
            "--low-rank-threshold",
            "1",
            "--out-comparison",
            str(out_comparison),
            "--out-jaccard",
            str(out_jaccard),
            "--out-level-specific",
            str(out_level_specific),
        ],
    )

    assert ablation.main() == 0

    jaccard_header = out_jaccard.read_text(encoding="utf-8").splitlines()[0]
    assert jaccard_header.split("\t") == [
        "top_k",
        "level_a",
        "level_b",
        "jaccard",
        "overlap",
        "size_a",
        "size_b",
        "union",
    ]
    specific_header = out_level_specific.read_text(encoding="utf-8").splitlines()[0]
    assert specific_header.split("\t") == [
        "variant_id",
        "gene",
        "chrom",
        "pos",
        "specific_to_level",
        "rank_at_specific_level",
        "rank_at_L0",
        "rank_at_L1",
        "score_at_specific_level",
    ]
    summary = yaml.safe_load(out_comparison.read_text(encoding="utf-8"))
    assert summary["levels_analysed"] == ["L0", "L1"]
    assert set(summary["jaccard_matrices"]["top_1"]) == {"L0_vs_L1"}
    assert summary["level_specific_variant_counts"] == {"L0": 1, "L1": 1}


def test_level_mode_one_file_warning_is_preserved(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """A single discovered level should still warn rather than fail."""
    ranking_dir = tmp_path / "rankings"
    ranking_dir.mkdir()
    write_variant_rankings(
        ranking_dir / "L0_sieve_variant_rankings.csv",
        [
            {
                "variant_id": "1:100_A",
                "gene_name": "GENE1",
                "gene_id": 1,
                "chromosome": "1",
                "position": 100,
                "z_attribution": 10.0,
            }
        ],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_ablation_rankings.py",
            "--ranking-dir",
            str(ranking_dir),
            "--out-comparison",
            str(tmp_path / "summary.yaml"),
            "--out-jaccard",
            str(tmp_path / "jaccard.tsv"),
            "--out-level-specific",
            str(tmp_path / "specific.tsv"),
        ],
    )

    assert ablation.main() == 0

    assert "Need at least 2 levels" in capsys.readouterr().err
