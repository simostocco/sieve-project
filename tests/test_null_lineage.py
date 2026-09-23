#!/usr/bin/env python3
"""
Tests for Phase 12C3A: null-dataset lineage and provenance.

Covers the pure src/data/null_lineage.py helpers, the strict CLI added to
scripts/create_null_baseline.py, and backward compatibility of the
historical (non-strict) generator paths.
"""

import copy
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts import create_null_baseline  # noqa: E402
from src.data import null_lineage  # noqa: E402
from src.data.vcf_parser import SampleVariants, VariantRecord  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_variant(gene="GENE1", pos=100, sift=0.05, polyphen=0.9):
    return VariantRecord(
        chrom="1",
        pos=pos,
        ref="A",
        alt="T",
        gene=gene,
        consequence="missense_variant",
        genotype=1,
        annotations={"sift": sift, "polyphen": polyphen},
    )


def _make_samples(n=10):
    return [
        SampleVariants(
            f"s{i}",
            label=i % 2,
            variants=[_make_variant(pos=100 + i)],
            sex="M" if i % 2 == 0 else "F",
        )
        for i in range(n)
    ]


def _save_artifact(path, samples, metadata=None):
    torch.save({"samples": samples, "metadata": metadata or {"genome_build": "GRCh37"}}, path)


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _create_valid_pair(tmp_path, n=10, seed=42):
    """Create a valid strict null pair and return (source_path, null_path, sidecar_path, samples, report)."""
    samples = _make_samples(n)
    source_path = tmp_path / "source.pt"
    _save_artifact(source_path, samples)
    output_path = tmp_path / "null.pt"
    report = create_null_baseline.create_strict_single_permutation(
        str(source_path), str(output_path), seed=seed, reuse=False, argv=["create_null_baseline.py"]
    )
    sidecar_path = null_lineage.sidecar_path_for(output_path)
    return source_path, output_path, sidecar_path, samples, report


# ---------------------------------------------------------------------------
# Label validation
# ---------------------------------------------------------------------------


class TestLabelValidation:
    def test_accepts_plain_zero_and_one(self):
        assert null_lineage.validate_label(0, index=0) == 0
        assert null_lineage.validate_label(1, index=0) == 1

    def test_rejects_bool(self):
        with pytest.raises(ValueError, match="bool"):
            null_lineage.validate_label(True, index=0)

    def test_rejects_out_of_range(self):
        with pytest.raises(ValueError):
            null_lineage.validate_label(2, index=0)

    def test_rejects_non_integer(self):
        with pytest.raises(ValueError):
            null_lineage.validate_label("1", index=0)
        with pytest.raises(ValueError):
            null_lineage.validate_label(1.0, index=0)

    def test_extract_ordered_labels_preserves_order(self):
        samples = _make_samples(6)
        labels = null_lineage.extract_ordered_labels(samples)
        assert labels == [s.label for s in samples]

    def test_labels_sha256_deterministic_and_order_sensitive(self):
        h1 = null_lineage.labels_sha256([0, 1, 0, 1])
        h2 = null_lineage.labels_sha256([0, 1, 0, 1])
        h3 = null_lineage.labels_sha256([1, 0, 0, 1])
        assert h1 == h2
        assert h1 != h3


# ---------------------------------------------------------------------------
# Sample-ID validation (delegated to split_plan; sanity only)
# ---------------------------------------------------------------------------


class TestSampleIdDelegation:
    def test_duplicate_sample_id_rejected(self):
        samples = _make_samples(3)
        samples[1] = SampleVariants("s0", label=0, variants=[], sex=None)
        with pytest.raises(ValueError, match="duplicate"):
            null_lineage.compute_source_lineage_facts(samples)

    def test_blank_sample_id_rejected(self):
        samples = _make_samples(3)
        samples[0] = SampleVariants("  ", label=0, variants=[], sex=None)
        with pytest.raises(ValueError):
            null_lineage.compute_source_lineage_facts(samples)


# ---------------------------------------------------------------------------
# Permutation validation and gather direction
# ---------------------------------------------------------------------------


class TestPermutationValidation:
    def test_true_permutation_accepted(self):
        assert null_lineage.validate_permutation_indices([2, 0, 1], n_samples=3) == [2, 0, 1]

    def test_duplicate_index_rejected(self):
        with pytest.raises(ValueError, match="exactly once"):
            null_lineage.validate_permutation_indices([0, 0, 2], n_samples=3)

    def test_missing_index_rejected(self):
        # length matches n_samples but values don't cover the full range
        with pytest.raises(ValueError):
            null_lineage.validate_permutation_indices([0, 1, 1], n_samples=3)

    def test_out_of_range_index_rejected(self):
        with pytest.raises(ValueError, match="out of range"):
            null_lineage.validate_permutation_indices([0, 1, 3], n_samples=3)

    def test_bool_index_rejected(self):
        with pytest.raises(ValueError, match="bool"):
            null_lineage.validate_permutation_indices([True, False], n_samples=2)

    def test_wrong_length_rejected(self):
        with pytest.raises(ValueError, match="length"):
            null_lineage.validate_permutation_indices([0, 1], n_samples=3)

    def test_numpy_array_accepted(self):
        indices = np.random.default_rng(0).permutation(5)
        validated = null_lineage.validate_permutation_indices(indices, n_samples=5)
        assert sorted(validated) == [0, 1, 2, 3, 4]
        assert all(isinstance(i, int) and not isinstance(i, bool) for i in validated)

    def test_gather_direction_hand_computed(self):
        # null_label[i] = original_label[permutation_indices[i]]  (gather, not scatter)
        original = [0, 1, 1, 0]
        indices = [2, 0, 3, 1]
        expected = [1, 0, 0, 1]
        assert null_lineage.apply_permutation_gather(original, indices) == expected

    def test_permutation_indices_sha256_deterministic_and_order_sensitive(self):
        h1 = null_lineage.permutation_indices_sha256([2, 0, 1])
        h2 = null_lineage.permutation_indices_sha256([2, 0, 1])
        h3 = null_lineage.permutation_indices_sha256([0, 2, 1])
        assert h1 == h2
        assert h1 != h3

    def test_class_counts_preserved_after_permutation(self):
        samples = _make_samples(20)
        source_sha = "a" * 64
        lineage = null_lineage.build_null_lineage(
            samples=samples,
            source_artifact_sha256=source_sha,
            permutation_indices=np.random.default_rng(1).permutation(len(samples)),
        )
        original_n_cases, original_n_controls = null_lineage.class_counts(
            null_lineage.extract_ordered_labels(samples)
        )
        assert lineage["n_cases"] == original_n_cases
        assert lineage["n_controls"] == original_n_controls

    def test_sample_ordering_preserved(self):
        samples = _make_samples(8)
        lineage = null_lineage.build_null_lineage(
            samples=samples,
            source_artifact_sha256="b" * 64,
            permutation_indices=np.random.default_rng(2).permutation(len(samples)),
        )
        assert [s.sample_id for s in lineage["null_samples"]] == [s.sample_id for s in samples]

    def test_source_sample_objects_not_mutated(self):
        samples = _make_samples(8)
        original_labels = [s.label for s in samples]
        null_lineage.build_null_lineage(
            samples=samples,
            source_artifact_sha256="c" * 64,
            permutation_indices=np.random.default_rng(3).permutation(len(samples)),
        )
        assert [s.label for s in samples] == original_labels

    def test_null_samples_are_new_objects(self):
        samples = _make_samples(4)
        lineage = null_lineage.build_null_lineage(
            samples=samples,
            source_artifact_sha256="d" * 64,
            permutation_indices=np.random.default_rng(4).permutation(len(samples)),
        )
        for source_sample, null_sample in zip(samples, lineage["null_samples"], strict=True):
            assert source_sample is not null_sample


# ---------------------------------------------------------------------------
# Semantic lineage hash (section 27, tests A-E)
# ---------------------------------------------------------------------------


class TestLineageSemanticHash:
    def _base_kwargs(self):
        return {
            "source_artifact_sha256": "s" * 64,
            "sample_ids_sha256": "i" * 64,
            "original_labels_sha256": "o" * 64,
            "permutation_indices": [2, 0, 1],
            "permuted_labels_sha256": "p" * 64,
        }

    def test_a_identical_transformation_same_hash(self):
        kwargs = self._base_kwargs()
        assert null_lineage.compute_lineage_sha256(**kwargs) == null_lineage.compute_lineage_sha256(
            **kwargs
        )

    def test_b_seed_is_not_a_lineage_parameter(self):
        # compute_lineage_sha256 has no seed parameter at all: identical
        # indices/hashes always produce identical lineage_sha256 regardless
        # of which seed conceptually produced those indices.
        kwargs = self._base_kwargs()
        h1 = null_lineage.compute_lineage_sha256(**kwargs)
        h2 = null_lineage.compute_lineage_sha256(**kwargs)
        assert h1 == h2
        assert "permutation_seed" not in null_lineage.compute_lineage_sha256.__code__.co_varnames

    def test_c_changed_permutation_index_changes_hash(self):
        kwargs = self._base_kwargs()
        h1 = null_lineage.compute_lineage_sha256(**kwargs)
        kwargs["permutation_indices"] = [1, 0, 2]
        h2 = null_lineage.compute_lineage_sha256(**kwargs)
        assert h1 != h2

    def test_d_changed_source_sha_changes_hash(self):
        kwargs = self._base_kwargs()
        h1 = null_lineage.compute_lineage_sha256(**kwargs)
        kwargs["source_artifact_sha256"] = "t" * 64
        h2 = null_lineage.compute_lineage_sha256(**kwargs)
        assert h1 != h2

    def test_e_changed_original_labels_hash_changes_lineage_hash(self):
        kwargs = self._base_kwargs()
        h1 = null_lineage.compute_lineage_sha256(**kwargs)
        kwargs["original_labels_sha256"] = "z" * 64
        h2 = null_lineage.compute_lineage_sha256(**kwargs)
        assert h1 != h2

    def test_seed_change_with_fixed_indices_end_to_end_same_lineage(self, tmp_path):
        samples = _make_samples(9)
        source_sha = "e" * 64
        fixed_indices = [8, 0, 1, 2, 3, 4, 5, 6, 7]
        lineage_a = null_lineage.build_null_lineage(
            samples=samples, source_artifact_sha256=source_sha, permutation_indices=fixed_indices
        )
        lineage_b = null_lineage.build_null_lineage(
            samples=samples, source_artifact_sha256=source_sha, permutation_indices=fixed_indices
        )
        assert lineage_a["lineage_sha256"] == lineage_b["lineage_sha256"]


# ---------------------------------------------------------------------------
# Non-label content equality (section 28)
# ---------------------------------------------------------------------------


class TestNonLabelContentEquality:
    def test_variant_record_equal_true_for_identical(self):
        v1 = _make_variant()
        v2 = _make_variant()
        assert null_lineage.variant_record_equal(v1, v2)

    def test_variant_record_equal_false_on_annotation_diff(self):
        v1 = _make_variant(sift=0.05)
        v2 = _make_variant(sift=0.5)
        assert not null_lineage.variant_record_equal(v1, v2)

    def test_variant_record_equal_false_on_position_diff(self):
        v1 = _make_variant(pos=100)
        v2 = _make_variant(pos=200)
        assert not null_lineage.variant_record_equal(v1, v2)

    def test_deep_equal_handles_none_and_nan(self):
        assert null_lineage._deep_equal(None, None)
        assert null_lineage._deep_equal(float("nan"), float("nan"))
        assert not null_lineage._deep_equal(None, 0.0)

    def test_build_null_samples_preserves_non_label_content(self):
        samples = _make_samples(12)
        indices = np.random.default_rng(5).permutation(len(samples))
        null_samples = null_lineage.build_null_samples(samples, indices)
        null_lineage.assert_samples_non_label_equal(samples, null_samples)
        # Only labels may differ, and they must differ according to the exact permutation.
        expected_labels = null_lineage.apply_permutation_gather(
            null_lineage.extract_ordered_labels(samples), list(indices)
        )
        assert [s.label for s in null_samples] == expected_labels

    def test_sample_non_label_mismatch_detects_sex_change(self):
        samples = _make_samples(3)
        tampered = copy.deepcopy(samples)
        tampered[1].sex = "F" if samples[1].sex == "M" else "M"
        mismatch = null_lineage.sample_non_label_mismatch(samples[1], tampered[1])
        assert mismatch is not None and "sex" in mismatch

    def test_sample_non_label_mismatch_detects_sample_id_change(self):
        samples = _make_samples(3)
        tampered = copy.deepcopy(samples)
        tampered[0].sample_id = "different"
        mismatch = null_lineage.sample_non_label_mismatch(samples[0], tampered[0])
        assert mismatch is not None and "sample_id" in mismatch

    def test_sample_non_label_mismatch_detects_variant_change(self):
        samples = _make_samples(3)
        tampered = copy.deepcopy(samples)
        tampered[0].variants[0].genotype = 2
        mismatch = null_lineage.sample_non_label_mismatch(samples[0], tampered[0])
        assert mismatch is not None and "variant" in mismatch

    def test_assert_samples_non_label_equal_raises_on_mismatch(self):
        samples = _make_samples(3)
        tampered = copy.deepcopy(samples)
        tampered[0].sex = "X"
        with pytest.raises(ValueError, match="non-label content mismatch"):
            null_lineage.assert_samples_non_label_equal(samples, tampered)


# ---------------------------------------------------------------------------
# Strict CLI: creation, embedded metadata, sidecar
# ---------------------------------------------------------------------------


class TestStrictCreation:
    def test_creates_output_and_sidecar(self, tmp_path):
        _, output_path, sidecar_path, _, report = _create_valid_pair(tmp_path)
        assert output_path.exists()
        assert sidecar_path.exists()
        assert report["reused"] is False

    def test_source_bytes_unchanged_after_creation(self, tmp_path):
        source_path, _, _, _, _ = _create_valid_pair(tmp_path)
        before = _file_sha256(source_path)
        # Reload to be extra sure nothing lazily mutates on load either.
        torch.load(source_path, weights_only=False)
        after = _file_sha256(source_path)
        assert before == after

    def test_source_loaded_labels_unchanged_after_creation(self, tmp_path):
        source_path, _, _, samples, _ = _create_valid_pair(tmp_path)
        reloaded = torch.load(source_path, weights_only=False)
        assert [s.label for s in reloaded["samples"]] == [s.label for s in samples]

    def test_embedded_metadata_schema(self, tmp_path):
        _, output_path, _, _, _ = _create_valid_pair(tmp_path)
        null_data = torch.load(output_path, weights_only=False)
        meta = null_data["_null_baseline_metadata"]
        required_keys = {
            "schema_version",
            "is_null_baseline",
            "source_artifact_path",
            "original_path",
            "permutation_seed",
            "generator",
            "source_artifact_sha256",
            "sample_ids_sha256",
            "original_labels_sha256",
            "permuted_labels_sha256",
            "permutation_indices",
            "permutation_indices_sha256",
            "lineage_sha256",
            "n_samples",
            "n_cases",
            "n_controls",
            "same_position_count",
        }
        assert required_keys <= set(meta.keys())
        assert meta["schema_version"] == 1
        assert meta["is_null_baseline"] is True
        assert "null_artifact_sha256" not in meta
        assert meta["generator"]["script"] == "scripts/create_null_baseline.py"
        assert isinstance(meta["generator"]["argv"], list)

    def test_sidecar_schema(self, tmp_path):
        _, _, sidecar_path, _, _ = _create_valid_pair(tmp_path)
        sidecar = yaml.safe_load(sidecar_path.read_text())
        assert sidecar["schema_version"] == 1
        assert set(sidecar["source"].keys()) == {"path", "sha256"}
        assert set(sidecar["null"].keys()) == {"path", "sha256"}
        assert set(sidecar["samples"].keys()) == {
            "n_samples",
            "sample_ids_sha256",
            "n_cases",
            "n_controls",
            "same_position_count",
        }
        assert set(sidecar["labels"].keys()) == {"original_sha256", "permuted_sha256"}
        assert set(sidecar["permutation"].keys()) == {"seed", "indices", "indices_sha256"}
        assert set(sidecar["generator"].keys()) == {"repository_revision", "argv", "script"}

    def test_same_position_count_present_in_both_embedded_and_sidecar(self, tmp_path):
        _, output_path, sidecar_path, _, _ = _create_valid_pair(tmp_path)
        null_data = torch.load(output_path, weights_only=False)
        sidecar = yaml.safe_load(sidecar_path.read_text())
        assert "same_position_count" in null_data["_null_baseline_metadata"]
        assert "same_position_count" in sidecar["samples"]
        assert (
            null_data["_null_baseline_metadata"]["same_position_count"]
            == sidecar["samples"]["same_position_count"]
        )

    def test_top_level_metadata_preserved_verbatim(self, tmp_path):
        samples = _make_samples(6)
        source_path = tmp_path / "source.pt"
        metadata = {"genome_build": "GRCh38", "num_samples": 6, "note": "x"}
        _save_artifact(source_path, samples, metadata=metadata)
        output_path = tmp_path / "null.pt"
        create_null_baseline.create_strict_single_permutation(
            str(source_path), str(output_path), seed=7, reuse=False, argv=["x"]
        )
        null_data = torch.load(output_path, weights_only=False)
        assert null_data["metadata"] == metadata

    def test_repository_revision_matches_git_head(self, tmp_path):
        _, output_path, _, _, _ = _create_valid_pair(tmp_path)
        null_data = torch.load(output_path, weights_only=False)
        recorded = null_data["_null_baseline_metadata"]["generator"]["repository_revision"]
        actual_head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        assert recorded == actual_head

    def test_sidecar_filename_is_deterministic(self, tmp_path):
        null_path = tmp_path / "some_dir" / "null.pt"
        assert (
            null_lineage.sidecar_path_for(null_path)
            == tmp_path / "some_dir" / "null.pt.null-lineage.yaml"
        )


# ---------------------------------------------------------------------------
# Pair validator (section 30)
# ---------------------------------------------------------------------------


class TestPairValidator:
    def test_valid_pair_passes(self, tmp_path):
        source_path, output_path, sidecar_path, _, report = _create_valid_pair(tmp_path)
        result = null_lineage.validate_null_pair(source_path, output_path, sidecar_path)
        assert result["lineage_sha256"] == report["lineage_sha256"]
        assert result["null_artifact_sha256"] == report["null_artifact_sha256"]

    def test_missing_source(self, tmp_path):
        source_path, output_path, sidecar_path, _, _ = _create_valid_pair(tmp_path)
        source_path.unlink()
        with pytest.raises(ValueError, match="does not exist"):
            null_lineage.validate_null_pair(source_path, output_path, sidecar_path)

    def test_missing_null(self, tmp_path):
        source_path, output_path, sidecar_path, _, _ = _create_valid_pair(tmp_path)
        output_path.unlink()
        with pytest.raises(ValueError, match="does not exist"):
            null_lineage.validate_null_pair(source_path, output_path, sidecar_path)

    def test_missing_sidecar(self, tmp_path):
        source_path, output_path, sidecar_path, _, _ = _create_valid_pair(tmp_path)
        sidecar_path.unlink()
        with pytest.raises(ValueError, match="does not exist"):
            null_lineage.validate_null_pair(source_path, output_path, sidecar_path)

    def test_directory_where_file_required(self, tmp_path):
        source_path, output_path, sidecar_path, _, _ = _create_valid_pair(tmp_path)
        sidecar_path.unlink()
        sidecar_path.mkdir()
        with pytest.raises(ValueError, match="is not a file"):
            null_lineage.validate_null_pair(source_path, output_path, sidecar_path)

    def test_malformed_sidecar_yaml(self, tmp_path):
        source_path, output_path, sidecar_path, _, _ = _create_valid_pair(tmp_path)
        sidecar_path.write_text(": : : not valid yaml : : :\n\t- broken [")
        with pytest.raises(ValueError):
            null_lineage.validate_null_pair(source_path, output_path, sidecar_path)

    def test_unsupported_schema_version(self, tmp_path):
        source_path, output_path, sidecar_path, _, _ = _create_valid_pair(tmp_path)
        sidecar = yaml.safe_load(sidecar_path.read_text())
        sidecar["schema_version"] = 2
        sidecar_path.write_text(yaml.safe_dump(sidecar, sort_keys=False))
        with pytest.raises(ValueError, match="schema_version"):
            null_lineage.validate_null_pair(source_path, output_path, sidecar_path)

    def test_source_sha_mismatch_via_sidecar_tamper(self, tmp_path):
        source_path, output_path, sidecar_path, _, _ = _create_valid_pair(tmp_path)
        sidecar = yaml.safe_load(sidecar_path.read_text())
        sidecar["source"]["sha256"] = "f" * 64
        sidecar_path.write_text(yaml.safe_dump(sidecar, sort_keys=False))
        with pytest.raises(ValueError, match="sidecar.source.sha256"):
            null_lineage.validate_null_pair(source_path, output_path, sidecar_path)

    def test_null_sha_mismatch_via_sidecar_tamper(self, tmp_path):
        source_path, output_path, sidecar_path, _, _ = _create_valid_pair(tmp_path)
        sidecar = yaml.safe_load(sidecar_path.read_text())
        sidecar["null"]["sha256"] = "f" * 64
        sidecar_path.write_text(yaml.safe_dump(sidecar, sort_keys=False))
        with pytest.raises(ValueError, match="sidecar.null.sha256"):
            null_lineage.validate_null_pair(source_path, output_path, sidecar_path)

    def test_null_sample_order_mismatch(self, tmp_path):
        source_path, output_path, sidecar_path, _, _ = _create_valid_pair(tmp_path)
        null_data = torch.load(output_path, weights_only=False)
        null_data["samples"] = list(reversed(null_data["samples"]))
        torch.save(null_data, output_path)
        sidecar = yaml.safe_load(sidecar_path.read_text())
        sidecar["null"]["sha256"] = _file_sha256(output_path)
        sidecar_path.write_text(yaml.safe_dump(sidecar, sort_keys=False))
        with pytest.raises(ValueError, match="ordering"):
            null_lineage.validate_null_pair(source_path, output_path, sidecar_path)

    def test_source_sample_order_mismatch(self, tmp_path):
        source_path, output_path, sidecar_path, _, _ = _create_valid_pair(tmp_path)
        source_data = torch.load(source_path, weights_only=False)
        source_data["samples"] = list(reversed(source_data["samples"]))
        torch.save(source_data, source_path)
        sidecar = yaml.safe_load(sidecar_path.read_text())
        sidecar["source"]["sha256"] = _file_sha256(source_path)
        sidecar_path.write_text(yaml.safe_dump(sidecar, sort_keys=False))
        with pytest.raises(ValueError, match="ordering"):
            null_lineage.validate_null_pair(source_path, output_path, sidecar_path)

    def test_original_label_hash_mismatch(self, tmp_path):
        source_path, output_path, sidecar_path, _, _ = _create_valid_pair(tmp_path)
        null_data = torch.load(output_path, weights_only=False)
        null_data["_null_baseline_metadata"]["original_labels_sha256"] = "f" * 64
        torch.save(null_data, output_path)
        sidecar = yaml.safe_load(sidecar_path.read_text())
        sidecar["null"]["sha256"] = _file_sha256(output_path)
        sidecar_path.write_text(yaml.safe_dump(sidecar, sort_keys=False))
        with pytest.raises(ValueError, match="embedded.original_labels_sha256"):
            null_lineage.validate_null_pair(source_path, output_path, sidecar_path)

    def test_permuted_label_hash_mismatch(self, tmp_path):
        source_path, output_path, sidecar_path, _, _ = _create_valid_pair(tmp_path)
        null_data = torch.load(output_path, weights_only=False)
        null_data["_null_baseline_metadata"]["permuted_labels_sha256"] = "f" * 64
        torch.save(null_data, output_path)
        sidecar = yaml.safe_load(sidecar_path.read_text())
        sidecar["null"]["sha256"] = _file_sha256(output_path)
        sidecar_path.write_text(yaml.safe_dump(sidecar, sort_keys=False))
        with pytest.raises(ValueError, match="embedded.permuted_labels_sha256"):
            null_lineage.validate_null_pair(source_path, output_path, sidecar_path)

    def test_permutation_mismatch_gather_reconstruction_fails(self, tmp_path):
        source_path, output_path, sidecar_path, _, _ = _create_valid_pair(tmp_path)
        null_data = torch.load(output_path, weights_only=False)
        indices = null_data["_null_baseline_metadata"]["permutation_indices"]
        indices[0], indices[1] = indices[1], indices[0]
        torch.save(null_data, output_path)
        sidecar = yaml.safe_load(sidecar_path.read_text())
        sidecar["null"]["sha256"] = _file_sha256(output_path)
        sidecar_path.write_text(yaml.safe_dump(sidecar, sort_keys=False))
        with pytest.raises(ValueError, match="gather reconstruction failed"):
            null_lineage.validate_null_pair(source_path, output_path, sidecar_path)

    def test_invalid_permutation_structure(self, tmp_path):
        source_path, output_path, sidecar_path, _, _ = _create_valid_pair(tmp_path)
        null_data = torch.load(output_path, weights_only=False)
        indices = null_data["_null_baseline_metadata"]["permutation_indices"]
        indices[0] = indices[1]  # duplicate -> not a true permutation
        torch.save(null_data, output_path)
        sidecar = yaml.safe_load(sidecar_path.read_text())
        sidecar["null"]["sha256"] = _file_sha256(output_path)
        sidecar_path.write_text(yaml.safe_dump(sidecar, sort_keys=False))
        with pytest.raises(ValueError, match="exactly once"):
            null_lineage.validate_null_pair(source_path, output_path, sidecar_path)

    def test_lineage_hash_mismatch_sidecar_only(self, tmp_path):
        source_path, output_path, sidecar_path, _, _ = _create_valid_pair(tmp_path)
        sidecar = yaml.safe_load(sidecar_path.read_text())
        sidecar["lineage_sha256"] = "f" * 64
        sidecar_path.write_text(yaml.safe_dump(sidecar, sort_keys=False))
        with pytest.raises(ValueError, match="sidecar.lineage_sha256"):
            null_lineage.validate_null_pair(source_path, output_path, sidecar_path)

    def test_class_count_mismatch(self, tmp_path):
        source_path, output_path, sidecar_path, _, _ = _create_valid_pair(tmp_path)
        null_data = torch.load(output_path, weights_only=False)
        null_data["_null_baseline_metadata"]["n_cases"] += 1
        torch.save(null_data, output_path)
        sidecar = yaml.safe_load(sidecar_path.read_text())
        sidecar["null"]["sha256"] = _file_sha256(output_path)
        sidecar_path.write_text(yaml.safe_dump(sidecar, sort_keys=False))
        with pytest.raises(ValueError, match="embedded.n_cases"):
            null_lineage.validate_null_pair(source_path, output_path, sidecar_path)

    def test_same_position_count_mismatch_embedded(self, tmp_path):
        source_path, output_path, sidecar_path, _, _ = _create_valid_pair(tmp_path)
        null_data = torch.load(output_path, weights_only=False)
        null_data["_null_baseline_metadata"]["same_position_count"] += 1
        torch.save(null_data, output_path)
        sidecar = yaml.safe_load(sidecar_path.read_text())
        sidecar["null"]["sha256"] = _file_sha256(output_path)
        sidecar_path.write_text(yaml.safe_dump(sidecar, sort_keys=False))
        with pytest.raises(ValueError, match="embedded.same_position_count"):
            null_lineage.validate_null_pair(source_path, output_path, sidecar_path)

    def test_same_position_count_mismatch_sidecar(self, tmp_path):
        source_path, output_path, sidecar_path, _, _ = _create_valid_pair(tmp_path)
        sidecar = yaml.safe_load(sidecar_path.read_text())
        sidecar["samples"]["same_position_count"] += 1
        sidecar_path.write_text(yaml.safe_dump(sidecar, sort_keys=False))
        with pytest.raises(ValueError, match="sidecar.samples.same_position_count"):
            null_lineage.validate_null_pair(source_path, output_path, sidecar_path)

    def test_non_label_variant_mismatch(self, tmp_path):
        source_path, output_path, sidecar_path, _, _ = _create_valid_pair(tmp_path)
        null_data = torch.load(output_path, weights_only=False)
        null_data["samples"][0].variants[0].genotype = 2
        torch.save(null_data, output_path)
        sidecar = yaml.safe_load(sidecar_path.read_text())
        sidecar["null"]["sha256"] = _file_sha256(output_path)
        sidecar_path.write_text(yaml.safe_dump(sidecar, sort_keys=False))
        with pytest.raises(ValueError, match="non-label content mismatch"):
            null_lineage.validate_null_pair(source_path, output_path, sidecar_path)

    def test_sex_mismatch(self, tmp_path):
        source_path, output_path, sidecar_path, _, _ = _create_valid_pair(tmp_path)
        null_data = torch.load(output_path, weights_only=False)
        null_data["samples"][0].sex = "X"
        torch.save(null_data, output_path)
        sidecar = yaml.safe_load(sidecar_path.read_text())
        sidecar["null"]["sha256"] = _file_sha256(output_path)
        sidecar_path.write_text(yaml.safe_dump(sidecar, sort_keys=False))
        with pytest.raises(ValueError, match="non-label content mismatch"):
            null_lineage.validate_null_pair(source_path, output_path, sidecar_path)

    def test_top_level_metadata_mismatch(self, tmp_path):
        source_path, output_path, sidecar_path, _, _ = _create_valid_pair(tmp_path)
        null_data = torch.load(output_path, weights_only=False)
        null_data["metadata"] = dict(null_data["metadata"])
        null_data["metadata"]["genome_build"] = "DIFFERENT"
        torch.save(null_data, output_path)
        sidecar = yaml.safe_load(sidecar_path.read_text())
        sidecar["null"]["sha256"] = _file_sha256(output_path)
        sidecar_path.write_text(yaml.safe_dump(sidecar, sort_keys=False))
        with pytest.raises(ValueError, match="top-level preprocessing metadata"):
            null_lineage.validate_null_pair(source_path, output_path, sidecar_path)

    def test_embedded_sidecar_disagreement_generator(self, tmp_path):
        source_path, output_path, sidecar_path, _, _ = _create_valid_pair(tmp_path)
        sidecar = yaml.safe_load(sidecar_path.read_text())
        sidecar["generator"]["script"] = "scripts/other_script.py"
        sidecar_path.write_text(yaml.safe_dump(sidecar, sort_keys=False))
        with pytest.raises(ValueError, match="sidecar.generator.script"):
            null_lineage.validate_null_pair(source_path, output_path, sidecar_path)

    def test_path_relocation_still_validates_when_bytes_match(self, tmp_path):
        # A pair whose files are moved to a different directory (paths
        # inside the sidecar/embedded metadata now stale) must still
        # validate as long as byte identities are unchanged.
        source_path, output_path, sidecar_path, _, report = _create_valid_pair(tmp_path)
        relocated_dir = tmp_path / "relocated"
        relocated_dir.mkdir()
        new_source = relocated_dir / source_path.name
        new_output = relocated_dir / output_path.name
        new_sidecar = relocated_dir / sidecar_path.name
        new_source.write_bytes(source_path.read_bytes())
        new_output.write_bytes(output_path.read_bytes())
        new_sidecar.write_bytes(sidecar_path.read_bytes())

        result = null_lineage.validate_null_pair(new_source, new_output, new_sidecar)
        assert result["lineage_sha256"] == report["lineage_sha256"]


# ---------------------------------------------------------------------------
# Strict schema completeness (review correction: incomplete provenance must
# fail closed rather than silently passing via .get() defaults)
# ---------------------------------------------------------------------------


def _resave_null(output_path, null_data):
    torch.save(null_data, output_path)


def _resave_sidecar(sidecar_path, sidecar):
    sidecar_path.write_text(yaml.safe_dump(sidecar, sort_keys=False))


def _sync_null_sha_into_sidecar(output_path, sidecar_path):
    sidecar = yaml.safe_load(sidecar_path.read_text())
    sidecar["null"]["sha256"] = _file_sha256(output_path)
    _resave_sidecar(sidecar_path, sidecar)


class TestStrictSchemaValidation:
    def test_embedded_missing_source_artifact_path(self, tmp_path):
        source_path, output_path, sidecar_path, _, _ = _create_valid_pair(tmp_path)
        null_data = torch.load(output_path, weights_only=False)
        del null_data["_null_baseline_metadata"]["source_artifact_path"]
        _resave_null(output_path, null_data)
        _sync_null_sha_into_sidecar(output_path, sidecar_path)
        with pytest.raises(ValueError, match="embedded.source_artifact_path"):
            null_lineage.validate_null_pair(source_path, output_path, sidecar_path)

    def test_embedded_missing_permutation_seed(self, tmp_path):
        source_path, output_path, sidecar_path, _, _ = _create_valid_pair(tmp_path)
        null_data = torch.load(output_path, weights_only=False)
        del null_data["_null_baseline_metadata"]["permutation_seed"]
        _resave_null(output_path, null_data)
        _sync_null_sha_into_sidecar(output_path, sidecar_path)
        with pytest.raises(ValueError, match="embedded.permutation_seed"):
            null_lineage.validate_null_pair(source_path, output_path, sidecar_path)

    def test_embedded_missing_generator_script(self, tmp_path):
        source_path, output_path, sidecar_path, _, _ = _create_valid_pair(tmp_path)
        null_data = torch.load(output_path, weights_only=False)
        del null_data["_null_baseline_metadata"]["generator"]["script"]
        _resave_null(output_path, null_data)
        _sync_null_sha_into_sidecar(output_path, sidecar_path)
        with pytest.raises(ValueError, match="embedded.generator.script"):
            null_lineage.validate_null_pair(source_path, output_path, sidecar_path)

    def test_embedded_generator_argv_wrong_type(self, tmp_path):
        source_path, output_path, sidecar_path, _, _ = _create_valid_pair(tmp_path)
        null_data = torch.load(output_path, weights_only=False)
        null_data["_null_baseline_metadata"]["generator"]["argv"] = "not-a-list"
        _resave_null(output_path, null_data)
        _sync_null_sha_into_sidecar(output_path, sidecar_path)
        with pytest.raises(ValueError, match="embedded.generator.argv"):
            null_lineage.validate_null_pair(source_path, output_path, sidecar_path)

    def test_embedded_sha_field_invalid_format(self, tmp_path):
        source_path, output_path, sidecar_path, _, _ = _create_valid_pair(tmp_path)
        null_data = torch.load(output_path, weights_only=False)
        null_data["_null_baseline_metadata"]["sample_ids_sha256"] = "not-a-valid-hex-digest"
        _resave_null(output_path, null_data)
        _sync_null_sha_into_sidecar(output_path, sidecar_path)
        with pytest.raises(ValueError, match="hexadecimal"):
            null_lineage.validate_null_pair(source_path, output_path, sidecar_path)

    def test_embedded_integer_field_is_bool(self, tmp_path):
        source_path, output_path, sidecar_path, _, _ = _create_valid_pair(tmp_path)
        null_data = torch.load(output_path, weights_only=False)
        null_data["_null_baseline_metadata"]["n_cases"] = True
        _resave_null(output_path, null_data)
        _sync_null_sha_into_sidecar(output_path, sidecar_path)
        with pytest.raises(ValueError, match="embedded.n_cases"):
            null_lineage.validate_null_pair(source_path, output_path, sidecar_path)

    def test_embedded_permutation_seed_is_bool(self, tmp_path):
        source_path, output_path, sidecar_path, _, _ = _create_valid_pair(tmp_path)
        null_data = torch.load(output_path, weights_only=False)
        null_data["_null_baseline_metadata"]["permutation_seed"] = True
        _resave_null(output_path, null_data)
        _sync_null_sha_into_sidecar(output_path, sidecar_path)
        with pytest.raises(ValueError, match="embedded.permutation_seed"):
            null_lineage.validate_null_pair(source_path, output_path, sidecar_path)

    def test_sidecar_missing_source_path(self, tmp_path):
        source_path, output_path, sidecar_path, _, _ = _create_valid_pair(tmp_path)
        sidecar = yaml.safe_load(sidecar_path.read_text())
        del sidecar["source"]["path"]
        _resave_sidecar(sidecar_path, sidecar)
        with pytest.raises(ValueError, match="sidecar.source.path"):
            null_lineage.validate_null_pair(source_path, output_path, sidecar_path)

    def test_sidecar_missing_null_path(self, tmp_path):
        source_path, output_path, sidecar_path, _, _ = _create_valid_pair(tmp_path)
        sidecar = yaml.safe_load(sidecar_path.read_text())
        del sidecar["null"]["path"]
        _resave_sidecar(sidecar_path, sidecar)
        with pytest.raises(ValueError, match="sidecar.null.path"):
            null_lineage.validate_null_pair(source_path, output_path, sidecar_path)

    def test_sidecar_missing_permutation_seed(self, tmp_path):
        source_path, output_path, sidecar_path, _, _ = _create_valid_pair(tmp_path)
        sidecar = yaml.safe_load(sidecar_path.read_text())
        del sidecar["permutation"]["seed"]
        _resave_sidecar(sidecar_path, sidecar)
        with pytest.raises(ValueError, match="sidecar.permutation.seed"):
            null_lineage.validate_null_pair(source_path, output_path, sidecar_path)

    def test_sidecar_missing_generator_script(self, tmp_path):
        source_path, output_path, sidecar_path, _, _ = _create_valid_pair(tmp_path)
        sidecar = yaml.safe_load(sidecar_path.read_text())
        del sidecar["generator"]["script"]
        _resave_sidecar(sidecar_path, sidecar)
        with pytest.raises(ValueError, match="sidecar.generator.script"):
            null_lineage.validate_null_pair(source_path, output_path, sidecar_path)

    def test_sidecar_sha_field_invalid_format(self, tmp_path):
        source_path, output_path, sidecar_path, _, _ = _create_valid_pair(tmp_path)
        sidecar = yaml.safe_load(sidecar_path.read_text())
        sidecar["labels"]["original_sha256"] = "short"
        _resave_sidecar(sidecar_path, sidecar)
        with pytest.raises(ValueError, match="hexadecimal"):
            null_lineage.validate_null_pair(source_path, output_path, sidecar_path)

    def test_sidecar_integer_field_wrong_type(self, tmp_path):
        source_path, output_path, sidecar_path, _, _ = _create_valid_pair(tmp_path)
        sidecar = yaml.safe_load(sidecar_path.read_text())
        sidecar["samples"]["n_samples"] = "10"
        _resave_sidecar(sidecar_path, sidecar)
        with pytest.raises(ValueError, match="sidecar.samples.n_samples"):
            null_lineage.validate_null_pair(source_path, output_path, sidecar_path)

    def test_sidecar_permutation_seed_is_bool(self, tmp_path):
        source_path, output_path, sidecar_path, _, _ = _create_valid_pair(tmp_path)
        sidecar = yaml.safe_load(sidecar_path.read_text())
        sidecar["permutation"]["seed"] = False
        _resave_sidecar(sidecar_path, sidecar)
        with pytest.raises(ValueError, match="sidecar.permutation.seed"):
            null_lineage.validate_null_pair(source_path, output_path, sidecar_path)

    def test_embedded_schema_helper_direct_missing_generator(self):
        embedded = {
            "schema_version": 1,
            "is_null_baseline": True,
            "source_artifact_path": "a",
            "original_path": "b",
            "permutation_seed": 1,
            "source_artifact_sha256": "a" * 64,
            "sample_ids_sha256": "a" * 64,
            "original_labels_sha256": "a" * 64,
            "permuted_labels_sha256": "a" * 64,
            "permutation_indices": [0, 1],
            "permutation_indices_sha256": "a" * 64,
            "lineage_sha256": "a" * 64,
            "n_samples": 2,
            "n_cases": 1,
            "n_controls": 1,
            "same_position_count": 0,
        }
        with pytest.raises(ValueError, match="embedded.generator"):
            null_lineage._validate_embedded_metadata_schema(embedded)

    def test_sidecar_schema_helper_direct_missing_samples(self):
        sidecar = {
            "schema_version": 1,
            "lineage_sha256": "a" * 64,
            "source": {"path": "a", "sha256": "a" * 64},
            "null": {"path": "b", "sha256": "a" * 64},
            "labels": {"original_sha256": "a" * 64, "permuted_sha256": "a" * 64},
            "permutation": {"seed": 1, "indices": [0, 1], "indices_sha256": "a" * 64},
            "generator": {"repository_revision": "unknown", "argv": ["x"], "script": "s.py"},
        }
        with pytest.raises(ValueError, match="sidecar.samples"):
            null_lineage._validate_sidecar_schema(sidecar)


# ---------------------------------------------------------------------------
# Immutability / reuse (section 31)
# ---------------------------------------------------------------------------


class TestImmutabilityReuse:
    def test_output_equal_to_source_rejected(self, tmp_path):
        samples = _make_samples(4)
        source_path = tmp_path / "source.pt"
        _save_artifact(source_path, samples)
        with pytest.raises(ValueError, match="must not equal the source path"):
            create_null_baseline.create_strict_single_permutation(
                str(source_path), str(source_path), seed=1, reuse=False, argv=["x"]
            )

    def test_output_exists_without_sidecar_rejected(self, tmp_path):
        samples = _make_samples(4)
        source_path = tmp_path / "source.pt"
        _save_artifact(source_path, samples)
        output_path = tmp_path / "null.pt"
        output_path.write_bytes(b"not a real artifact")
        before = output_path.read_bytes()
        with pytest.raises(ValueError, match="without a lineage sidecar"):
            create_null_baseline.create_strict_single_permutation(
                str(source_path), str(output_path), seed=1, reuse=False, argv=["x"]
            )
        assert output_path.read_bytes() == before  # untouched pre-existing file

    def test_sidecar_exists_without_output_rejected(self, tmp_path):
        samples = _make_samples(4)
        source_path = tmp_path / "source.pt"
        _save_artifact(source_path, samples)
        output_path = tmp_path / "null.pt"
        sidecar_path = null_lineage.sidecar_path_for(output_path)
        sidecar_path.write_text("schema_version: 1\n")
        with pytest.raises(ValueError, match="incomplete pair"):
            create_null_baseline.create_strict_single_permutation(
                str(source_path), str(output_path), seed=1, reuse=False, argv=["x"]
            )

    def test_valid_pair_no_reuse_flag_rejected(self, tmp_path):
        source_path, output_path, _, _, _ = _create_valid_pair(tmp_path)
        with pytest.raises(ValueError, match="never silently overwrites"):
            create_null_baseline.create_strict_single_permutation(
                str(source_path), str(output_path), seed=42, reuse=False, argv=["x"]
            )

    def test_valid_pair_reuse_matching_identity_succeeds(self, tmp_path):
        source_path, output_path, _, _, report = _create_valid_pair(tmp_path, seed=42)
        reuse_report = create_null_baseline.create_strict_single_permutation(
            str(source_path), str(output_path), seed=42, reuse=True, argv=["x"]
        )
        assert reuse_report["reused"] is True
        assert reuse_report["lineage_sha256"] == report["lineage_sha256"]

    def test_valid_pair_reuse_different_identity_rejected(self, tmp_path):
        source_path, output_path, _, samples, _ = _create_valid_pair(tmp_path, seed=42)
        # A different seed will (overwhelmingly likely, for n=10) draw a
        # different permutation and therefore a different lineage_sha256.
        with pytest.raises(ValueError, match="different lineage_sha256"):
            create_null_baseline.create_strict_single_permutation(
                str(source_path), str(output_path), seed=43, reuse=True, argv=["x"]
            )

    def test_mismatched_existing_pair_reuse_rejected(self, tmp_path):
        source_path, output_path, sidecar_path, _, _ = _create_valid_pair(tmp_path, seed=42)
        sidecar = yaml.safe_load(sidecar_path.read_text())
        sidecar["lineage_sha256"] = "f" * 64
        sidecar_path.write_text(yaml.safe_dump(sidecar, sort_keys=False))
        with pytest.raises(ValueError):
            create_null_baseline.create_strict_single_permutation(
                str(source_path), str(output_path), seed=42, reuse=True, argv=["x"]
            )


# ---------------------------------------------------------------------------
# Atomic-write / failure cleanup (section 33)
# ---------------------------------------------------------------------------


class TestAtomicFailureCleanup:
    def test_rollback_when_sidecar_publish_fails(self, tmp_path, monkeypatch):
        samples = _make_samples(6)
        source_path = tmp_path / "source.pt"
        _save_artifact(source_path, samples)
        output_path = tmp_path / "null.pt"

        def boom(*args, **kwargs):
            raise RuntimeError("simulated sidecar publish failure")

        monkeypatch.setattr(create_null_baseline.null_lineage, "write_sidecar", boom)

        with pytest.raises(RuntimeError, match="simulated sidecar publish failure"):
            create_null_baseline.create_strict_single_permutation(
                str(source_path), str(output_path), seed=1, reuse=False, argv=["x"]
            )

        # Best-effort rollback: the artifact created by this invocation must not remain.
        assert not output_path.exists()
        assert not null_lineage.sidecar_path_for(output_path).exists()
        # No leftover temp files in the output directory.
        assert list(tmp_path.glob(".null.pt.tmp-*")) == []
        assert list(tmp_path.glob(".null.pt.null-lineage.yaml.tmp-*")) == []

    def test_retry_after_rollback_succeeds(self, tmp_path, monkeypatch):
        samples = _make_samples(6)
        source_path = tmp_path / "source.pt"
        _save_artifact(source_path, samples)
        output_path = tmp_path / "null.pt"

        def boom(*args, **kwargs):
            raise RuntimeError("simulated sidecar publish failure")

        monkeypatch.setattr(create_null_baseline.null_lineage, "write_sidecar", boom)
        with pytest.raises(RuntimeError):
            create_null_baseline.create_strict_single_permutation(
                str(source_path), str(output_path), seed=1, reuse=False, argv=["x"]
            )
        monkeypatch.undo()

        report = create_null_baseline.create_strict_single_permutation(
            str(source_path), str(output_path), seed=1, reuse=False, argv=["x"]
        )
        assert report["reused"] is False
        assert output_path.exists()

    def test_rollback_never_deletes_preexisting_artifact(self, tmp_path, monkeypatch):
        # Establish a valid pair first (this "pre-existing" artifact must survive).
        source_path, output_path, sidecar_path, _, first_report = _create_valid_pair(
            tmp_path, seed=1
        )
        before_bytes = output_path.read_bytes()

        def boom(*args, **kwargs):
            raise RuntimeError("simulated failure during reuse validation path")

        # Force a fresh-creation attempt at the SAME path without --reuse: this must be
        # rejected by the existing-output policy before any write occurs, so nothing
        # created by a prior invocation can be rolled back by this one.
        with pytest.raises(ValueError, match="never silently overwrites"):
            create_null_baseline.create_strict_single_permutation(
                str(source_path), str(output_path), seed=1, reuse=False, argv=["x"]
            )
        assert output_path.read_bytes() == before_bytes


# ---------------------------------------------------------------------------
# Backward compatibility (section 32)
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_historical_create_single_permutation_unchanged(self, tmp_path):
        input_path = tmp_path / "input.pt"
        output_path = tmp_path / "output.pt"
        labels = torch.cat([torch.ones(10), torch.zeros(10)]).long()
        torch.save({"labels": labels}, input_path)

        stats = create_null_baseline.create_single_permutation(
            str(input_path), str(output_path), seed=7
        )
        assert stats["n_cases"] == 10
        assert stats["n_controls"] == 10

        permuted = torch.load(output_path, weights_only=False)
        assert "_null_baseline_metadata" in permuted
        legacy_meta = permuted["_null_baseline_metadata"]
        # Historical shape only: no strict-only keys leak into legacy metadata.
        assert set(legacy_meta.keys()) == {
            "is_null_baseline",
            "permutation_seed",
            "original_path",
            "n_samples",
            "n_cases",
            "n_controls",
            "same_position_count",
        }
        assert not null_lineage.sidecar_path_for(output_path).exists()

    def test_historical_multi_permutation_workflow_unchanged(self, tmp_path):
        input_path = tmp_path / "input.pt"
        output_dir = tmp_path / "perms"
        torch.save({"labels": torch.randint(0, 2, (30,))}, input_path)

        all_stats = create_null_baseline.create_multiple_permutations(
            str(input_path), str(output_dir), n_permutations=3, base_seed=100
        )
        assert len(all_stats) == 3
        for i in range(3):
            perm_file = output_dir / f"preprocessed_NULL_perm{i}.pt"
            assert perm_file.exists()
            assert not null_lineage.sidecar_path_for(perm_file).exists()
        assert (output_dir / "permutation_summary.txt").exists()

    def test_cli_help_documents_strict_flags(self):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "create_null_baseline.py"), "--help"],
            capture_output=True,
            text=True,
            check=True,
        )
        assert "--strict-lineage" in result.stdout
        assert "--reuse" in result.stdout

    def test_cli_historical_invocation_creates_no_sidecar(self, tmp_path):
        input_path = tmp_path / "input.pt"
        output_path = tmp_path / "output.pt"
        torch.save({"labels": torch.randint(0, 2, (12,))}, input_path)

        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "create_null_baseline.py"),
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--seed",
                "5",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert output_path.exists()
        assert not null_lineage.sidecar_path_for(output_path).exists()

    def test_cli_rejects_strict_lineage_with_output_dir(self, tmp_path):
        input_path = tmp_path / "input.pt"
        torch.save({"labels": torch.randint(0, 2, (12,))}, input_path)
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "create_null_baseline.py"),
                "--input",
                str(input_path),
                "--output-dir",
                str(tmp_path / "out"),
                "--strict-lineage",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "does not support --output-dir" in result.stderr

    def test_cli_rejects_reuse_without_strict_lineage(self, tmp_path):
        input_path = tmp_path / "input.pt"
        output_path = tmp_path / "output.pt"
        torch.save({"labels": torch.randint(0, 2, (12,))}, input_path)
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "create_null_baseline.py"),
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--reuse",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "--reuse requires --strict-lineage" in result.stderr


# ---------------------------------------------------------------------------
# Optional hardening (narrow, review-approved): reject null-of-null sources,
# and clean up a fresh publication that fails its own self-check.
# ---------------------------------------------------------------------------


class TestOptionalHardening:
    def test_rejects_source_already_marked_as_null_baseline(self, tmp_path):
        samples = _make_samples(6)
        source_path = tmp_path / "source.pt"
        torch.save(
            {
                "samples": samples,
                "metadata": {"genome_build": "GRCh37"},
                "_null_baseline_metadata": {"is_null_baseline": True, "permutation_seed": 1},
            },
            source_path,
        )
        output_path = tmp_path / "null.pt"
        with pytest.raises(ValueError, match="already a null baseline"):
            create_null_baseline.create_strict_single_permutation(
                str(source_path), str(output_path), seed=1, reuse=False, argv=["x"]
            )

    def test_self_check_failure_removes_both_freshly_published_files(self, tmp_path, monkeypatch):
        samples = _make_samples(6)
        source_path = tmp_path / "source.pt"
        _save_artifact(source_path, samples)
        output_path = tmp_path / "null.pt"

        def boom(*args, **kwargs):
            raise ValueError("simulated self-check failure")

        monkeypatch.setattr(create_null_baseline.null_lineage, "validate_null_pair", boom)

        with pytest.raises(RuntimeError, match="failed its own self-check"):
            create_null_baseline.create_strict_single_permutation(
                str(source_path), str(output_path), seed=1, reuse=False, argv=["x"]
            )

        assert not output_path.exists()
        assert not null_lineage.sidecar_path_for(output_path).exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
