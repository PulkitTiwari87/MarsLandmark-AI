"""Tests for the acquisition script's safety behavior (not the network path)."""

from src.data.download import EXPECTED_SHA256, PRIMARY_URL, main


def test_refuses_placeholder_url(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sys.argv",
        ["download.py", "--output-dir", str(tmp_path), "--url", "https://zenodo.org/records/REPLACE_WITH_X/files/f.zip"],
    )
    exit_code = main()
    assert exit_code == 1


def test_default_url_is_not_a_placeholder():
    assert "REPLACE_WITH" not in PRIMARY_URL
    assert PRIMARY_URL == "https://zenodo.org/records/2538136/files/hirise-map-proj-v3.zip"


def test_expected_checksum_is_set_and_well_formed():
    assert EXPECTED_SHA256 is not None
    assert len(EXPECTED_SHA256) == 64
    int(EXPECTED_SHA256, 16)  # raises if not valid hex
