"""Tests for the acquisition script's safety behavior (not the network path)."""

from src.data.download import main


def test_refuses_placeholder_url(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("sys.argv", ["download.py", "--output-dir", str(tmp_path)])
    exit_code = main()
    assert exit_code == 1
