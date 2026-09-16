from src.data.splits import check_filename_leakage, class_distribution, load_split


def test_load_split(tmp_path):
    (tmp_path / "train-labels.txt").write_text("a.jpg 0\nb.jpg 1\n")
    samples = load_split(tmp_path, "train")
    assert samples == [("a.jpg", 0), ("b.jpg", 1)]


def test_check_filename_leakage_detects_duplicate():
    splits = {
        "train": [("a.jpg", 0), ("b.jpg", 1)],
        "val": [("b.jpg", 1), ("c.jpg", 2)],
    }
    leaks = check_filename_leakage(splits)
    assert leaks == {"b.jpg": ["train", "val"]}


def test_check_filename_leakage_clean():
    splits = {
        "train": [("a.jpg", 0)],
        "val": [("b.jpg", 1)],
    }
    assert check_filename_leakage(splits) == {}


def test_class_distribution():
    samples = [("a.jpg", 0), ("b.jpg", 0), ("c.jpg", 1)]
    dist = class_distribution(samples)
    assert dist == {0: 2, 1: 1}
