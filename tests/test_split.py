"""Tests for the leakage-safe grouped split."""

import random

from src.data.split import assign_splits, split_report


def _synthetic_label_map(n_strips=40, seed=7):
    rng = random.Random(seed)
    classes = [str(c) for c in range(8)]
    label_map = {}
    for strip_idx in range(n_strips):
        sid = f"ESP_{strip_idx:06d}_0001_RED"
        n_crops = rng.randint(5, 60)
        for crop_idx in range(n_crops):
            cls = rng.choices(classes, weights=[60, 8, 3, 5, 4, 1, 3, 1])[0]
            suffix = rng.choice(["", "-r90", "-r180", "-r270", "-fh", "-fv", "-brt"])
            fname = f"{sid}-{crop_idx:04d}{suffix}.jpg"
            label_map[fname] = cls
    return label_map


def test_no_strip_crosses_a_split_boundary():
    label_map = _synthetic_label_map()
    assignment = assign_splits(label_map, val_fraction=0.15, test_fraction=0.15)

    from src.data.common import strip_id_for

    strip_to_splits: dict[str, set[str]] = {}
    for fname, split in assignment.items():
        sid = strip_id_for(fname)
        strip_to_splits.setdefault(sid, set()).add(split)

    offenders = {sid: splits for sid, splits in strip_to_splits.items() if len(splits) > 1}
    assert offenders == {}


def test_every_image_assigned_exactly_once():
    label_map = _synthetic_label_map()
    assignment = assign_splits(label_map)
    assert set(assignment.keys()) == set(label_map.keys())
    assert set(assignment.values()) <= {"train", "val", "test"}


def test_split_fractions_are_reasonably_close_to_target():
    label_map = _synthetic_label_map(n_strips=80, seed=3)
    assignment = assign_splits(label_map, val_fraction=0.15, test_fraction=0.15)
    report = split_report(label_map, assignment)

    # Greedy heuristic over a small synthetic set won't hit targets exactly;
    # assert it's in the right ballpark, not exact.
    assert 0.55 < report["per_split_fraction"]["train"] < 0.85
    assert 0.05 < report["per_split_fraction"]["val"] < 0.30
    assert 0.05 < report["per_split_fraction"]["test"] < 0.30


def test_deterministic_given_same_input():
    label_map = _synthetic_label_map()
    a1 = assign_splits(label_map, val_fraction=0.2, test_fraction=0.2)
    a2 = assign_splits(label_map, val_fraction=0.2, test_fraction=0.2)
    assert a1 == a2


def test_rejects_invalid_fractions():
    label_map = _synthetic_label_map(n_strips=2)
    try:
        assign_splits(label_map, val_fraction=0.6, test_fraction=0.6)
        assert False, "expected ValueError"
    except ValueError:
        pass
