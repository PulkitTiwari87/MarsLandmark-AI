from src.data.labels import DEFAULT_CLASS_NAMES, load_class_map, parse_label_file


def test_parse_label_file(tmp_path):
    label_file = tmp_path / "labels.txt"
    label_file.write_text("# comment\nimg_0001.jpg 0\nimg_0002.jpg 3\n\n")
    samples = parse_label_file(label_file)
    assert samples == [("img_0001.jpg", 0), ("img_0002.jpg", 3)]


def test_load_class_map_falls_back_to_default(tmp_path):
    missing = tmp_path / "does_not_exist.txt"
    class_map = load_class_map(missing)
    assert class_map == dict(enumerate(DEFAULT_CLASS_NAMES))


def test_load_class_map_from_file(tmp_path):
    class_map_file = tmp_path / "class_map.txt"
    class_map_file.write_text("0,crater\n1,other\n")
    class_map = load_class_map(class_map_file)
    assert class_map == {0: "crater", 1: "other"}
