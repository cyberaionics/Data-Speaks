from windowing import calculate_window_label


SEIZURE = [(130.0, 212.0)]


def test_labels_around_seizure():
    assert calculate_window_label(64.0, 68.0, SEIZURE) == 0
    assert calculate_window_label(66.0, 70.0, SEIZURE) == -1

    assert calculate_window_label(126.0, 130.0, SEIZURE) == -1
    assert calculate_window_label(128.0, 132.0, SEIZURE) == 1
    assert calculate_window_label(130.0, 134.0, SEIZURE) == 1

    assert calculate_window_label(208.0, 212.0, SEIZURE) == 1
    assert calculate_window_label(210.0, 214.0, SEIZURE) == 1

    assert calculate_window_label(212.0, 216.0, SEIZURE) == -1
    assert calculate_window_label(272.0, 276.0, SEIZURE) == -1
    assert calculate_window_label(274.0, 278.0, SEIZURE) == 0


if __name__ == "__main__":
    test_labels_around_seizure()
    print("All window-label tests passed.")
