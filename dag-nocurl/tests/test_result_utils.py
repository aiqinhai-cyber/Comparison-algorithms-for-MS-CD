from result_utils import mean_std_text


def test_mean_std_text_can_format_percentages() -> None:
    assert mean_std_text(0.81234, 0.01234, scale=100.0) == "81.23 ± 1.23"
