import pytest

from pythermalcomfort.models import discomfort_index
from tests.conftest import Urls, retrieve_reference_table, validate_result


def test_discomfort_index(get_test_url, retrieve_data) -> None:
    """Test that the function calculates the Discomfort Index correctly for various inputs."""
    reference_table = retrieve_reference_table(
        get_test_url,
        retrieve_data,
        Urls.DISCOMFORT_INDEX.name,
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = discomfort_index(**inputs)

        validate_result(result, outputs, tolerance)


def test_discomfort_index_docstring_example() -> None:
    """The published example must match the values the function returns."""
    scalar = discomfort_index(tdb=25, rh=50)
    assert scalar.di == pytest.approx(22.1)
    assert scalar.discomfort_condition == "Less than 50% feels discomfort"

    array = discomfort_index(tdb=[25, 30], rh=[50, 60])
    assert array.di == pytest.approx([22.1, 26.6])
    assert array.discomfort_condition.tolist() == [
        "Less than 50% feels discomfort",
        "More than 50% feels discomfort",
    ]
    example = discomfort_index.__doc__.split("Examples", 1)[1]
    assert "[22.1, 26.6]" in example
    assert "27.3" not in example
    assert "More than 50% feels discomfort" in example
