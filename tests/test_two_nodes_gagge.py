import pytest

from pythermalcomfort.models import two_nodes_gagge
from tests.conftest import Urls, retrieve_reference_table, validate_result


def test_two_nodes(get_test_url, retrieve_data) -> None:
    """Test that the function calculates the two nodes Gagge model correctly for various inputs."""
    reference_table = retrieve_reference_table(
        get_test_url,
        retrieve_data,
        Urls.TWO_NODES.name,
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = two_nodes_gagge(**inputs)

        validate_result(result, outputs, tolerance)


def test_two_nodes_gagge_docstring_example() -> None:
    """The published example must match the values the function returns."""
    scalar = two_nodes_gagge(tdb=25, tr=25, v=0.1, rh=50, clo=0.5, met=1.2)
    assert scalar.w == pytest.approx(0.12)

    array = two_nodes_gagge(tdb=[25, 25], tr=25, v=0.3, rh=50, met=1.2, clo=0.5)
    assert array.e_skin == pytest.approx([16.17, 16.17])

    example = two_nodes_gagge.__doc__.split("Examples", 1)[1]
    assert "0.12" in example
    assert "[16.17, 16.17]" in example
    assert "100.0" not in example
