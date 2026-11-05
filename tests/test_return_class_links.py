"""Return-type docstrings must name classes that exist."""

import pythermalcomfort.classes_return as classes_return
from pythermalcomfort.models.cooling_effect import cooling_effect
from pythermalcomfort.models.net import net
from pythermalcomfort.models.set_tmp import set_tmp
from pythermalcomfort.models.vertical_tmp_grad_ppd import vertical_tmp_grad_ppd


def test_return_class_sphinx_links_name_real_classes() -> None:
    """The documented return class is the class the function returns."""
    cases = (
        (net, "NET"),
        (set_tmp, "SET"),
        (cooling_effect, "CE"),
        (vertical_tmp_grad_ppd, "VerticalTGradPPD"),
    )
    stale = (
        "classes_return.Net`",
        "classes_return.SetTmp`",
        "classes_return.CoolingEffect`",
        "classes_return.VerticalTmpGradPPD`",
    )
    for function, class_name in cases:
        target = f":py:class:`~pythermalcomfort.classes_return.{class_name}`"
        assert target in function.__doc__
        assert f"returned `{class_name}` instance" in function.__doc__
        assert hasattr(classes_return, class_name)
        for name in stale:
            assert name not in function.__doc__
