import pandas as pd
import pytest

from features import prepare_features


def row(**kw):
    base = {"age": 30, "education_num": 9, "hours_per_week": 40,
            "workclass": "Private", "occupation": "Sales"}
    return pd.DataFrame([{**base, **kw}])


@pytest.mark.parametrize("edu,level", [(1, 1), (3, 1), (4, 2), (8, 2), (9, 3), (12, 3), (13, 4), (16, 4)])
def test_education_level_bins(edu, level):
    assert prepare_features(row(education_num=edu))["education_level"].iloc[0] == level


def test_groups():
    out = prepare_features(row(workclass="State-gov", occupation="Prof-specialty"))
    assert out["workclass_group"].iloc[0] == "Government"
    assert out["occupation_group"].iloc[0] == "Management-Professional"


@pytest.mark.parametrize("missing", [None, "?"])
def test_missing_values_become_unknown(missing):
    out = prepare_features(row(workclass=missing, occupation=missing))
    assert out["workclass_group"].iloc[0] == "Other-unknown"
    assert out["occupation_group"].iloc[0] == "Unknown"


def test_unknown_category_raises():
    with pytest.raises(ValueError):
        prepare_features(row(occupation="Astronaute"))
