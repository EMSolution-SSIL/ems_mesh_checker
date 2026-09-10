from __future__ import annotations

import pytest

from ems_mesh_checker.cli import (
    _selected_property_groups,
    _validate_merge_groups,
)


def test_property_filter_and_merge_build_expected_result_sets() -> None:
    groups = _selected_property_groups(
        (1, 2, 3, 4),
        requested_properties=(1, 2, 4),
        merge_groups=((1, 2),),
    )

    assert groups == ((1, 2), (4,))


def test_merge_without_filter_keeps_other_properties_separate() -> None:
    groups = _selected_property_groups(
        (1, 2, 3, 4),
        requested_properties=None,
        merge_groups=((1, 2),),
    )

    assert groups == ((1, 2), (3,), (4,))


def test_merge_groups_must_not_overlap() -> None:
    with pytest.raises(ValueError, match="must not overlap"):
        _validate_merge_groups(((1, 2), (2, 3)))


def test_merge_members_must_be_selected_when_filter_is_used() -> None:
    with pytest.raises(ValueError, match="must remain selected"):
        _selected_property_groups(
            (1, 2, 3),
            requested_properties=(1, 3),
            merge_groups=((1, 2),),
        )


def test_property_exclusion_keeps_other_properties_separate() -> None:
    groups = _selected_property_groups(
        (1, 2, 3, 4),
        requested_properties=None,
        excluded_properties=(2, 4),
    )

    assert groups == ((1,), (3,))


def test_property_exclusion_can_be_combined_with_merge() -> None:
    groups = _selected_property_groups(
        (1, 2, 3, 4),
        requested_properties=None,
        excluded_properties=(4,),
        merge_groups=((1, 2),),
    )

    assert groups == ((1, 2), (3,))


def test_excluded_property_must_exist() -> None:
    with pytest.raises(ValueError, match="excluded Property IDs are not present"):
        _selected_property_groups(
            (1, 2, 3),
            requested_properties=None,
            excluded_properties=(9,),
        )


def test_property_exclusion_must_leave_at_least_one_property() -> None:
    with pytest.raises(ValueError, match="no Property IDs remain"):
        _selected_property_groups(
            (1, 2),
            requested_properties=None,
            excluded_properties=(1, 2),
        )


def test_excluded_property_cannot_be_a_merge_member() -> None:
    with pytest.raises(ValueError, match="must remain selected"):
        _selected_property_groups(
            (1, 2, 3),
            requested_properties=None,
            excluded_properties=(2,),
            merge_groups=((1, 2),),
        )
