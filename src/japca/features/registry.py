from __future__ import annotations

from dataclasses import dataclass

from japca.config import load_feature_groups_config


@dataclass(frozen=True)
class FeatureSetSpec:
    name: str
    include_groups: tuple[str, ...]
    variables: tuple[str, ...]
    derived: tuple[str, ...]
    group_membership: dict[str, dict[str, tuple[str, ...]]]


def resolve_feature_set(name: str) -> FeatureSetSpec:
    config = load_feature_groups_config()
    groups = config["groups"]
    feature_set = config["feature_sets"][name]
    include_groups = tuple(feature_set["include_groups"])

    variables: list[str] = []
    derived: list[str] = []
    membership: dict[str, dict[str, tuple[str, ...]]] = {}
    for group_name in include_groups:
        group = groups[group_name]
        group_variables = tuple(group.get("variables", []))
        group_derived = tuple(group.get("derived", []))
        membership[group_name] = {
            "variables": group_variables,
            "derived": group_derived,
        }
        variables.extend(group_variables)
        derived.extend(group_derived)

    return FeatureSetSpec(
        name=name,
        include_groups=include_groups,
        variables=tuple(dict.fromkeys(variables)),
        derived=tuple(dict.fromkeys(derived)),
        group_membership=membership,
    )
