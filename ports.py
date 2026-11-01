from abc import ABC, abstractmethod
from numbers import Real
from typing import TypeAlias

from geometric_features import Feature, GeometricFeature


VarianceValue: TypeAlias = float | tuple[float, ...]


class FeatureExtractor(ABC):
    @abstractmethod
    def extract(self, mask: str) -> list[Feature]:
        """Extract geometric features from a serialized mask."""

    def variance_from(
        self, first_mask: str, second_mask: str
    ) -> dict[GeometricFeature, VarianceValue]:
        """Return per-feature population variance for shared numeric outputs.

        Tuple-valued features return one variance per coordinate. Categorical
        features and values without a matching numeric counterpart are skipped.
        """
        first = {feature.kind: feature.value for feature in self.extract(first_mask)}
        second = {feature.kind: feature.value for feature in self.extract(second_mask)}
        variances = {}
        for kind in first.keys() & second.keys():
            left, right = first[kind], second[kind]
            if isinstance(left, Real) and isinstance(right, Real):
                variances[kind] = ((float(left) - float(right)) / 2) ** 2
            elif (
                isinstance(left, tuple)
                and isinstance(right, tuple)
                and len(left) == len(right)
                and all(isinstance(value, Real) for value in (*left, *right))
            ):
                variances[kind] = tuple(
                    ((float(a) - float(b)) / 2) ** 2 for a, b in zip(left, right)
                )
        return variances

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and vars(self) == vars(other)

    __hash__ = None
