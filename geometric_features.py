from collections import deque
from dataclasses import dataclass
from enum import Enum, IntEnum
from math import pi
from typing import TypeAlias


class GeometricFeature(IntEnum):
    AREA = 1
    PERIMETER = 2
    CENTROID = 3
    BOUNDING_BOX = 4
    COMPONENT_COUNT = 5
    HOLE_COUNT = 6
    COMPONENTS = 7
    SHAPE_CLASSES = 8


class ShapeClass(str, Enum):
    POINT = "point"
    LINE = "line"
    RECTANGLE = "rectangle"
    TRIANGLE = "triangle"
    CIRCLE = "circle"
    IRREGULAR = "irregular"


@dataclass(frozen=True)
class Component:
    index: int
    area: int
    perimeter: int
    centroid: tuple[float, float]
    bounding_box: tuple[int, int, int, int]
    hole_count: int
    shape: ShapeClass


FeatureValue: TypeAlias = (
    int
    | float
    | tuple[float, float]
    | tuple[int, int, int, int]
    | tuple[Component, ...]
    | tuple[ShapeClass, ...]
)


@dataclass(frozen=True)
class Feature:
    kind: GeometricFeature
    value: FeatureValue


def _validate_bitmap(bitmap: list[list[int]]) -> None:
    if not isinstance(bitmap, list) or any(not isinstance(row, list) for row in bitmap):
        raise TypeError("bitmap must be a list of lists")
    width = len(bitmap[0]) if bitmap else 0
    if any(len(row) != width for row in bitmap):
        raise ValueError("bitmap rows must all have the same length")
    if any(
        not isinstance(value, int) or isinstance(value, bool)
        for row in bitmap
        for value in row
    ):
        raise TypeError("bitmap cells must be integers")


def _measure(points: set[tuple[int, int]]) -> tuple[int, tuple[float, float], tuple[int, int, int, int], int]:
    area = len(points)
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    perimeter = sum(
        (x + dx, y + dy) not in points
        for x, y in points
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0))
    )
    centroid = (sum(xs) / area, sum(ys) / area)
    bounds = (min(xs), min(ys), max(xs), max(ys))
    return area, centroid, bounds, perimeter


def _components(foreground: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    remaining = set(foreground)
    result = []
    neighbors = tuple(
        (dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dx or dy
    )
    while remaining:
        start = min(remaining, key=lambda point: (point[1], point[0]))
        remaining.remove(start)
        component = {start}
        queue = deque([start])
        while queue:
            x, y = queue.popleft()
            for dx, dy in neighbors:
                neighbor = (x + dx, y + dy)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        result.append(component)
    return result


def _convex_hull_vertex_count(points: set[tuple[int, int]]) -> int:
    ordered = sorted(points)
    if len(ordered) <= 1:
        return len(ordered)

    def cross(origin: tuple[int, int], a: tuple[int, int], b: tuple[int, int]) -> int:
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])

    lower = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return len(lower[:-1] + upper[:-1])


def _classify_shape(
    points: set[tuple[int, int]],
    bounds: tuple[int, int, int, int],
    perimeter: int,
) -> ShapeClass:
    if len(points) == 1:
        return ShapeClass.POINT

    ordered = list(points)
    x0, y0 = ordered[0]
    x1, y1 = ordered[1]
    if all((x - x0) * (y1 - y0) == (y - y0) * (x1 - x0) for x, y in ordered[2:]):
        return ShapeClass.LINE

    min_x, min_y, max_x, max_y = bounds
    width, height = max_x - min_x + 1, max_y - min_y + 1
    area = len(points)
    if area == width * height:
        return ShapeClass.RECTANGLE

    density = area / (width * height)
    hull_vertices = _convex_hull_vertex_count(points)
    if hull_vertices == 3 and density >= 0.35:
        return ShapeClass.TRIANGLE

    aspect_ratio = max(width, height) / min(width, height)
    circularity = 4 * pi * area / (perimeter * perimeter)
    if aspect_ratio <= 1.25 and hull_vertices >= 4 and circularity >= 0.35:
        return ShapeClass.CIRCLE
    return ShapeClass.IRREGULAR


def _hole_counts(
    bitmap: list[list[int]],
    components: list[set[tuple[int, int]]],
) -> tuple[int, list[int]]:
    height = len(bitmap)
    width = len(bitmap[0]) if bitmap else 0
    labels = {point: i for i, points in enumerate(components) for point in points}
    background = {
        (x, y)
        for y, row in enumerate(bitmap)
        for x, value in enumerate(row)
        if value == 0
    }
    hole_total = 0
    component_holes = [0] * len(components)
    while background:
        start = background.pop()
        region = {start}
        queue = deque([start])
        touches_edge = False
        adjacent_components = set()
        while queue:
            x, y = queue.popleft()
            touches_edge |= x == 0 or y == 0 or x == width - 1 or y == height - 1
            for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                neighbor = (x + dx, y + dy)
                if neighbor in background:
                    background.remove(neighbor)
                    region.add(neighbor)
                    queue.append(neighbor)
                elif neighbor in labels:
                    adjacent_components.add(labels[neighbor])
        if not touches_edge:
            hole_total += 1
            for index in adjacent_components:
                component_holes[index] += 1
    return hole_total, component_holes


def extract_features(bitmap: list[list[int]]) -> list[Feature]:
    """Extract global and per-component features from a 2D integer bitmap.

    Every nonzero cell is foreground. Foreground components use 8-connectivity;
    perimeter and background holes use 4-connectivity. Centroids use (x, y)
    pixel-index coordinates, and bounding boxes are inclusive.
    """
    _validate_bitmap(bitmap)
    foreground = {
        (x, y)
        for y, row in enumerate(bitmap)
        for x, value in enumerate(row)
        if value != 0
    }
    if not foreground:
        return []

    area, centroid, bounds, perimeter = _measure(foreground)
    components = _components(foreground)
    hole_count, component_holes = _hole_counts(bitmap, components)
    details = []
    for index, points in enumerate(components):
        component_area, component_centroid, component_bounds, component_perimeter = _measure(points)
        details.append(
            Component(
                index=index,
                area=component_area,
                perimeter=component_perimeter,
                centroid=component_centroid,
                bounding_box=component_bounds,
                hole_count=component_holes[index],
                shape=_classify_shape(points, component_bounds, component_perimeter),
            )
        )

    return [
        Feature(GeometricFeature.AREA, area),
        Feature(GeometricFeature.PERIMETER, perimeter),
        Feature(GeometricFeature.CENTROID, centroid),
        Feature(GeometricFeature.BOUNDING_BOX, bounds),
        Feature(GeometricFeature.COMPONENT_COUNT, len(components)),
        Feature(GeometricFeature.HOLE_COUNT, hole_count),
        Feature(GeometricFeature.COMPONENTS, tuple(details)),
        Feature(GeometricFeature.SHAPE_CLASSES, tuple(component.shape for component in details)),
    ]
