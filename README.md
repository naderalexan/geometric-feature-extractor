# Geometric feature extractor

Extracts measurements, connected components, holes, and approximate shape classes from a 2D integer bitmap. Every nonzero cell is foreground.

```python
from geometric_features import GeometricFeature, extract_features

features = extract_features([
    [1, 1, 1],
    [1, 0, 1],
    [1, 1, 1],
])
by_kind = {feature.kind: feature.value for feature in features}
print(by_kind[GeometricFeature.HOLE_COUNT])  # 1
print(by_kind[GeometricFeature.COMPONENTS][0].shape)  # ShapeClass.RECTANGLE
```

`GeometricFeature` is an integer enum: `AREA=1`, `PERIMETER=2`, `CENTROID=3`, `BOUNDING_BOX=4`, `COMPONENT_COUNT=5`, `HOLE_COUNT=6`, `COMPONENTS=7`, and `SHAPE_CLASSES=8`. Each `Feature` contains a kind and value. Component details include area, perimeter, centroid, inclusive bounding box, hole count, and a `ShapeClass` (`POINT`, `LINE`, `RECTANGLE`, `TRIANGLE`, `CIRCLE`, or `IRREGULAR`). Component order is top-to-bottom, then left-to-right.

Foreground components use 8-connectivity; perimeter and background holes use 4-connectivity. Centroids use `(x, y)` pixel-index coordinates; bounding boxes are `(min_x, min_y, max_x, max_y)`. Shape classification is heuristic and intended for clean, filled bitmap shapes, not arbitrary contours. Empty foreground returns an empty list. Run tests with `python -m unittest`.
