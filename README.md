# Geometric feature extractor

A small Python library for extracting geometric measurements, connected components, holes, and approximate shape classes from binary segmentation masks.

## Segment Anything masks

SAM's default binary-mask output is a 2D boolean array in each mask's `segmentation` field, not a bit-string. Automatic mask generation can also return uncompressed or COCO-compressed RLE. This adapter accepts a **string serialization** of the default mask (JSON 2D array or JSON object with `segmentation`) and uncompressed COCO RLE. It also accepts row-separated bits (`101\n111`) or dimensioned row-major bits (`3x2:101111`). A flattened bit string without dimensions is inherently ambiguous; dimensioned bits or JSON are recommended.

References: [SAM README](https://github.com/facebookresearch/segment-anything) · [Automatic mask generator](https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/automatic_mask_generator.py)

```python
from adapters import BitmapFeatureExtractor
from geometric_features import GeometricFeature

extractor = BitmapFeatureExtractor()
features = extractor.extract('{"segmentation": [[true, true], [true, false]]}')
by_kind = {feature.kind: feature.value for feature in features}
print(by_kind[GeometricFeature.AREA])  # 3
print(by_kind[GeometricFeature.COMPONENTS][0].shape)
print(extractor.variance_from("2x2:1000", "2x2:1100"))
```

## Architecture and features

`FeatureExtractor` in `ports.py` is the input-string port. `BitmapFeatureExtractor` in `adapters.py` parses a serialized SAM mask and delegates to the bitmap implementation in `geometric_features.py`. `variance_from(mask_a, mask_b)` returns population variance per shared numeric feature; tuple-valued measurements return one variance per coordinate. Categorical values are omitted.

`GeometricFeature` is an integer enum: `AREA=1`, `PERIMETER=2`, `CENTROID=3`, `BOUNDING_BOX=4`, `COMPONENT_COUNT=5`, `HOLE_COUNT=6`, `COMPONENTS=7`, and `SHAPE_CLASSES=8`. Each component reports area, perimeter, centroid, inclusive bounding box, hole count, and a `ShapeClass` (`POINT`, `LINE`, `RECTANGLE`, `TRIANGLE`, `CIRCLE`, or `IRREGULAR`). Components use 8-connectivity; perimeter and holes use 4-connectivity. Centroids use `(x, y)` pixel-index coordinates. Shape classification is heuristic and intended for clean, filled shapes.

## Development

Requires Python 3.10+. Manage the project with [uv](https://docs.astral.sh/uv/):

```sh
uv sync
uv run python -m unittest -v
```
