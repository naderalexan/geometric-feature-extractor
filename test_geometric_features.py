import unittest

from geometric_features import GeometricFeature, ShapeClass, extract_features


def feature_value(features, kind):
    return next(feature.value for feature in features if feature.kind == kind)


class ExtractFeaturesTests(unittest.TestCase):
    def test_extracts_measurements_for_single_pixel(self):
        features = extract_features([[1]])
        self.assertEqual(
            [feature.value for feature in features[:4]],
            [1, 4, (0.0, 0.0), (0, 0, 0, 0)],
        )
        self.assertEqual(feature_value(features, GeometricFeature.SHAPE_CLASSES), (ShapeClass.POINT,))

    def test_extracts_measurements_for_rectangle(self):
        features = extract_features([[1, 1, 0], [1, 1, 0]])
        self.assertEqual([feature.value for feature in features[:4]], [4, 8, (0.5, 0.5), (0, 0, 1, 1)])
        self.assertEqual(feature_value(features, GeometricFeature.SHAPE_CLASSES), (ShapeClass.RECTANGLE,))

    def test_detects_line_triangle_and_circle(self):
        line = extract_features([[1, 1, 1]])
        triangle = extract_features([[1, 0, 0], [1, 1, 0], [1, 1, 1]])
        circle = extract_features([
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
        ])
        self.assertEqual(feature_value(line, GeometricFeature.SHAPE_CLASSES), (ShapeClass.LINE,))
        self.assertEqual(feature_value(triangle, GeometricFeature.SHAPE_CLASSES), (ShapeClass.TRIANGLE,))
        self.assertEqual(feature_value(circle, GeometricFeature.SHAPE_CLASSES), (ShapeClass.CIRCLE,))

    def test_detects_components_and_holes(self):
        features = extract_features([
            [1, 1, 1, 0, 1],
            [1, 0, 1, 0, 0],
            [1, 1, 1, 0, 1],
        ])
        self.assertEqual(feature_value(features, GeometricFeature.COMPONENT_COUNT), 3)
        self.assertEqual(feature_value(features, GeometricFeature.HOLE_COUNT), 1)
        components = feature_value(features, GeometricFeature.COMPONENTS)
        self.assertEqual([component.hole_count for component in components], [1, 0, 0])

    def test_nonzero_cells_are_foreground(self):
        self.assertEqual(extract_features([[0, -1]])[0].value, 1)

    def test_empty_bitmap_has_no_features(self):
        self.assertEqual(extract_features([]), [])
        self.assertEqual(extract_features([[], []]), [])

    def test_rejects_ragged_rows(self):
        with self.assertRaises(ValueError):
            extract_features([[1], [1, 0]])

    def test_rejects_non_integer_cells(self):
        with self.assertRaises(TypeError):
            extract_features([[1, 0.5]])

    def test_enum_values_are_integers(self):
        self.assertEqual(GeometricFeature.AREA.value, 1)
        self.assertEqual(GeometricFeature.AREA.name, "AREA")


if __name__ == "__main__":
    unittest.main()
