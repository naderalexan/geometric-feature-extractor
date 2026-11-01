import unittest

from adapters import BitmapFeatureExtractor
from geometric_features import GeometricFeature
from ports import FeatureExtractor


class BitmapFeatureExtractorTests(unittest.TestCase):
    def setUp(self):
        self.extractor = BitmapFeatureExtractor()

    def test_adapter_satisfies_extractor_port(self):
        self.assertIsInstance(self.extractor, FeatureExtractor)
        self.assertEqual(self.extractor, BitmapFeatureExtractor())
        self.assertNotEqual(self.extractor, object())

    def test_extracts_a_json_serialized_sam_binary_mask(self):
        features = self.extractor.extract('{"segmentation": [[true, true], [false, true]]}')
        self.assertEqual(features[0].value, 3)
        self.assertEqual(features[1].value, 8)

    def test_extracts_uncompressed_coco_rle(self):
        features = self.extractor.extract('{"size": [2, 3], "counts": [0, 1, 2, 1, 2]}')
        self.assertEqual(features[0].value, 2)

    def test_extracts_dimensioned_and_multiline_bits(self):
        dimensioned = self.extractor.extract("3x2:100010")
        rows = self.extractor.extract("100\n010")
        self.assertEqual(dimensioned[0].value, 2)
        self.assertEqual(rows[0].value, 2)

    def test_rejects_compressed_rle_with_actionable_error(self):
        with self.assertRaisesRegex(ValueError, "integer counts"):
            self.extractor.extract('{"size": [1, 1], "counts": "compressed"}')

    def test_variance_is_reported_per_numeric_feature(self):
        variances = self.extractor.variance_from("[[1,0],[0,0]]", "[[1,1],[0,0]]")
        self.assertEqual(variances[GeometricFeature.AREA], 0.25)
        self.assertEqual(variances[GeometricFeature.PERIMETER], 1.0)
        self.assertEqual(variances[GeometricFeature.CENTROID], (0.0625, 0.0))
        self.assertEqual(variances[GeometricFeature.COMPONENT_COUNT], 0.0)
        self.assertNotIn(GeometricFeature.SHAPE_CLASSES, variances)


if __name__ == "__main__":
    unittest.main()
