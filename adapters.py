import json
import re

from geometric_features import Feature, extract_features
from ports import FeatureExtractor


_DIMENSIONED_BITS = re.compile(r"\s*(\d+)\s*[xX]\s*(\d+)\s*:\s*([01]*)\s*")


class BitmapFeatureExtractor(FeatureExtractor):
    """Adapter for SAM binary masks serialized as JSON, RLE, or bit rows."""

    def extract(self, mask: str) -> list[Feature]:
        bitmap = self._decode(mask)
        return extract_features(bitmap)

    @classmethod
    def _decode(cls, mask: str) -> list[list[int]]:
        if not isinstance(mask, str):
            raise TypeError("mask must be a string")

        dimensioned = _DIMENSIONED_BITS.fullmatch(mask)
        if dimensioned:
            width, height, bits = int(dimensioned[1]), int(dimensioned[2]), dimensioned[3]
            if len(bits) != width * height:
                raise ValueError("bit count does not match mask dimensions")
            if width == 0 or height == 0:
                if width == height == 0:
                    return []
                raise ValueError("mask dimensions must both be zero or both be positive")
            return [
                [int(bit) for bit in bits[offset : offset + width]]
                for offset in range(0, len(bits), width)
            ]

        if re.fullmatch(r"\s*[01]+(?:\s*\n\s*[01]+)*\s*", mask):
            return cls._decode_bit_rows(mask)

        try:
            payload = json.loads(mask)
        except json.JSONDecodeError:
            return cls._decode_bit_rows(mask)

        if isinstance(payload, dict):
            payload = payload.get("segmentation", payload)
            if isinstance(payload, dict):
                return cls._decode_uncompressed_rle(payload)
        return cls._decode_matrix(payload)

    @classmethod
    def _decode_matrix(cls, matrix: object) -> list[list[int]]:
        if not isinstance(matrix, list) or any(not isinstance(row, list) for row in matrix):
            raise ValueError("mask JSON must contain a 2D array or uncompressed RLE")
        bitmap = []
        for row in matrix:
            if any(not isinstance(value, (bool, int)) or value not in (0, 1) for value in row):
                raise ValueError("SAM binary mask cells must be 0 or 1")
            bitmap.append([int(value) for value in row])
        return bitmap

    @classmethod
    def _decode_uncompressed_rle(cls, rle: dict) -> list[list[int]]:
        size = rle.get("size")
        counts = rle.get("counts")
        if (
            not isinstance(size, list)
            or len(size) != 2
            or any(not isinstance(value, int) or value < 0 for value in size)
            or not isinstance(counts, list)
            or any(not isinstance(count, int) or count < 0 for count in counts)
        ):
            raise ValueError("RLE mask must have [height, width] size and integer counts")

        height, width = size
        total = height * width
        if sum(counts) != total:
            raise ValueError("RLE counts must sum to the mask area")
        flat = []
        foreground = False
        for count in counts:
            flat.extend([int(foreground)] * count)
            foreground = not foreground

        bitmap = [[0] * width for _ in range(height)]
        for index, value in enumerate(flat):
            if value:
                y, x = index % height, index // height
                bitmap[y][x] = 1
        return bitmap

    @classmethod
    def _decode_bit_rows(cls, mask: str) -> list[list[int]]:
        rows = [line.strip() for line in mask.splitlines() if line.strip()]
        if not rows or any(set(row) - {"0", "1"} for row in rows):
            raise ValueError("mask must be JSON, dimensioned bits, or rows of 0/1 bits")
        if any(len(row) != len(rows[0]) for row in rows):
            raise ValueError("bit mask rows must all have the same length")
        return [[int(bit) for bit in row] for row in rows]
