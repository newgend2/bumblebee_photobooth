import sys
import unittest
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR / "tools"))

from label_studio_to_yolo_seg import label_lines_from_results


def polygon_result(points):
    return {
        "type": "polygonlabels",
        "value": {
            "polygonlabels": ["aruco_paper"],
            "points": points,
        },
    }


class LabelStudioToYoloSegTest(unittest.TestCase):
    def assert_exported_points(self, points):
        lines, warnings = label_lines_from_results([polygon_result(points)])
        self.assertEqual([], warnings)
        self.assertEqual(1, len(lines))
        parts = lines[0].split()
        self.assertEqual("0", parts[0])
        self.assertEqual(1 + len(points) * 2, len(parts))

    def test_exports_four_point_polygon(self):
        self.assert_exported_points([
            [10, 10],
            [90, 10],
            [90, 90],
            [10, 90],
        ])

    def test_exports_five_point_polygon(self):
        self.assert_exported_points([
            [10, 10],
            [55, 8],
            [90, 10],
            [90, 90],
            [10, 90],
        ])

    def test_exports_eight_point_polygon(self):
        self.assert_exported_points([
            [10, 10],
            [45, 8],
            [90, 10],
            [92, 45],
            [90, 90],
            [55, 92],
            [10, 90],
            [8, 45],
        ])

    def test_rejects_too_few_points(self):
        lines, warnings = label_lines_from_results([
            polygon_result([
                [10, 10],
                [90, 10],
                [90, 90],
            ])
        ])
        self.assertEqual([], lines)
        self.assertIn("4-8 points", warnings[0])


if __name__ == "__main__":
    unittest.main()
