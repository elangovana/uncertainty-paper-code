import os.path
from unittest import TestCase

from pyparsing import results

from medical_uncertainty.gemini_inference import GeminiInference


class TestGeminiInference(TestCase):
    def test_infer(self):
        # Arrange
        # Image
        image = os.path.join(os.path.dirname(__file__), "data", "view1_64741_frontal.jpg")
        sut = GeminiInference()

        # Act
        results = sut.infer(image)

        print(results)
