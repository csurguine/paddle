"""
PADDLE - Predictive Analytics and Data-driven Decision Learning Engine

A comprehensive framework for predictive and prescriptive analytics on large-scale datasets.
"""

__version__ = "0.1.0"
__author__ = "PADDLE Team"

from paddle.data.generator import DataGenerator
from paddle.models.predictive import PredictiveModels
from paddle.models.prescriptive import PrescriptiveModels

__all__ = [
    "DataGenerator",
    "PredictiveModels",
    "PrescriptiveModels",
]
