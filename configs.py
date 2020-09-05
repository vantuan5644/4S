import argparse
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Argument of insightface model
IMAGE_SIZE = (112, 112)

COSINE_THRESHOLD = 0

# Path to ONNX model
ARCFACE_WEIGHT = os.path.join(ROOT_DIR, 'weights/arcface/resnet100.onnx')

