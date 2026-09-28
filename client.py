"""Multi-Layer Perceptron (MLP) Feedforward Engine.
100% Python Standard Library.
"""

import math
import random

class DenseMLP:
    """Feedforward Multi-Layer Perceptron with Xavier uniform weight initialization."""
    def __init__(self, layer_sizes: list):
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            limit = math.sqrt(6.0 / (fan_in + fan_out))
            W = [[random.uniform(-limit, limit) for _ in range(fan_out)] for _ in range(fan_in)]
            b = [0.0 for _ in range(fan_out)]
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, x: list) -> list:
        curr = x
        for l, (W, b) in enumerate(zip(self.weights, self.biases)):
            fan_in = len(W)
            fan_out = len(W[0])
            out = []
            for j in range(fan_out):
                val = sum(curr[i] * W[i][j] for i in range(fan_in)) + b[j]
                if l < len(self.weights) - 1:
                    val = max(0.0, val)  # ReLU on hidden layers
                out.append(val)
            curr = out
        return curr
