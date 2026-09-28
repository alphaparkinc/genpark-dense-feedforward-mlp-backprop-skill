# genpark-dense-feedforward-mlp-backprop-skill

[![GitHub Stars](https://img.shields.io/github/stars/alphaparkinc/genpark-dense-feedforward-mlp-backprop-skill?style=social)](https://github.com/alphaparkinc/genpark-dense-feedforward-mlp-backprop-skill)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Zero External Dependencies](https://img.shields.io/badge/dependencies-0%20(pure%20standard%20library)-brightgreen.svg)](client.py)
[![MCP Ready](https://img.shields.io/badge/MCP-Ready-purple.svg)](mcp_server.py)

> **Multi-layer perceptron (MLP) feedforward inference with Xavier/Glorot initialization and activations**

Part of the **GenPark Autonomous Agent Matrix**, developed for production AI agents implementing on-device deep learning primitives, attention blocks, and differentiable computational graphs.

---

## 🏗️ Architecture

```mermaid
flowchart TD
    A[Tensors / Layer Inputs] --> B[genpark-dense-feedforward-mlp-backprop-skill]
    B --> C[Pure Python Standard Library Autograd / NN Engine]
    C --> D[Activated Embeddings / Attention Weights / Gradients]
    B --> E[MCP Protocol Endpoint stdio]
    E --> F[Cursor / Claude Desktop / Windsurf Integration]
```

## 🚀 Quickstart

### Native Python Execution
```bash
python example_usage.py
```

### Standard Library Verification
```python
from client import *
```

### MCP Server (Claude Desktop / Cursor)
```json
{
  "mcpServers": {
    "genpark-dense-feedforward-mlp-backprop-skill": {
      "command": "python",
      "args": ["-m", "genpark_dense_feedforward_mlp_backprop_skill.mcp_server"]
    }
  }
}
```

## 📄 License
MIT License.
