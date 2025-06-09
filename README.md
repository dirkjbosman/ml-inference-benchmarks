# ml-inference-benchmarks

## Overview:

This project compares the inference performance of a simple machine learning model across multiple languages, with a focus on Python and C++. Using the same ONNX-serialized model, we evaluate how fast each language can run predictions over 1000 iterations.

The pipeline includes:
- ✅ A mock churn prediction model (RandomForest) trained in Python
- 📦 Exported to ONNX format for cross-platform compatibility
- 🐍 Python-based inference using onnxruntime
- 💻 C++-based inference using the ONNX Runtime C++ API
- 🐳 Docker-based benchmarking setup for reproducibility

This repo is ideal for developers interested in:
- Profiling ML inference latency
- Understanding ONNX Runtime usage across languages
- Comparing Python vs. C++ performance in real-world deployment

## Steps

### Step 1: Train Your Model

```sh
docker build -t train-model -f train/Dockerfile .
docker run --rm -v $(pwd)/model:/app/model train-model
```

### Step 2: Build & Run Benchmark Container To Determine Speed of Inference

#### (a) Python
```sh
docker build -t bench-python -f benchmark/python/Dockerfile .
docker run --rm bench-python
```

#### (b) C++
```sh
docker build -t bench-cpp  -f benchmark/cpp/Dockerfile .
docker run --rm bench-cpp
```

### Step 3: View Inference Benchmark Results (1000 iterations)
| Language | Total Time (ms) | Avg Time/Inference (ms) |
|----------|------------------|------------------------|
| Python   | 43               | 0.04                   |
| C++      | 32.461           | 0.032461               |
