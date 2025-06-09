# ml-inference-benchmarks

## Overview:

This project compares the inference performance of a simple machine learning model across multiple languages, with a focus on Python and C++. Using the same ONNX-serialized model, we can evaluate how fast each language can run predictions over 1000 iterations.

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
- Use it as a base to expand your own tests on

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

Key Takeaways From My Tests:
* C++ outperformed Python by ~32% in this setup (as expected).
* Both use ONNX Runtime under the hood, but C++ avoids Python’s interpreter overhead.
* The performance gap is modest for small models, but could widen with larger models or heavier preprocessing.
* While Python wins in developer speed and ease of use, C++ (or Rust/Go) might be worth exploring if you’re chasing ultra-low latency on edge or CPU-bound systems. 
* It was interesting to see that Rust or Go is not so well supported in ONNX. Potential gap for development or better support in the future.

| Language | Total Time (ms) | Avg Time/Inference (ms) |
|----------|------------------|------------------------|
| Python   | 43               | 0.04                   |
| C++      | 32.461           | 0.032461               |
