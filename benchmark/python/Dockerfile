FROM python:3.10-slim
RUN pip install onnxruntime numpy
COPY ../model /model
COPY benchmark/python/benchmark.py /app/
CMD ["python", "/app/benchmark.py"]
