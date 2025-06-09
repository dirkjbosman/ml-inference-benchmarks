FROM ubuntu:22.04 AS builder

RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    wget \
    unzip \
    curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /src

RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-linux-x64-1.17.0.tgz && \
    tar -xzf onnxruntime-linux-x64-1.17.0.tgz

COPY benchmark/cpp/main.cpp .
COPY benchmark/cpp/CMakeLists.txt .

ENV ONNXRUNTIME_DIR=/src/onnxruntime-linux-x64-1.17.0
RUN cmake . -DONNXRUNTIME_DIR=$ONNXRUNTIME_DIR && make

FROM ubuntu:22.04

RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /src/cpp_benchmark /app/cpp_benchmark
COPY --from=builder /src/onnxruntime-linux-x64-1.17.0/lib/libonnxruntime.so.1.17.0 /usr/lib/
RUN ln -s /usr/lib/libonnxruntime.so.1.17.0 /usr/lib/libonnxruntime.so

COPY model /app/model
WORKDIR /app

CMD ["./cpp_benchmark"]
