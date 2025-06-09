#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <chrono>

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "benchmark");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    Ort::Session session(env, "model/churn_model.onnx", session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    const char* input_names[] = {
        "last_login_days",
        "support_tickets",
        "subscription_length_months",
        "country"
    };
    const char* output_names[] = { "output_label" };

    std::vector<int64_t> str_shape{1, 1};
    std::vector<int64_t> float_shape{1, 1};

    std::vector<float> last_login_days = {4.0f};
    Ort::Value input_float = Ort::Value::CreateTensor<float>(
        allocator.GetInfo(), last_login_days.data(), last_login_days.size(),
        float_shape.data(), float_shape.size());

    std::vector<const char*> tickets = { "2" };
    std::vector<const char*> subs = { "3" };
    std::vector<const char*> country = { "1" };

    Ort::Value input_tickets = Ort::Value::CreateTensor(allocator, str_shape.data(), str_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);
    Ort::Value input_subs = Ort::Value::CreateTensor(allocator, str_shape.data(), str_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);
    Ort::Value input_country = Ort::Value::CreateTensor(allocator, str_shape.data(), str_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);

    input_tickets.FillStringTensor(tickets.data(), tickets.size());
    input_subs.FillStringTensor(subs.data(), subs.size());
    input_country.FillStringTensor(country.data(), country.size());

    std::vector<Ort::Value> inputs;
    inputs.emplace_back(std::move(input_float));
    inputs.emplace_back(std::move(input_tickets));
    inputs.emplace_back(std::move(input_subs));
    inputs.emplace_back(std::move(input_country));

    auto _ = session.Run(Ort::RunOptions{nullptr}, input_names, inputs.data(), 4, output_names, 1);

    const int iterations = 1000;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        auto out = session.Run(Ort::RunOptions{nullptr}, input_names, inputs.data(), 4, output_names, 1);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> total = end - start;

    std::cout << "## Inference Benchmark Results (1000 iterations)\n";
    std::cout << "| Language | Total Time (ms) | Avg Time/Inference (ms) |\n";
    std::cout << "|----------|------------------|--------------------------|\n";
    std::cout << "| C++      | " << total.count() << " | " << (total.count() / iterations) << " |\n";

    return 0;
}
