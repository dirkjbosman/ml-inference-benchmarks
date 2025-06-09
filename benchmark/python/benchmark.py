import onnxruntime as ort
import numpy as np
import time

session = ort.InferenceSession("/model/churn_model.onnx")

input_names = [i.name for i in session.get_inputs()]
print("Expected inputs:", set(input_names))

input_data = {
    "last_login_days": np.array([[2.0]], dtype=np.float32),
    "support_tickets": np.array([["3"]], dtype=str),
    "subscription_length_months": np.array([["12"]], dtype=str),
    "country": np.array([["US"]], dtype=str)
}

session.run(None, input_data)

start = time.time()
for _ in range(1000):
    session.run(None, input_data)
end = time.time()

total_ms = (end - start) * 1000
avg_ms = total_ms / 1000

print("\n## Inference Benchmark Results (1000 iterations)")
print("| Language | Total Time (ms) | Avg Time/Inference (ms) |")
print("|----------|------------------|--------------------------|")
print(f"| Python   | {total_ms:.0f}             | {avg_ms:.2f}                      |")
