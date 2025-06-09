import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType
import onnx


df = pd.read_csv("mock_churn_dataset.csv")
X = df.drop(columns=["churn"])
y = df["churn"]

numeric_features = ["last_login_days"]
categorical_features = ["support_tickets", "subscription_length_months", "country"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier())
])

pipeline.fit(X, y)

initial_types = [
    ("last_login_days", FloatTensorType([None, 1])),
    ("support_tickets", StringTensorType([None, 1])),
    ("subscription_length_months", StringTensorType([None, 1])),
    ("country", StringTensorType([None, 1])),
]

onnx_model = convert_sklearn(pipeline, initial_types=initial_types, target_opset=11)

onnx_model.ir_version = 7

os.makedirs("model", exist_ok=True)
with open("model/churn_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("Model trained and saved to model/churn_model.onnx")
