FROM python:3.10-slim

WORKDIR /app

COPY train/train_model.py .
COPY train/mock_churn_dataset.csv .
COPY train/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "train_model.py"]
