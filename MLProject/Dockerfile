FROM python:3.12-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir mlflow==2.19.0 tensorflow==2.18.0 numpy==1.26.4 pandas==2.2.2 scikit-learn==1.5.1

CMD ["mlflow", "run", ".", "-e", "main"]