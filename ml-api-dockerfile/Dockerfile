FROM ghcr.io/mlflow/mlflow:v3.10.0

WORKDIR /mlflow

VOLUME /mlflow

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]