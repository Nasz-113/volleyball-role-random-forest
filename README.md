# End-to-End MLOps for Machine Learning
Learning full process of MLOps using various open source tools.

Overview
- This repository is a practical, end-to-end MLOps demo combining a FastAPI backend with a data processing/pipeline layer using MLflow, Prefect, and a small training workflow. It is organized as a monorepo with separate backend and pipeline components.

Key components
- backend: FastAPI-based REST API for health checks, prediction, and (light) training orchestration.
- pipeline: ML workflow orchestration, including MLflow experiments, deployment scripts, and orchestration flows (Prefect).
- shared data & models: sample artifacts, pre-trained models, label maps, and example datasets used by the pipeline.

Getting started
- Prerequisites: Python 3.12+ (recommended), Docker (optional for containerized runs), and Git.
- Install dependencies for local development (one-off per component):
-   Backend: `pip install -r backend/app/requirements.txt`
-   Pipeline: `pip install -r pipeline/requirements.txt`
- Optional: create and activate a virtual environment per component.

Project structure
- root/
  - README.md  (this file)
- backend/
  - app/        (FastAPI application, including main.py, routes, services, models)
  - Dockerfile
- pipeline/
  - flows/      (Prefect flows and ML orchestration)
  - Dockerfile
- Notebooks, artifacts and example data are stored under pipeline/flows/mlruns and pipeline/flows/mlflow.db.

Common run flows (local development)
- Backend API
  - Move into the backend directory and start the FastAPI server:
    ```bash
    cd backend
    python -m venv .venv
    source .venv/bin/activate
    pip install -r app/requirements.txt
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    ```
- MLflow server (experience tracking for the pipeline)
  - From the root or pipeline directory:
    ```bash
    mlflow server --backend-store-uri sqlite:///pipeline/flows/mlflow.db --default-artifact-root pipeline/flows/mlruns
    ```
- Prefect Server (orchestrator for flows)
  - From the root or pipeline directory:
    ```bash
    prefect server start
    ```
- Prefect deployment (if you have a deployment script)
  - From the pipeline/flows directory:
    ```bash
    python3 deploy.py
    ```
- End-to-end API usage (example)
  - With the backend running, you can call the prediction endpoint (configure your own client as needed).

- Docker and orchestration
- The repository already contains a compose configuration at the repo root: `compose.yaml`. Use Docker Compose to bring up the full stack.
- Supported commands (choose one):
  - `docker compose -f compose.yaml up -d`  (modern Docker CLI, space-separated command)
  - `docker-compose -f compose.yaml up -d`    (classic CLI)
- The compose file wires the following services:
  - ml-api (backend FastAPI)
  - ml-pipeline (pipeline components)
  - mlflow (MLflow tracking server)
  - prefect-server (Prefect server)
- Ports exposed by the compose file:
  - ml-api: 8000
  - mlflow: 5000
  - prefect-server: 4200
- The file also defines a shared network and optional data volumes under ./data.

Environment + configuration
- The backend uses a .env file for local configuration (placed at backend/app/.env in this repo). Ensure to copy and fill necessary values (database URLs, API keys, etc.).
- The pipeline stores MLflow state under pipeline/flows/mlruns and a sqlite database at pipeline/flows/mlflow.db for example persistence.

Contributing
- See CONTRIBUTING.md (if present) or open an issue to discuss improvements.

Notes
- This project is a learning scaffold. Depending on your environment, you may need to adjust paths to script entry points, Python versions, or service commands.
 
## Future Improvements
- Docker and orchestration
  - Add a docker-compose.override.yaml for development vs production, tighten service definitions, and pin image versions.
  - Consider using named volumes for persistence (database data, MLflow artifacts) and documenting backup/restore strategies.
- Health checks and reliability
  - Add healthcheck blocks in compose.yaml for ml-api, mlflow, and prefect-server; implement readiness checks before dependent services start.
- Observability and security
  - Integrate simple logging/metrics (e.g., Prometheus/Grafana) and ensure sensitive data is not logged.
- CI/CD and automation
  - Add GitHub Actions workflow to run linting, tests, build Docker images, and run docker-compose up -d for smoke tests.
- Documentation and developer experience
  - Flesh out per-component READMEs (backend/pipeline) with local dev setup, API docs, and contribution guidelines.
  - Provide an .env.sample and a script to populate environment-specific values.
- Testing
  - Add unit tests for backend endpoints and data validation; end-to-end tests that mock the pipeline endpoints.
- Data management
  - Add scripts for data seeding and model versioning; consider MLflow registry and sensorized tracking.
