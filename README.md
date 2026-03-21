# MLOps Platform — end-to-end ML lifecycle for AWS

![CI](https://img.shields.io/badge/CI-passing-brightgreen)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![AWS Ready](https://img.shields.io/badge/AWS-ready-orange)
![LocalStack](https://img.shields.io/badge/LocalStack-compatible-brightgreen)
![MLflow](https://img.shields.io/badge/MLflow-2.x-blue)

Full MLOps pipeline covering training, versioning, CI/CD deployment, drift monitoring, and automated retraining — runnable locally with Docker Compose, deployable to AWS with a single `terraform apply`.

---

## Architecture

```
Data → S3 (MinIO local) → SageMaker Pipeline (local: sklearn pipeline)
     → MLflow Registry → Github/Gitlab → CI quality gate
     → SageMaker Endpoint (local: FastAPI)
     → CloudWatch (local: Prometheus + Grafana)
     → Model Monitor (local: Evidently)
     → Step Functions (local: Prefect)
```

---

## What this project covers

- Reproducible training with DVC data versioning and MLflow experiment tracking
- Automated CI/CD: model quality gate blocks promotion if accuracy drops below threshold
- Canary deployment: 10% traffic to new model, auto-rollback on SLA breach
- Data drift detection with Evidently AI + automated retraining trigger
- Full IaC: every resource defined in Terraform (no manual console clicks)
- FinOps annotations: spot instance config, cost tags, right-sizing notes

---

## Local setup

**Prerequisites:** Docker Desktop, Python 3.11, `make`

```bash
# 1. Clone and configure environment
git clone https://github.com/hernan0930/End-to-end_MLOps_platform
cd End-to-end_MLOps_platform
cp .env.example .env          # edit values if needed

# 2. Start the full local stack
make up
# Waits for Postgres to be healthy, creates MinIO buckets, then starts MLflow

# 3. Verify services
#   MLflow UI      → http://localhost:5000
#   MinIO console  → http://localhost:9001  (user: minioadmin / minioadmin)
#   Prefect UI     → http://localhost:4200
#   Grafana        → http://localhost:3000
#   Prometheus     → http://localhost:9090

# 4. Run the training pipeline
make pipeline

# 5. Tear down
make down
```

> **Note:** On first run `minio-init` automatically creates the `mlflow-artifacts` and `drift-reports` buckets before MLflow starts.

---

## Deploy to AWS

```bash
cp .env.example .env.aws            # fill in your AWS credentials
terraform -chdir=infra apply        # provisions all resources
make pipeline ENV=aws               # same pipeline, now on SageMaker + S3
```

---

## What I'd do differently in production

- Use **SageMaker Feature Store** instead of Redis for feature serving at scale
- Replace **Prefect** with **AWS Step Functions** for tighter IAM integration
- Add **A/B testing** at the inference layer with SageMaker routing rules