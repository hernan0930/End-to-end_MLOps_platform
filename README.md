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
     → MLflow Registry → GitLab CI quality gate
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

## Run locally in 3 commands

```bash
git clone https://github.com/you/mlops-platform
docker compose up -d    # starts MLflow, MinIO, Grafana, Prefect
make pipeline           # trains model, registers, deploys locally
```

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