pipeline:
	docker compose up -d
	python pipelines/train.py
	python pipelines/monitor.py
	python serving/app.py &

deploy:
	ENV=$(ENV) terraform -chdir=infra apply -auto-approve
	docker build -t mlops-serving ./serving
	aws ecr get-login-password | docker login --username AWS ...
	docker push $(ECR_URI)/mlops-serving:latest

test:
	pytest tests/ -v --cov=pipelines

lint:
	ruff check . && mypy pipelines/