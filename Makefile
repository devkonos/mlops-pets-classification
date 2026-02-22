.PHONY: help install install-dev clean test lint format docker-build docker-run docker-compose train mlflow-ui k8s-deploy smoke-test

help:
	@echo "Cats vs Dogs MLOps Pipeline - Available Commands"
	@echo "=================================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install          - Install production dependencies"
	@echo "  make install-dev      - Install development dependencies"
	@echo "  make clean            - Clean temporary files and caches"
	@echo ""
	@echo "Testing & Code Quality:"
	@echo "  make test             - Run unit tests"
	@echo "  make test-cov         - Run tests with coverage"
	@echo "  make lint             - Run code linting (flake8)"
	@echo "  make format           - Format code with black"
	@echo ""
	@echo "Machine Learning:"
	@echo "  make train            - Train model (simple_cnn)"
	@echo "  make train-resnet     - Train model (resnet18)"
	@echo "  make mlflow-ui        - Start MLflow UI on port 5000"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build     - Build Docker image"
	@echo "  make docker-run       - Run Docker container locally"
	@echo "  make docker-compose   - Run full stack with docker-compose"
	@echo ""
	@echo "API:"
	@echo "  make api-server       - Start API server locally"
	@echo "  make api-test         - Test API with curl commands"
	@echo ""
	@echo "Kubernetes:"
	@echo "  make k8s-deploy       - Deploy to Kubernetes"
	@echo "  make k8s-logs         - View pod logs"
	@echo "  make k8s-status       - Check deployment status"
	@echo ""
	@echo "Monitoring:"
	@echo "  make smoke-test       - Run smoke tests"
	@echo "  make dashboard        - Start Grafana dashboard"
	@echo ""

# Setup
install:
	pip install --upgrade pip
	pip install -r requirements.txt

install-dev: install
	pip install pytest pytest-cov flake8 black jupyter notebook

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache/ .coverage htmlcov/ dist/ build/ *.egg-info/
	rm -rf .tox/ .hypothesis/

# Testing & Code Quality
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint:
	flake8 src/ tests/ --max-line-length=100 --statistics

format:
	black src/ tests/

format-check:
	black --check src/ tests/

# Model Training
train:
	python src/models/train.py \
		--data-dir data/splits \
		--model simple_cnn \
		--epochs 20 \
		--batch-size 32 \
		--lr 0.001

train-resnet:
	python src/models/train.py \
		--data-dir data/splits \
		--model resnet18 \
		--epochs 20 \
		--batch-size 32 \
		--lr 0.001

mlflow-ui:
	mlflow ui --host 0.0.0.0 --port 5000

# Docker
docker-build:
	docker build -t cats-dogs-api:latest .

docker-run: docker-build
	docker run -it --rm \
		-p 8000:8000 \
		-v $$(pwd)/models:/app/models \
		-v $$(pwd)/logs:/app/logs \
		cats-dogs-api:latest

docker-stop:
	docker stop $$(docker ps -q --filter "ancestor=cats-dogs-api:latest") 2>/dev/null || true

docker-compose-up:
	docker-compose up -d
	@echo "✅ Services started:"
	@echo "  - API: http://localhost:8000"
	@echo "  - MLflow: http://localhost:5000"
	@echo "  - Prometheus: http://localhost:9090"
	@echo "  - Grafana: http://localhost:3000 (admin/admin)"

docker-compose-down:
	docker-compose down

# API
api-server:
	python -m uvicorn src.api.app:app \
		--host 0.0.0.0 \
		--port 8000 \
		--reload

api-test:
	@echo "Testing API endpoints..."
	curl -s http://localhost:8000/health | jq .
	curl -s http://localhost:8000/info | jq .
	curl -s http://localhost:8000/metrics | head -20

# Kubernetes
k8s-deploy:
	kubectl apply -f k8s/deployment.yaml
	@echo "✅ Deployment applied. Checking status..."
	kubectl Get pods -n mlops

k8s-logs:
	kubectl logs -n mlops -l app=cats-dogs-api -f

k8s-status:
	kubectl get all -n mlops

# Smoke Tests
smoke-test:
	bash scripts/smoke_test.sh http://localhost:8000

smoke-test-k8s:
	bash scripts/smoke_test.sh http://cats-dogs-api.mlops.svc.cluster.local

# DVC
dvc-init:
	dvc init
	dvc remote add -d storage ./data

dvc-pull:
	dvc pull

dvc-push:
	dvc push

# Git
git-setup:
	git init
	git add .
	git commit -m "Initial MLOps pipeline setup"

# Development
dev-setup: install-dev git-setup
	@echo "✅ Development environment ready!"
	@echo "Next steps:"
	@echo "  1. pip install -r requirements.txt"
	@echo "  2. make test (to verify setup)"
	@echo "  3. make train (to train model)"

dev-notebook:
	jupyter notebook notebooks/

# Production
prod-build: lint test docker-build
	@echo "✅ Production build successful!"

prod-deploy: prod-build k8s-deploy smoke-test-k8s
	@echo "✅ Production deployment successful!"

# Cleanup
deep-clean: clean
	rm -rf mlruns/ .dvc/ logs/ models/artifacts/
	docker system prune -f
	@echo "✅ Deep clean completed"
