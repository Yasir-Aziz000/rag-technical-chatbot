.PHONY: install dev run test lint format clean

install:
	pip install -r requirements.txt

dev:
	pip install -r requirements-dev.txt

run:
	streamlit run app.py

test:
	pytest tests/ -v --tb=short

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

lint:
	flake8 src/ app.py tests/ --max-line-length=100

format:
	black src/ app.py tests/ --line-length=100
	isort src/ app.py tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache/ .mypy_cache/ htmlcov/ .coverage

reset-db:
	rm -rf chroma_db/
	@echo "Vector store cleared."
