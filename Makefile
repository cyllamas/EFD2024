# Project Variables
VENV=.venv
PYTHON=$(VENV)/bin/python
PIP=$(VENV)/bin/pip

# Create virtual environment
init:
	@echo "Creating virtual environment..."
	python3 -m venv $(VENV)
	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Setup complete! Run 'source $(VENV)/bin/activate' to activate the environment."

train:
	@echo "Running train.py..."
	$(PYTHON) src/train.py

evaluate:
	@echo "Running evaluate.py..."
	$(PYTHON) src/evaluate.py

.PHONY: predict

predict:
	@echo "Running tests..."
	$(PYTHON) src/predict.py

clean:
	@echo "Cleaning up cache files..."
	rm -rf $(VENV)

help:
	@echo "Makefile commands:"
	@echo "  make init      - Create virtual env and install dependencies"
	@echo "  make train     - Run the training script"
	@echo "  make evaluate  - Run the evaluation script"
	@echo "  make predict   - Run the predict script"
	@echo "  make clean     - Remove cache files and virtual environment"
