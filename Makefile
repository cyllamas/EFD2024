VENV=.venv
PYTHON=$(VENV)/bin/python
PIP=$(VENV)/bin/pip
DVC=.dvc

init:
	@echo "Creating virtual environment..."
	python3 -m venv $(VENV)
	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Dependencies installed. Setup complete!"
	@echo "To activate the virtual environment, run: source $(VENV)/bin/activate"

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

predict_api:
	@echo "Running predict_api.py for predictions via API..."
	$(PYTHON) src/predict_api.py

dvc:
	@echo "Initializing DVC..."
	$(VENV)/bin/dvc init 

mlflow:
	@echo "Starting MLflow server in the background..."
	@echo "MLflow server is now running at http://127.0.0.1:8080"
	@echo "To execute another command, open a new terminal window."
	@echo "To stop the server, press Ctrl + C in this terminal to terminate the process."
	@$(PYTHON) -m mlflow server --host 127.0.0.1 --port 8080 

clean:
	@echo "Cleaning up cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf $(VENV)
	rm -rf $(DVC)
	rm -rf .pytest_cache

help:
	@echo "Makefile commands:"
	@echo "  make init      	- Create virtual env and install dependencies"
	@echo "  make train     	- Run the training script"
	@echo "  make evaluate 	 	- Run the evaluation script"
	@echo "  make predict   	- Run the predict script"
	@echo "  make clean     	- Remove cache files and virtual environment"
	@echo "  make predict_api 	- Run the API prediction script"
	@echo "  make dvc         	- Initialize DVC environment"
