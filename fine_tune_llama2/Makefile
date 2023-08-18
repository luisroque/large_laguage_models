.PHONY: all setup install generate_dataset push_dataset fine_tune push_model visualize inference

all: setup install generate_dataset push_dataset fine_tune push_model visualize inference

setup:
	@echo "Setting up the conda environment..."
	conda create -n fine_tune_llama2 python=3.10

install:
	@echo "Installing required packages..."
	python -m pip install -r requirements.txt

generate_dataset:
	@echo "Generating new dataset..."
	python generate_dataset.py

push_dataset:
	@echo "Pushing dataset to Hugging Face..."
	python push_dataset_to_hf.py

fine_tune:
	@echo "Fine-tuning and saving the model..."
	python fine_tune.py

merge_models:
	@echo "Running merge..."
	python merge_models.py

push_model:
	@echo "Pushing model to Hugging Face..."
	python push_model_to_hf.py

new_model_inference:
	@echo "Running inference with the new model..."
	python inference.py new_model

llama2_inference:
	@echo "Running inference with the Llama2 model..."
	python inference.py llama2
