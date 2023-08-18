from fine_tune_llama2 import (
    load_data,
    initialize_model_and_tokenizer,
    fine_tune_and_save_model,
    visualize_and_save,
)


def fine_tune():
    transformed_dataset = load_data()
    model, tokenizer = initialize_model_and_tokenizer()
    _, metrics = fine_tune_and_save_model(
        model, tokenizer, transformed_dataset["train"], transformed_dataset["test"]
    )
    exec_time = metrics["exec_time"]
    memory_usage = metrics["peak_mem_consumption"]
    visualize_and_save(exec_time, memory_usage)


if __name__ == "__main__":
    fine_tune()
