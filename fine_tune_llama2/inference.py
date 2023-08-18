from transformers import (
    pipeline,
    AutoTokenizer,
)
from fine_tune_llama2 import Config
import argparse


def generate_response(model_name, tokenizer, prompt, max_length=600):
    """Generate a response using the specified model."""
    pipe = pipeline(
        task="text-generation",
        model=model_name,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    result = pipe(f"{prompt}")
    return result[0]["generated_text"]


def main(model_to_run):
    prompt = (
        f"[INST] <<SYS>>\n{Config.SYSTEM_MESSAGE}\n<</SYS>>\n\n"
        f"Write a function that reverses a linked list. [/INST]"
    )

    if model_to_run == "new_model":
        new_tokenizer = AutoTokenizer.from_pretrained(Config.HF_HUB_MODEL_NAME)
        new_model_response = generate_response(
            Config.HF_HUB_MODEL_NAME, new_tokenizer, prompt
        )
        print("Response from new model:")
        print(new_model_response)
    else:
        llama_model_name = Config.MODEL_NAME
        llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
        llama_model_response = generate_response(
            llama_model_name, llama_tokenizer, prompt
        )

        print("\nResponse from Llama2 base model:")
        print(llama_model_response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run different models.")
    parser.add_argument(
        "model_to_run", type=str, help='Which model to run: "new_model" or "llama2"'
    )
    args = parser.parse_args()

    main(args.model_to_run)
