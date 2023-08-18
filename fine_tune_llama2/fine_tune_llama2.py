import os
import torch
import time
import functools
from datasets import load_dataset
from transformers import (
    TrainingArguments,
    pipeline,
    BitsAndBytesConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    EarlyStoppingCallback,
)
from datasets import Dataset
import pandas as pd
from bs4 import BeautifulSoup
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import matplotlib.pyplot as plt
import pickle


class Config:
    MODEL_NAME = "meta-llama/Llama-2-7b-hf"
    OUTPUT_DIR = "./results"
    NEW_DATASET_NAME_COMPLETE = "luisroque/instruct-python-500k"
    NEW_DATASET_NAME = "luisroque/instruct-python-llama2-20k"
    NEW_DATASET_NAME_LOCAL = "instruct-python-500k.pkl"
    NEW_MODEL_PATH = "./Llama-2-7b-minipython-instruct"
    NEW_MODEL_PATH_MERGE = "./Llama-2-7b-minipython-instruct-merge"
    NEW_MODEL_NAME = "Llama-2-7b-minipython-instruct"
    HF_HUB_MODEL_NAME = "luisroque/Llama-2-7b-minipython-instruct"
    SYSTEM_MESSAGE = "Given a puzzle-like code question, provide a well-reasoned, step-by-step Python solution."
    NUM_EPOCHS = 1
    BATCH_SIZE = 2
    GRAD_ACC_STEPS = 1
    SAVE_STEPS = 50
    LOG_STEPS = 5
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 0.001
    MAX_GRAD_NORM = 0.3
    SCHEDULER_TYPE = "cosine"
    PER_DEVICE_TRAIN_BATCH_SIZE = 4
    PER_DEVICE_EVAL_BATCH_SIZE = 4
    OPTIM = "paged_adamw_32bit"
    FP16 = False
    BF16 = False
    MAX_STEPS = 1000
    WARMUP_RATIO = 0.03
    GROUP_BY_LENGTH = 3
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.1
    LORA_R = 64
    DEVICE_MAP = {"": 0}
    USE_4BIT = True
    BNB_4BIT_COMPUTE_DTYPE = "float16"
    BNB_4BIT_COMPUTE_QUANT_TYPE = "nf4"
    USE_NESTED_QUANT = False


def time_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result, metrics = func(*args, **kwargs)
        end_time = time.time()
        exec_time = end_time - start_time
        metrics["exec_time"] = exec_time
        return result, metrics

    return wrapper


def memory_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        result, metrics = func(*args, **kwargs)
        peak_mem = torch.cuda.max_memory_allocated()
        peak_mem_consumption = peak_mem / 1e9
        metrics["peak_mem_consumption"] = peak_mem_consumption
        return result, metrics

    return wrapper


def visualize_and_save(exec_time, memory_usage):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel("Run")
    ax1.set_ylabel("Execution Time (seconds)", color="tab:blue")
    ax1.plot(exec_time, color="tab:blue", marker="o", label="Execution Time")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()

    ax2.set_ylabel("Memory Consumption (GB)", color="tab:red")
    ax2.plot(memory_usage, color="tab:red", marker="o", label="Memory Consumption")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    fig.tight_layout()
    plt.title("Execution Time and Memory Consumption Over a Run")

    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/exec_time_and_memory.png")
    plt.close()


def load_data():
    """Load the new dataset."""
    dataset = load_dataset(Config.NEW_DATASET_NAME)
    return dataset


def load_data_to_fine_tune():
    """Load the dataset and filter for Python language."""
    dtypes_questions = {"Id": "int32", "Score": "int16", "Title": "str", "Body": "str"}
    df_questions = pd.read_csv(
        "Questions.csv",
        usecols=["Id", "Score", "Title", "Body"],
        encoding="ISO-8859-1",
        dtype=dtypes_questions,
    )

    dtypes_answers = {
        "Id": "int32",
        "ParentId": "int32",
        "Score": "int16",
        "Body": "str",
    }
    df_answers = pd.read_csv(
        "Answers.csv",
        usecols=["Id", "ParentId", "Score", "Body"],
        encoding="ISO-8859-1",
        dtype=dtypes_answers,
    )

    merged = pd.merge(
        df_questions, df_answers, left_on="Id", right_on="ParentId", how="inner"
    )
    # Sort by score of the answer in descending order and drop duplicates based on question ID
    merged = merged.sort_values(by="Score_y", ascending=False).drop_duplicates(
        subset="Id_x", keep="first"
    )

    # Remove HTML tags using BeautifulSoup
    merged["Body_x"] = merged["Body_x"].apply(
        lambda x: BeautifulSoup(x, "lxml").get_text()
    )
    merged["Body_y"] = merged["Body_y"].apply(
        lambda x: BeautifulSoup(x, "lxml").get_text()
    )

    merged["combined_question"] = merged["Title"] + ": " + merged["Body_x"]

    # Rename and select the desired columns
    final_df = merged[["Score_x", "Score_y", "combined_question", "Body_y"]]
    final_df.columns = ["score_question", "score_answer", "question", "answer"]

    final_df = final_df[
        (final_df["score_question"] >= 0) & (final_df["score_answer"] >= 0)
    ]

    # Contains code that resembles python code
    final_df = final_df[
        final_df["question"].apply(contains_code)
        | final_df["answer"].apply(contains_code)
    ]

    return final_df


def contains_code(text):
    python_keywords = [
        "def",
        "class",
        "import",
        "print",
        "return",
        "for",
        "while",
        "if",
        "else",
        "elif",
        "try",
        "except",
        "lambda",
        "list",
        "dict",
        "set",
        "str",
        "=",
        "{",
        "}",
        "(",
        ")",
    ]

    for keyword in python_keywords:
        if keyword in text:
            return True
    return False


def store_dataset_locally(dataset):
    """Store the dataset locally using pickle."""
    with open(Config.NEW_DATASET_NAME_LOCAL, "wb") as file:
        pickle.dump(dataset, file)


def load_dataset_from_local():
    """Load the dataset from a local directory using pickle."""
    with open(Config.NEW_DATASET_NAME_LOCAL, "rb") as file:
        dataset = pickle.load(file)
    return dataset


def transform_dataset_format(df):
    """Transform the dataframe into a specified format."""

    def transform(row):
        user_text = row["question"]
        assistant_text = row["answer"]
        return f"<s>[INST] <</SYS>>\n{Config.SYSTEM_MESSAGE.strip()}\n<</SYS>>\n\n" \
               f"{user_text} [/INST] {assistant_text} </s>"

    transformed_df = df.apply(transform, axis=1).to_frame(name="text")
    transformed_df.reset_index(drop=True, inplace=True)

    return transformed_df


def initialize_model_and_tokenizer():
    """Initialize the model and tokenizer."""

    compute_dtype = getattr(torch, Config.BNB_4BIT_COMPUTE_DTYPE)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=Config.USE_4BIT,
        bnb_4bit_quant_type=Config.BNB_4BIT_COMPUTE_QUANT_TYPE,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=Config.USE_NESTED_QUANT,
    )
    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_NAME, quantization_config=bnb_config, device_map=Config.DEVICE_MAP
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


def configure_training_args():
    """Configure training arguments."""
    return TrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        num_train_epochs=Config.NUM_EPOCHS,
        per_device_train_batch_size=Config.PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=Config.GRAD_ACC_STEPS,
        optim=Config.OPTIM,
        save_steps=Config.SAVE_STEPS,
        logging_steps=Config.LOG_STEPS,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        fp16=Config.FP16,
        bf16=Config.BF16,
        max_grad_norm=Config.MAX_GRAD_NORM,
        max_steps=Config.MAX_STEPS,
        warmup_ratio=Config.WARMUP_RATIO,
        group_by_length=Config.GROUP_BY_LENGTH,
        lr_scheduler_type=Config.SCHEDULER_TYPE,
        report_to="all",
        evaluation_strategy="steps",
        eval_steps=50,
        load_best_model_at_end=True,
    )


@memory_decorator
@time_decorator
def fine_tune_and_save_model(model, tokenizer, train_dataset, val_dataset):
    """Fine-tune the model and save it."""

    peft_config = LoraConfig(
        lora_alpha=Config.LORA_ALPHA,
        lora_dropout=Config.LORA_DROPOUT,
        r=Config.LORA_R,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()

    training_args = configure_training_args()

    early_stopping = EarlyStoppingCallback(early_stopping_patience=4)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        peft_config=peft_config,
        tokenizer=tokenizer,
        args=training_args,
        max_seq_length=512,
        callbacks=[early_stopping],
    )
    trainer.train()

    if not os.path.exists(Config.NEW_MODEL_PATH):
        os.makedirs(Config.NEW_MODEL_PATH)

    trainer.model.save_pretrained(Config.NEW_MODEL_PATH)
    tokenizer.save_pretrained(Config.NEW_MODEL_PATH)

    del model
    torch.cuda.empty_cache()

    return None, {}


def generate_code_from_prompt(model, tokenizer):
    """Generate code based on the provided system message using a pre-trained model and tokenizer."""
    prompt = (
        f"[INST] <<SYS>>\n{Config.SYSTEM_MESSAGE}\n<</SYS>>\n\n"
        f"Write a function that reverses a linked list. [/INST]"
    )

    pipe = pipeline(
        task="text-generation", model=model, tokenizer=tokenizer, max_length=500
    )

    result = pipe(prompt)
    generated_text = result[0]["generated_text"]

    return generated_text


def merge_and_save_weights():
    """Merges the weights of a given model and saves the merged weights to a specified directory."""

    if not os.path.exists(Config.NEW_MODEL_PATH_MERGE):
        os.makedirs(Config.NEW_MODEL_PATH_MERGE)

    base_model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_NAME,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=Config.DEVICE_MAP,
    )
    model = PeftModel.from_pretrained(base_model, Config.NEW_MODEL_NAME)
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model.save_pretrained(Config.NEW_MODEL_PATH)
    tokenizer.save_pretrained(Config.NEW_MODEL_PATH)


def publish_to_hugging_face(df, dataset_name, top=None):
    """Publish the transformed dataframe to Hugging Face datasets."""
    dataset = Dataset.from_pandas(df)

    if top is not None:
        dataset = dataset.select(range(top))
        splits = dataset.train_test_split(test_size=1000, shuffle=True)
    else:
        splits = dataset

    splits.push_to_hub(dataset_name)


def print_trainable_parameters(model):
    """Prints the number of trainable parameters in the model."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable%: {100 * trainable_params / all_param}"
    )


def push_model_to_hub():
    """Push the fine-tuned model and tokenizer to the Hugging Face Hub."""
    model = AutoModelForCausalLM.from_pretrained(Config.NEW_MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(Config.NEW_MODEL_PATH)

    model.push_to_hub(Config.HF_HUB_MODEL_NAME, use_temp_dir=False)
    tokenizer.push_to_hub(Config.HF_HUB_MODEL_NAME, use_temp_dir=False)
