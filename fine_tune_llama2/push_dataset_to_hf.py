from fine_tune_llama2 import (
    load_dataset_from_local,
    publish_to_hugging_face,
    Config
)


def push_dataset():
    transformed_dataset = load_dataset_from_local()

    # publish_to_hugging_face(transformed_dataset, Config.NEW_DATASET_NAME, top=20000)
    publish_to_hugging_face(transformed_dataset, Config.NEW_DATASET_NAME_COMPLETE)


if __name__ == "__main__":
    push_dataset()
