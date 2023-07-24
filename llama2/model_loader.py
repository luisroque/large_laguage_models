import os
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from dotenv import load_dotenv

load_dotenv()


class ModelLoader:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.config = AutoConfig.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_auth_token=os.getenv("HUGGINGFACE_TOKEN"),
        )
        self.model = self._load_model()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, use_auth_token=os.getenv("HUGGINGFACE_TOKEN")
        )

    def _load_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            config=self.config,
            trust_remote_code=True,
            load_in_4bit=True,
            device_map="auto",
            use_auth_token=os.getenv("HUGGINGFACE_TOKEN"),
        )
        return model
