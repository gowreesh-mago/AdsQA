"""
Model Loader for AdsQA Evaluation Pipeline
Supports loading video inference models and evaluator models based on config
"""

import yaml
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen3VLForConditionalGeneration,
    AutoProcessor
)


class ModelLoader:
    """Load models based on configuration file"""

    def __init__(self, config_path="evaluation/config.yaml"):
        """
        Initialize ModelLoader with configuration

        Args:
            config_path: Path to YAML configuration file
        """
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

    def load_video_model(self, model_type="qwen3_vl_8b"):
        """
        Load video inference model

        Args:
            model_type: Type of video model (qwen3_vl_8b, qwen2_5_vl_7b)

        Returns:
            Tuple of (model, processor, config)
        """
        config = self.config["models"]["video_inference"][model_type]

        if model_type == "qwen3_vl_8b":
            print(f"Loading Qwen3-VL-8B-Instruct from {config['model_path']}...")
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                config["model_path"],
                torch_dtype="auto",
                device_map="auto"
            )
            processor = AutoProcessor.from_pretrained(config["model_path"])
            print("Qwen3-VL-8B-Instruct loaded successfully!")
        else:
            raise ValueError(f"Unknown video model type: {model_type}")

        return model, processor, config

    def load_evaluator_model(self, model_type="qwen2_5_7b_instruct"):
        """
        Load evaluation model

        Args:
            model_type: Type of evaluator model (qwen2_5_7b_instruct, gpt4o)

        Returns:
            Tuple of (model, tokenizer, config)
            For gpt4o, returns (None, None, config)
        """
        config = self.config["models"]["evaluator"][model_type]

        if model_type == "gpt4o":
            # GPT-4o uses OpenAI API, no local model
            return None, None, config

        if model_type == "qwen2_5_7b_instruct":
            print(f"Loading Qwen2.5-7B-Instruct from {config['model_path']}...")
            model = AutoModelForCausalLM.from_pretrained(
                config["model_path"],
                torch_dtype=torch.float16,
                device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(config["model_path"])
            model.eval()
            print("Qwen2.5-7B-Instruct loaded successfully!")
            return model, tokenizer, config
        else:
            raise ValueError(f"Unknown evaluator model type: {model_type}")
