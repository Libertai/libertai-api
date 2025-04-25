import json
import os
from typing import Dict, List

from dotenv import load_dotenv


class ServerConfig:
    type: str
    url: str
    weight: int
    gpu: bool
    completion_paths: str
    api_type: str
    prompt_format: str

    def __init__(self, type: str, url: str, weight: int, gpu: bool, completion_paths: str, api_type: str, prompt_format):
        self.type = type
        self.url = url
        self.completion_paths = completion_paths
        self.api_type = api_type
        self.weight = weight
        self.gpu = gpu
        self.prompt_format = prompt_format


class _Config:
    BACKEND_API_URL: str | None
    BACKEND_SECRET_TOKEN: str | None
    MODELS: Dict[str, List[ServerConfig]]
    REPORT_USAGE: bool
    FORWARD_AUTH: bool

    def __init__(self):
        load_dotenv()

        self.BACKEND_API_URL = os.getenv("BACKEND_API_URL")
        self.BACKEND_SECRET_TOKEN = os.getenv("BACKEND_SECRET_TOKEN")
        self.REPORT_USAGE = bool(os.getenv("REPORT_USAGE", True))
        self.FORWARD_AUTH = bool(os.getenv("FORWARD_AUTH", True))

        # Load models configuration from environment variable or file
        models_config = os.getenv("MODELS_CONFIG")
        self.MODELS = {}

        if models_config:
            try:
                with open(models_config) as f:
                    models_data = json.load(f)
                    print(models_data)
                    for model_name, servers in models_data.items():
                        self.MODELS[model_name.lower()] = [
                            ServerConfig(
                                type=server.get("type"),
                                url=server.get("url"),
                                weight=server.get("weight", 1),
                                gpu=server.get("gpu", False),
                                completion_paths=server.get("completion_paths"),
                                api_type=server.get("api_type"),
                                prompt_format=server.get("prompt_format")
                            )
                            for server in servers
                        ]
            except json.JSONDecodeError as error:
                print(f"Error on {models_config} file")
                print(error)


config = _Config()
