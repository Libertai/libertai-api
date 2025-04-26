import json
import os
from typing import Dict, List

from dotenv import load_dotenv


class ServerConfig:
    url: str
    weight: int
    gpu: bool
    completion_paths: str

    def __init__(self, url: str, weight: int, gpu: bool, completion_paths: str):
        self.url = url
        self.completion_paths = completion_paths
        self.weight = weight
        self.gpu = gpu


class _Config:
    BACKEND_API_URL: str
    BACKEND_SECRET_TOKEN: str
    MODELS: Dict[str, List[ServerConfig]]
    REPORT_USAGE: bool
    FORWARD_AUTH: bool

    def __init__(self):
        load_dotenv()

        self.BACKEND_API_URL = os.getenv("BACKEND_API_URL")
        self.BACKEND_SECRET_TOKEN = os.getenv("BACKEND_SECRET_TOKEN")
        self.REPORT_USAGE = os.getenv("REPORT_USAGE", "True").lower() == "true"
        self.FORWARD_AUTH = os.getenv("FORWARD_AUTH", "True").lower() == "true"

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
                                url=server.get("url"),
                                weight=server.get("weight", 1),
                                gpu=server.get("gpu", False),
                                completion_paths=server.get("completion_paths"),
                            )
                            for server in servers
                        ]
            except json.JSONDecodeError as error:
                print(f"Error on {models_config} file")
                print(error)


config = _Config()
