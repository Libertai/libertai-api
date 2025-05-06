import json
import os

from dotenv import load_dotenv
from pydantic import BaseModel


class ServerConfig(BaseModel):
    url: str
    weight: int = 1
    gpu: bool = False
    completion_paths: list[str]


class _Config:
    BACKEND_API_URL: str
    BACKEND_SECRET_TOKEN: str
    MODELS: dict[str, list[ServerConfig]]

    def __init__(self):
        load_dotenv()

        self.BACKEND_API_URL = os.getenv("BACKEND_API_URL")
        self.BACKEND_SECRET_TOKEN = os.getenv("BACKEND_SECRET_TOKEN")

        # Load models configuration from environment variable or file
        models_config = os.getenv("MODELS_CONFIG")
        self.MODELS = {}

        if models_config:
            try:
                with open(models_config) as f:
                    models_data = json.load(f)
                    for model_name, servers in models_data.items():
                        self.MODELS[model_name.lower()] = [ServerConfig(**server) for server in servers]
            except json.JSONDecodeError as error:
                print(f"Error on {models_config} file")
                print(error)


config = _Config()
