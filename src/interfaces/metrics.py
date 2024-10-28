from typing import Optional

from pydantic import BaseModel


class Metric(BaseModel):
    owner: str
    key: str
    ts: int
    count: int


class Metrics(BaseModel):
    values: list[Metric]
    owner: Optional[str]
