from dataclasses import dataclass, field


@dataclass
class Chunk:
    id: str
    text: str
    doc_id: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Result:
    answer: str
    contexts: list[Chunk]
    llm_calls: int = 0
    model_seconds: float = 0.0
    trace: dict = field(default_factory=dict)
