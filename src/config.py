from dataclasses import dataclass

@dataclass
class ModelConfig:
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 100
    max_depth: int = 10
