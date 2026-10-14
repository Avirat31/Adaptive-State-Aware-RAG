from .base import DatasetAdapter, DatasetError, validate_examples
from .bioasq import BioASQAdapter
from .jsonl_adapter import JsonlAdapter
from .medhopqa import MedHopQAAdapter
from .mirage import MirageAdapter
from .pubmedqa import PubMedQAAdapter

_REGISTRY = {"synthetic": JsonlAdapter, "jsonl": JsonlAdapter, "pubmedqa": PubMedQAAdapter, "bioasq": BioASQAdapter,
             "mirage": MirageAdapter, "medhopqa": MedHopQAAdapter}


def get_adapter(cfg: dict) -> DatasetAdapter:
    return _REGISTRY[cfg["dataset"]["name"]](cfg)


__all__ = ["get_adapter", "validate_examples", "DatasetError", "DatasetAdapter"]
