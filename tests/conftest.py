import copy
from pathlib import Path

import pytest

from adaptive_biorag.config import load_config
from adaptive_biorag.generation.llm_provider import LLMClientBase, LLMResponse
from adaptive_biorag.schemas import Passage

ROOT = Path(__file__).resolve().parents[1]


class ScriptedLLM(LLMClientBase):
    """Offline stand-in for an LLM: returns queued responses (or raises queued exceptions)."""
    model = "scripted"

    def __init__(self, replies):
        super().__init__()
        self.replies = list(replies)

    def _complete(self, system, user, json_mode):
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return LLMResponse(r, "scripted", 10, 5)


@pytest.fixture
def cfg(tmp_path):
    c = load_config(ROOT / "configs" / "default.yaml")
    c = copy.deepcopy(c)
    c["corpus"]["index_dir"] = str(tmp_path / "idx")
    c["project"]["output_dir"] = str(tmp_path / "out")
    return c


@pytest.fixture
def passages():
    def P(i, title, text):
        return Passage(f"{i}::p0", i, text, title, "t")
    return [P("A", "Kinase one", "Gene KRN7 encodes kernase-7 a serine kinase involved in the cell cycle."),
            P("B", "Drug valtrexin", "Valtrexin inhibits kernase-7 in biochemical assays."),
            P("C", "Migraine", "Tavorin reduced migraine days in a randomized trial."),
            P("D", "Psoriasis", "Zelmab cleared plaque psoriasis in many patients at week sixteen.")]
