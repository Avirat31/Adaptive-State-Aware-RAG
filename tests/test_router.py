import pytest

from adaptive_biorag.routing.heuristic_router import FixedRouter, HeuristicRouter, build_router
from adaptive_biorag.routing.learned_router import LearnedRouter, classification_report

R = HeuristicRouter(0.4)


@pytest.mark.parametrize("q", ["What does the gene KRN7 encode?", "What is the target of Zelmab?", "Is NRF9 involved in oxidative stress?"])
def test_simple_questions_single_step(q):
    d = R.route(q)
    assert d.route == "single_step" and d.complexity_score < 0.4 and d.reason_codes


@pytest.mark.parametrize("q", ["Which small molecule inhibits the kinase encoded by the gene associated with Hollenbeck syndrome?",
                               "Does drug A reduce risk, and how do the trial and the cohort study differ?",
                               "Compare BRCA1 versus TP53 mutation effects; what pathways connect them?"])
def test_compound_questions_multi_step(q):
    d = R.route(q)
    assert d.route == "multi_step" and 0 <= d.complexity_score <= 1 and d.features["n_words"] > 0


def test_ambiguous_borderline_and_threshold_configurable():
    q = "How does metformin affect glucose levels?"
    s = R.route(q).complexity_score
    assert 0 < s < 0.4 and R.route(q).route == "single_step"
    assert HeuristicRouter(s).route(q).route == "multi_step" and HeuristicRouter(s + 0.01).route(q).route == "single_step"


def test_fixed_router_and_factory(cfg):
    assert FixedRouter("multi_step").route("anything").reason_codes == ["routing_disabled"]
    cfg["routing"].update(enabled=False, forced_route="multi_step")
    assert build_router(cfg).route("What is X?").route == "multi_step"


def test_learned_router_requires_labels_and_trains():
    with pytest.raises(ValueError):
        LearnedRouter().fit(["a?"], [1])
    qs = ["What is X?", "What does Y encode?", "Which gene is associated with A and what pathway connects it to B?",
          "Compare drug A versus B; how do outcomes differ?"] * 3
    y = [0, 0, 1, 1] * 3
    r = LearnedRouter().fit(qs, y)
    r.tune_threshold(qs, y)
    assert r.route(qs[2]).route == "multi_step" and r.route(qs[0]).method == "learned"
    assert classification_report([0, 1], [0, 1])["accuracy"] == 1.0
