from multi_agents.domain_tools.bio.ontology_multilabel_metrics import (
    close_scores,
    close_truth,
    frequency_baseline,
    item_centric_fmax,
    mean_namespace_fmax,
    protein_centric_fmax,
)


def test_ontology_closure_and_fmax():
    parents = {
        "GO:child": {"GO:parent"},
        "GO:parent": {"GO:root"},
    }
    truth = {"P1": {"GO:child"}}
    scores = {"P1": {"GO:child": 0.9}}

    assert close_truth(truth, parents)["P1"] == {
        "GO:child",
        "GO:parent",
        "GO:root",
    }
    assert close_scores(scores, parents)["P1"]["GO:parent"] == 0.9

    result = item_centric_fmax(truth, scores, parents, threshold_count=11)
    assert result.fmax == 1.0
    assert protein_centric_fmax(truth, scores, parents, threshold_count=11).fmax == 1.0


def test_mean_namespace_fmax_and_frequency_baseline():
    truth = {
        "P1": {"GO:mf1", "GO:bp1"},
        "P2": {"GO:mf1"},
    }
    scores = {
        "P1": {"GO:mf1": 0.8, "GO:bp1": 0.7},
        "P2": {"GO:mf1": 0.9},
    }
    namespaces = {
        "molecular_function": {"GO:mf1"},
        "biological_process": {"GO:bp1"},
    }

    results = mean_namespace_fmax(truth, scores, namespaces, threshold_count=11)
    assert results["mean_fmax"] == 1.0

    baseline = frequency_baseline(truth, ["T1"], top_k=1)
    assert baseline["T1"] == {"GO:mf1": 1.0}
