import numpy as np
import pytest

from cellmap.conformal import (
    aps_scores,
    calibrate,
    conformal_quantile,
    fit_linear_probe,
    knn_proba,
    lac_scores,
    prediction_sets,
    probe_proba,
    set_metrics,
    true_label_scores,
)


def blobs(n: int, seed: int, shift: float = 0.0):
    """Three overlapping Gaussian classes in 5 dimensions; `shift` moves every cell."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    centers = np.array([[0, 0, 0, 0, 0], [1.5, 0, 0, 0, 0], [0, 1.5, 0, 0, 0]], float)
    return centers[y] + rng.normal(0, 1, (n, 5)) + shift, y


def test_quantile_matches_the_definition():
    scores = np.arange(1, 11) / 10  # 0.1 .. 1.0, n = 10
    # ceil(11 * 0.9) = 10 -> the 10th smallest
    assert conformal_quantile(scores, 0.1) == 1.0
    # ceil(11 * 0.8) = 9
    assert conformal_quantile(scores, 0.2) == 0.9
    # too few cells for 95 percent: ceil(11 * 0.95) = 11 > 10
    assert conformal_quantile(scores, 0.05) == float("inf")
    with pytest.raises(ValueError):
        conformal_quantile(scores, 1.0)


def test_aps_scores_are_cumulative_mass_with_randomization():
    probs = np.array([[0.6, 0.3, 0.1]])

    class Fixed:
        def uniform(self, size):
            return np.full(size, 0.5)

    s = aps_scores(probs, Fixed())
    np.testing.assert_allclose(s, [[0.3, 0.6 + 0.15, 0.9 + 0.05]])
    # with U = 1 the score of the last label is the total mass
    s1 = aps_scores(probs, type("One", (), {"uniform": lambda self, size: np.ones(size)})())
    np.testing.assert_allclose(s1[0, 2], 1.0)


def test_aps_sets_are_prefixes_in_probability_order():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(6), size=500)
    sets = prediction_sets(aps_scores(probs, rng), np.full(6, 0.7))
    order = np.argsort(-probs, axis=1)
    ranked = np.take_along_axis(sets, order, axis=1)
    # once a label is excluded, every less probable label is excluded too
    assert not np.any(~ranked[:, :-1] & ranked[:, 1:])


@pytest.mark.parametrize("score", [lac_scores, aps_scores])
def test_marginal_coverage_holds_on_exchangeable_data(score):
    """Average over 200 repeats: coverage within 1 point of 90 percent, as the theory says."""
    x_tr, y_tr = blobs(600, 0)
    model = fit_linear_probe(x_tr, y_tr)
    classes = np.arange(3)
    coverages = []
    for rep in range(200):
        x, y = blobs(400, 1000 + rep)
        p = probe_proba(model, x, classes)
        s = score(p, np.random.default_rng(rep))
        q = calibrate(s[:200], y[:200], alpha=0.1)
        sets = prediction_sets(s[200:], q)
        coverages.append(set_metrics(sets, y[200:])["coverage"])
    assert abs(np.mean(coverages) - 0.9) < 0.01


def test_coverage_breaks_when_test_cells_are_shifted():
    """The guarantee needs exchangeability: shift the test cells and it fails."""
    x_tr, y_tr = blobs(600, 0)
    model = fit_linear_probe(x_tr, y_tr)
    classes = np.arange(3)
    x_cal, y_cal = blobs(500, 1)
    x_te, y_te = blobs(500, 2, shift=1.5)
    q = calibrate(lac_scores(probe_proba(model, x_cal, classes)), y_cal, alpha=0.1)
    sets = prediction_sets(lac_scores(probe_proba(model, x_te, classes)), q)
    assert set_metrics(sets, y_te)["coverage"] < 0.85


def test_mondrian_covers_every_class():
    x_tr, y_tr = blobs(600, 0)
    model = fit_linear_probe(x_tr, y_tr)
    classes = np.arange(3)
    worst = []
    for rep in range(50):
        x, y = blobs(1500, 2000 + rep)
        s = lac_scores(probe_proba(model, x, classes))
        q = calibrate(s[:750], y[:750], alpha=0.1, mondrian=True)
        worst.append(set_metrics(prediction_sets(s[750:], q), y[750:])["worst_class_coverage"])
    assert np.mean(worst) > 0.87


def test_label_without_calibration_cells_is_always_included():
    s = np.random.default_rng(0).uniform(size=(30, 3))
    q = calibrate(s, np.zeros(30, int), alpha=0.1, mondrian=True)
    assert np.isinf(q[1]) and np.isinf(q[2])


def test_knn_proba_counts_neighbours():
    train = np.array([[0.0], [0.1], [0.2], [5.0]])
    labels = np.array(["a", "a", "b", "b"])
    p = knn_proba(train, labels, np.array([[0.05]]), np.array(["a", "b"]), k=3)
    np.testing.assert_allclose(p, [[2 / 3, 1 / 3]])


def test_probe_proba_puts_unseen_classes_at_zero():
    x, y = blobs(300, 0)
    model = fit_linear_probe(x, y)
    p = probe_proba(model, x[:5], np.array([0, 1, 2, 3]))
    assert p.shape == (5, 4)
    np.testing.assert_allclose(p[:, 3], 0.0)
    np.testing.assert_allclose(p.sum(1), 1.0)


def test_set_metrics_handles_unknown_labels():
    sets = np.array([[True, False], [True, True], [False, False]])
    m = set_metrics(sets, np.array([0, 1, -1]), min_class_cells=1)
    assert m["coverage"] == 1.0 and m["mean_size"] == 1.5
    assert true_label_scores(np.array([[0.1, 0.9]]), np.array([1]))[0] == 0.9
