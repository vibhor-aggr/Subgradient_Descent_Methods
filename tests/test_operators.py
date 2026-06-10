import numpy as np

from subgrad_methods.operators import project_l1_ball, project_l2_ball, soft_threshold


def test_soft_threshold_matches_hand_computed_values():
    values = np.array([-3.0, -0.5, 0.25, 2.0])
    actual = soft_threshold(values, 1.0)
    expected = np.array([-2.0, -0.0, 0.0, 1.0])
    np.testing.assert_allclose(actual, expected)


def test_l1_projection_keeps_feasible_vector_unchanged():
    values = np.array([0.25, -0.25, 0.5])
    projected = project_l1_ball(values, 2.0)
    np.testing.assert_allclose(projected, values)


def test_l1_projection_respects_radius():
    values = np.array([3.0, -1.0, 0.5])
    projected = project_l1_ball(values, 2.0)
    assert np.sum(np.abs(projected)) <= 2.0 + 1e-10


def test_l2_projection_respects_radius():
    values = np.array([3.0, 4.0])
    projected = project_l2_ball(values, 2.0)
    assert np.linalg.norm(projected) <= 2.0 + 1e-10
    np.testing.assert_allclose(projected, np.array([1.2, 1.6]))
