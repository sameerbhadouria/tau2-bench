"""
Tests for weighted reward evaluation.
"""
import json
from pathlib import Path

import pytest

from tau2.data_model.tasks import Task
from tau2.evaluator.evaluator import calculate_reward_weights


def test_calculate_reward_weights_basic():
    """Test basic weight calculation."""
    # Load a real task
    tasks_path = Path("data/tau2/domains/airline/tasks.json")
    if not tasks_path.exists():
        pytest.skip("Airline tasks data not available")

    with open(tasks_path) as f:
        task_data = json.load(f)

    # Test task 3 which has both DB and COMMUNICATE
    task = Task(**task_data[3])
    weights = calculate_reward_weights(task)

    # Should have weights for active components
    assert len(weights) > 0

    # Weights should sum to 1.0
    total = sum(weights.values())
    assert abs(total - 1.0) < 1e-6, f"Weights don't sum to 1: {total}"

    # All weights should be positive
    for component, weight in weights.items():
        assert weight > 0, f"{component} has non-positive weight: {weight}"


def test_calculate_reward_weights_normalization():
    """Test that weights are properly normalized."""
    tasks_path = Path("data/tau2/domains/airline/tasks.json")
    if not tasks_path.exists():
        pytest.skip("Airline tasks data not available")

    with open(tasks_path) as f:
        task_data = json.load(f)

    # Test multiple tasks
    for i in range(min(10, len(task_data))):
        task = Task(**task_data[i])
        weights = calculate_reward_weights(task)

        if weights:  # Only check if there are weights
            total = sum(weights.values())
            assert abs(total - 1.0) < 1e-6, f"Task {i}: weights don't sum to 1: {total}"


def test_calculate_reward_weights_distribution():
    """Test that weight distribution makes sense."""
    tasks_path = Path("data/tau2/domains/airline/tasks.json")
    if not tasks_path.exists():
        pytest.skip("Airline tasks data not available")

    with open(tasks_path) as f:
        task_data = json.load(f)

    # Collect statistics
    weight_distributions = []

    for task_dict in task_data[:10]:
        task = Task(**task_dict)
        weights = calculate_reward_weights(task)
        if weights:
            weight_distributions.append(weights)

    # At least some tasks should have weights
    assert len(weight_distributions) > 0, "No tasks had weights calculated"

    print(f"\nAnalyzed {len(weight_distributions)} tasks with weights")

    # Show some examples
    for i, weights in enumerate(weight_distributions[:5]):
        print(f"\nTask {i} weights:")
        for component, weight in sorted(weights.items()):
            print(f"  {component}: {weight:.4f} ({weight*100:.1f}%)")


def test_weighted_vs_binary_scenario():
    """
    Test that demonstrates the difference between weighted and binary rewards.
    This is a conceptual test showing the benefit of weighted rewards.
    """
    # Scenario: Task with 3 actions (60% weight) and 2 communicate checks (40% weight)
    action_weight = 0.6
    communicate_weight = 0.4

    # Case 1: Agent completes actions but misses communication
    action_success = 1.0
    communicate_success = 0.0

    # Binary reward (multiplicative)
    binary_reward = action_success * communicate_success
    assert binary_reward == 0.0, "Binary gives no credit"

    # Weighted reward (additive)
    weighted_reward = (action_weight * action_success +
                      communicate_weight * communicate_success)
    assert weighted_reward == 0.6, "Weighted gives partial credit"

    # Case 2: Agent completes everything
    action_success = 1.0
    communicate_success = 1.0

    binary_reward = action_success * communicate_success
    weighted_reward = (action_weight * action_success +
                      communicate_weight * communicate_success)

    # Both should give full credit
    assert binary_reward == 1.0
    assert weighted_reward == 1.0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])

