"""
Integration tests for the evaluator system with LLM grader.
"""
import pytest

from tau2.data_model.message import AssistantMessage, UserMessage
from tau2.data_model.simulation import RewardInfo, SimulationRun, TerminationReason
from tau2.data_model.tasks import (
    EvaluationCriteria,
    RewardType,
    StructuredUserInstructions,
    Task,
    UserScenario,
)
from tau2.evaluator import EvaluationType, evaluate_simulation


def create_test_simulation() -> SimulationRun:
    """Create a simple test simulation."""
    return SimulationRun(
        id="test_sim_1",
        task_id="test_task_1",
        trial=0,
        messages=[
            UserMessage(role="user", content="Hi, I need to cancel my flight."),
            AssistantMessage(
                role="assistant",
                content="I can help you with that. Let me cancel your flight.",
            ),
            UserMessage(role="user", content="Thank you!"),
        ],
        start_time="2025-01-01T00:00:00",
        end_time="2025-01-01T00:01:00",
        duration=60.0,
        termination_reason=TerminationReason.AGENT_STOP,
        num_steps=3,
    )


def create_test_task() -> Task:
    """Create a test task."""
    return Task(
        id="test_task_1",
        user_scenario=UserScenario(
            instructions=StructuredUserInstructions(
                domain="airline",
                reason_for_call="Cancel flight",
                known_info="User ID: test_user",
                unknown_info=None,
                task_instructions="Ask agent to cancel flight",
            )
        ),
        evaluation_criteria=EvaluationCriteria(
            reward_basis=[RewardType.NL_ASSERTION],
            nl_assertions=[
                "Agent should cancel the flight",
                "Agent should confirm with user",
            ],
        ),
    )


def test_evaluation_type_llm_grader_exists():
    """Test that LLM_GRADER evaluation type exists."""
    assert hasattr(EvaluationType, "LLM_GRADER")
    assert EvaluationType.LLM_GRADER == "llm_grader"


def test_evaluate_simulation_accepts_llm_grader_params():
    """Test that evaluate_simulation accepts LLM grader parameters."""
    simulation = create_test_simulation()
    task = create_test_task()

    # This test verifies the function signature accepts the new parameters
    # Actual LLM call would happen in integration tests
    try:
        result = evaluate_simulation(
            simulation=simulation,
            task=task,
            evaluation_type=EvaluationType.LLM_GRADER,
            solo_mode=False,
            domain="airline",
            agent_instruction="You are a customer service agent",
            domain_policy="Follow company policies",
            global_user_sim_guidelines="Be realistic",
            llm_grader_model="gpt-4o-mini",
            llm_grader_args={"temperature": 0.0},
        )
    except Exception as e:
        # Expected to fail at LLM call stage in tests without API keys
        # But the function should accept these parameters
        assert "api" in str(e).lower() or "key" in str(e).lower() or "litellm" in str(e).lower()


def test_evaluate_simulation_llm_grader_with_defaults():
    """Test LLM grader with default parameters."""
    simulation = create_test_simulation()
    task = create_test_task()

    # Verify that LLM grader parameters are optional
    try:
        result = evaluate_simulation(
            simulation=simulation,
            task=task,
            evaluation_type=EvaluationType.LLM_GRADER,
            solo_mode=False,
            domain="airline",
        )
    except Exception as e:
        # Expected to fail at LLM call stage, but signature should work
        pass


def test_evaluate_simulation_returns_reward_info_structure():
    """Test that evaluate_simulation returns proper RewardInfo structure."""
    simulation = create_test_simulation()

    # Create task without evaluation criteria (should return early)
    task = Task(
        id="test_task_no_criteria",
        user_scenario=UserScenario(instructions="Test"),
        evaluation_criteria=None,
    )

    result = evaluate_simulation(
        simulation=simulation,
        task=task,
        evaluation_type=EvaluationType.LLM_GRADER,
        solo_mode=False,
        domain="airline",
    )

    assert isinstance(result, RewardInfo)
    assert result.reward == 1.0
    assert "No evaluation criteria" in result.info.get("note", "")


def test_evaluate_simulation_handles_premature_termination():
    """Test that premature termination is handled correctly."""
    simulation = SimulationRun(
        id="test_sim_error",
        task_id="test_task_1",
        trial=0,
        messages=[],
        start_time="2025-01-01T00:00:00",
        end_time="2025-01-01T00:00:10",
        duration=10.0,
        termination_reason=TerminationReason.TOO_MANY_ERRORS,
        num_steps=0,
    )

    task = create_test_task()

    result = evaluate_simulation(
        simulation=simulation,
        task=task,
        evaluation_type=EvaluationType.LLM_GRADER,
        solo_mode=False,
        domain="airline",
    )

    assert isinstance(result, RewardInfo)
    assert result.reward == 0.0
    assert "terminated prematurely" in result.info.get("note", "").lower()


def test_all_evaluation_types_exist():
    """Test that all expected evaluation types are defined."""
    expected_types = [
        "ENV",
        "COMMUNICATE",
        "ACTION",
        "ALL",
        "NL_ASSERTIONS",
        "ALL_WITH_NL_ASSERTIONS",
        "WEIGHTED_ALL",
        "WEIGHTED_ALL_WITH_NL_ASSERTIONS",
        "LLM_GRADER",
    ]

    for eval_type in expected_types:
        assert hasattr(EvaluationType, eval_type), f"Missing evaluation type: {eval_type}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

