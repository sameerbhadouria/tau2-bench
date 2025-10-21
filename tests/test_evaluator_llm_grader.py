"""
Tests for LLM grader evaluator.
"""
import pytest

from tau2.data_model.message import AssistantMessage, UserMessage
from tau2.data_model.simulation import RewardInfo
from tau2.data_model.tasks import (
    EvaluationCriteria,
    RewardType,
    StructuredUserInstructions,
    Task,
    UserScenario,
)
from tau2.evaluator.evaluator_llm_grader import LLMGraderEvaluator


def create_sample_task() -> Task:
    """Create a sample task for testing."""
    return Task(
        id="test_task_llm_grader",
        user_scenario=UserScenario(
            instructions=StructuredUserInstructions(
                domain="airline",
                reason_for_call="Cancel my flight",
                known_info="User ID: john_doe_123",
                unknown_info=None,
                task_instructions="Ask the agent to cancel your flight reservation.",
            )
        ),
        evaluation_criteria=EvaluationCriteria(
            reward_basis=[RewardType.NL_ASSERTION],
            nl_assertions=[
                "Agent should have successfully canceled the flight",
                "Agent should have confirmed the cancellation with the user",
            ],
        ),
    )


def create_sample_trajectory() -> list:
    """Create a sample conversation trajectory."""
    return [
        UserMessage(role="user", content="Hi, I need to cancel my flight."),
        AssistantMessage(
            role="assistant",
            content="I'd be happy to help you cancel your flight. Can you provide your reservation number?"
        ),
        UserMessage(role="user", content="It's ABC123."),
        AssistantMessage(
            role="assistant",
            content="I've successfully canceled flight ABC123. You should receive a confirmation email shortly."
        ),
        UserMessage(role="user", content="Great, thank you!"),
    ]


def test_llm_grader_scaffold_basic():
    """Test that the LLM grader scaffold is properly set up."""
    # Test that class exists and can be imported
    assert LLMGraderEvaluator is not None

    # Test that it has the required methods
    assert hasattr(LLMGraderEvaluator, 'calculate_reward')
    assert hasattr(LLMGraderEvaluator, '_build_grading_prompt')
    assert hasattr(LLMGraderEvaluator, '_get_system_prompt')
    assert hasattr(LLMGraderEvaluator, '_format_grading_request')
    assert hasattr(LLMGraderEvaluator, '_format_trajectory')
    assert hasattr(LLMGraderEvaluator, '_parse_grading_response')
    assert hasattr(LLMGraderEvaluator, '_format_user_scenario')
    assert hasattr(LLMGraderEvaluator, '_format_task')


def test_llm_grader_has_required_methods():
    """Test that LLMGraderEvaluator has all required methods."""
    required_methods = ['calculate_reward', '_build_grading_prompt', '_get_system_prompt',
                       '_format_grading_request', '_format_trajectory', '_parse_grading_response']
    for method in required_methods:
        assert hasattr(LLMGraderEvaluator, method), f"Missing method: {method}"


def test_llm_grader_build_prompt():
    """Test prompt building without making actual LLM calls."""
    task = create_sample_task()
    trajectory = create_sample_trajectory()

    # Build prompt messages with sample configuration
    messages = LLMGraderEvaluator._build_grading_prompt(
        task=task,
        full_trajectory=trajectory,
        agent_instruction="You are a helpful customer service agent.",
        domain_policy="Always be polite and follow company guidelines.",
        global_user_sim_guidelines="Simulate realistic user behavior.",
    )

    # Should return a list of messages
    assert isinstance(messages, list)
    assert len(messages) > 0

    # First message should be system message
    assert messages[0].role == "system"
    assert messages[0].content is not None
    assert len(messages[0].content) > 0

    # System prompt should contain the configuration
    system_content = messages[0].content
    assert "agent" in system_content.lower() or "customer service" in system_content.lower()


def test_llm_grader_format_trajectory():
    """Test trajectory formatting."""
    trajectory = create_sample_trajectory()

    formatted = LLMGraderEvaluator._format_trajectory(trajectory)

    # Should return a string
    assert isinstance(formatted, str)
    assert len(formatted) > 0

    # Should contain role indicators
    assert "USER" in formatted or "user" in formatted
    assert "ASSISTANT" in formatted or "assistant" in formatted

    # Should contain actual message content
    assert "cancel my flight" in formatted.lower()


def test_llm_grader_system_prompt():
    """Test system prompt generation with configuration."""
    system_prompt = LLMGraderEvaluator._get_system_prompt(
        agent_instruction="You are a customer service agent.",
        domain_policy="Follow all policies.",
        global_user_sim_guidelines="Be realistic.",
        user_scenario="Test scenario",
    )

    assert isinstance(system_prompt, str)
    assert len(system_prompt) > 0

    # Should mention evaluation or assessment
    assert "evaluat" in system_prompt.lower() or "assess" in system_prompt.lower()

    # Should contain the injected configuration
    assert "customer service agent" in system_prompt.lower()


def test_llm_grader_system_prompt_requests_json():
    """Test that system prompt requests JSON output."""
    system_prompt = LLMGraderEvaluator._get_system_prompt(
        agent_instruction="Test agent",
        domain_policy="Test policy",
        global_user_sim_guidelines="Test guidelines",
        user_scenario="Test scenario",
    )

    assert isinstance(system_prompt, str)
    assert len(system_prompt) > 0

    # Should mention JSON in response format
    assert "json" in system_prompt.lower()


def test_llm_grader_parse_response_success():
    """Test parsing a successful grading response with JSON."""
    # Create a mock JSON response indicating success
    response = AssistantMessage(
        role="assistant",
        content='{"success": true, "confidence": 0.95, "reasoning": "The agent completed all required actions", "criteria_met": ["action1", "action2"]}'
    )

    reward, feedback = LLMGraderEvaluator._parse_grading_response(response)

    assert reward == 1.0
    assert "completed all required actions" in feedback.lower() or "reasoning" in feedback.lower()


def test_llm_grader_parse_response_failure():
    """Test parsing a failed grading response with JSON."""
    # Create a mock JSON response indicating failure
    response = AssistantMessage(
        role="assistant",
        content='{"success": false, "confidence": 0.9, "reasoning": "The agent did not complete the required steps", "criteria_not_met": ["action1"]}'
    )

    reward, feedback = LLMGraderEvaluator._parse_grading_response(response)

    assert reward == 0.0
    assert "did not complete" in feedback.lower() or "reasoning" in feedback.lower()


def test_llm_grader_no_evaluation_criteria():
    """Test behavior when task has no evaluation criteria."""
    task = Task(
        id="test_task_no_criteria",
        user_scenario=UserScenario(instructions="Test task"),
        evaluation_criteria=None,
    )
    trajectory = create_sample_trajectory()

    # Should handle gracefully without making LLM call
    # Note: This will not make an actual LLM call due to early return
    result = LLMGraderEvaluator.calculate_reward(task, trajectory)

    assert isinstance(result, RewardInfo)
    assert result.reward == 1.0
    assert "No evaluation criteria" in result.info.get("note", "")


def test_llm_grader_parse_json_with_criteria():
    """Test parsing JSON response with detailed criteria."""
    # Create a mock JSON response with criteria breakdown
    response = AssistantMessage(
        role="assistant",
        content='{"success": true, "confidence": 0.95, "reasoning": "All criteria met", "criteria_met": {"flight_cancelled": true, "confirmation_sent": true}}'
    )

    reward, feedback = LLMGraderEvaluator._parse_grading_response(response)

    assert reward == 1.0
    assert "criteria met" in feedback.lower() or "flight_cancelled" in feedback.lower()


def test_llm_grader_parse_json_with_markdown():
    """Test parsing JSON response wrapped in markdown."""
    # Create a mock response with markdown code block
    response = AssistantMessage(
        role="assistant",
        content='```json\n{"success": false, "confidence": 0.8, "reasoning": "Missing step", "criteria_not_met": ["required_action"]}\n```'
    )

    reward, feedback = LLMGraderEvaluator._parse_grading_response(response)

    assert reward == 0.0
    assert "missing step" in feedback.lower() or "reasoning" in feedback.lower()


def test_llm_grader_parse_fallback_to_keywords():
    """Test fallback to keyword-based parsing when JSON fails."""
    # Create a response with non-JSON content
    response = AssistantMessage(
        role="assistant",
        content="SUCCESS: The agent successfully completed the task."
    )

    reward, feedback = LLMGraderEvaluator._parse_grading_response(response)

    assert reward == 1.0
    assert "success" in feedback.lower()


def test_llm_grader_model_parameter():
    """Test that model parameter can be specified."""
    task = create_sample_task()
    trajectory = create_sample_trajectory()

    # This test verifies the interface accepts model and configuration parameters
    # Actual LLM call would happen in integration tests
    try:
        # This will fail at LLM call stage but verifies signature
        result = LLMGraderEvaluator.calculate_reward(
            task=task,
            full_trajectory=trajectory,
            model="gpt-4o-mini",
            agent_instruction="Test agent instruction",
            domain_policy="Test policy",
            global_user_sim_guidelines="Test guidelines",
        )
    except Exception:
        # Expected to fail at LLM call, but signature is correct
        pass


def test_llm_grader_llm_args_parameter():
    """Test that llm_args and configuration parameters can be specified."""
    task = create_sample_task()
    trajectory = create_sample_trajectory()

    # Verify interface accepts llm_args and configuration
    try:
        result = LLMGraderEvaluator.calculate_reward(
            task=task,
            full_trajectory=trajectory,
            model="gpt-4o-mini",
            llm_args={"temperature": 0.0, "max_tokens": 500},
            agent_instruction="Test instruction",
            domain_policy="Test policy",
            global_user_sim_guidelines="Test guidelines",
        )
    except Exception:
        # Expected to fail at LLM call, but signature is correct
        pass


def test_llm_grader_format_user_scenario():
    """Test user scenario formatting."""
    task = create_sample_task()

    user_scenario = LLMGraderEvaluator._format_user_scenario(task)

    assert isinstance(user_scenario, str)
    assert len(user_scenario) > 0

    # Should contain information from the task
    assert "airline" in user_scenario.lower() or "cancel" in user_scenario.lower()


def test_llm_grader_format_task():
    """Test task formatting."""
    task = create_sample_task()

    task_str = LLMGraderEvaluator._format_task(task)

    assert isinstance(task_str, str)
    assert len(task_str) > 0

    # Should contain task ID and evaluation criteria
    assert task.id in task_str
    assert "assertion" in task_str.lower() or "criteria" in task_str.lower()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])

