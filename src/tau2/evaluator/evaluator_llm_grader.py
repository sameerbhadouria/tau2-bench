"""
LLM-based grader for evaluating agent trajectories using language models.

This evaluator uses LLMs to assess agent performance on tasks by analyzing
conversation trajectories and task requirements.
"""

from typing import Any, Optional

from loguru import logger

from tau2.data_model.message import Message, SystemMessage, UserMessage
from tau2.data_model.simulation import RewardInfo
from tau2.data_model.tasks import RewardType, Task
from tau2.evaluator.evaluator_base import EvaluatorBase
from tau2.utils.llm_utils import generate


class LLMGraderEvaluator(EvaluatorBase):
    """
    Evaluates agent performance using an LLM as a grader.

    This evaluator leverages language models to assess whether agents successfully
    completed tasks by analyzing the conversation trajectory, task requirements,
    and evaluation criteria.
    """

    SYSTEM_PROMPT_TEMPLATE = """
        You are an expert evaluator assessing whether an agent successfully completed a customer service task in a simulated environment.

        The agent is provided the following instruction:

        {agent_instruction}

        \n\n

        On the other hand, the user is going to be provided with the following instruction:

        {global_user_sim_guidelines}

        <scenario>
        {user_scenario}
        </scenario>

        \n\n

        You will be provided with 2 inputs:
        1. <task> - A structured description of the evaluation task requirements, including:
            - Task ID and description (purpose, relevant policies, notes)
            - Evaluation criteria:
              * Expected Actions: Specific tool calls the agent/user should make
              * Communication Requirements: Information the agent should communicate
              * Natural Language Assertions: Success conditions in natural language
              * Environment Assertions: Expected state changes in the environment
        2. <conversation_trajectory> - The complete conversation between agent and user, formatted as:
            - Numbered messages showing the role (USER/ASSISTANT/TOOL) and content
            - Tool calls made by the agent with their names
            - Full chronological sequence of the interaction

       Your goal is to quantify the agent's performance based on the task definition and conversation trajectory.
       Respond with a JSON object with the following fields:
       - success: true or false
       - confidence: 0.0-1.0
       - reasoning: brief explanation of your decision
       - criteria_met: list of criteria that were met
       - criteria_not_met: list of criteria that were not met
       - criteria_partially_met: list of criteria that were partially met
       - criteria_not_applicable: list of criteria that were not applicable
       - criteria_not_evaluated: list of criteria that were not evaluated
       - criteria_not_evaluated_reason: brief explanation of why the criteria were not evaluated
       - criteria_not_evaluated_reason: brief explanation of why the criteria were not evaluated
    """

    USER_PROMPT_TEMPLATE = """
    <task>
    {task}
    </task>
    <conversation_trajectory>
    {conversation_trajectory}
    </conversation_trajectory>
    """

    @classmethod
    def calculate_reward(
        cls,
        task: Task,
        full_trajectory: list[Message],
        model: Optional[str] = None,
        llm_args: Optional[dict] = None,
        agent_instruction: Optional[str] = None,
        domain_policy: Optional[str] = None,
        global_user_sim_guidelines: Optional[str] = None,
        **kwargs: Any,
    ) -> RewardInfo:
        """
        Calculate the reward using an LLM as a grader.

        Args:
            task: The task definition with evaluation criteria
            full_trajectory: Complete conversation trajectory between agent and user
            model: LLM model to use for grading (e.g., "gpt-4o", "claude-3-5-sonnet")
                   If None, uses the default grader model from config
            llm_args: Additional arguments to pass to the LLM (temperature, etc.)
            agent_instruction: Instructions given to the agent (e.g., system prompt)
            domain_policy: Domain-specific policies the agent should follow
            global_user_sim_guidelines: Global guidelines for user simulator behavior
            **kwargs: Additional keyword arguments

        Returns:
            RewardInfo with the graded reward and detailed feedback
        """
        if task.evaluation_criteria is None:
            return RewardInfo(
                reward=1.0,
                info={"note": "No evaluation criteria"},
                reward_breakdown={RewardType.NL_ASSERTION: 1.0},
            )

        # Use default model if none specified
        if model is None:
            from tau2.config import DEFAULT_LLM_NL_ASSERTIONS
            model = DEFAULT_LLM_NL_ASSERTIONS
            logger.debug(f"Using default LLM grader model: {model}")

        # Use default args if none specified
        if llm_args is None:
            from tau2.config import DEFAULT_LLM_NL_ASSERTIONS_ARGS
            llm_args = DEFAULT_LLM_NL_ASSERTIONS_ARGS.copy()

        # Build the grading prompt
        grading_messages = cls._build_grading_prompt(
            task=task,
            full_trajectory=full_trajectory,
            agent_instruction=agent_instruction,
            domain_policy=domain_policy,
            global_user_sim_guidelines=global_user_sim_guidelines,
        )

        # Call the LLM to grade the trajectory
        try:
            response = generate(
                model=model,
                messages=grading_messages,
                **llm_args,
            )

            # Parse the LLM response
            reward, result_dict = cls._parse_grading_response(response)

            # Build info dict with all available fields
            info_dict = {
                "grading_model": model,
                "grading_cost": response.cost if hasattr(response, 'cost') else None,
            }

            # Add all fields from the result_dict (including criteria fields)
            info_dict.update(result_dict)

            return RewardInfo(
                reward=reward,
                info=info_dict,
                reward_breakdown={RewardType.NL_ASSERTION: reward},
            )

        except Exception as e:
            logger.error(f"Error during LLM grading: {e}")
            return RewardInfo(
                reward=0.0,
                info={
                    "error": str(e),
                    "note": "LLM grading failed",
                },
                reward_breakdown={RewardType.NL_ASSERTION: 0.0},
            )

    @classmethod
    def _build_grading_prompt(
        cls,
        task: Task,
        full_trajectory: list[Message],
        agent_instruction: Optional[str] = None,
        domain_policy: Optional[str] = None,
        global_user_sim_guidelines: Optional[str] = None,
    ) -> list[Message]:
        """
        Build the prompt for the LLM grader.

        This method constructs messages that:
        1. Explain the grading task to the LLM
        2. Provide the task definition and evaluation criteria
        3. Present the conversation trajectory
        4. Ask for a structured grading decision

        Args:
            task: The task definition
            full_trajectory: The conversation trajectory to grade
            agent_instruction: Instructions given to the agent
            domain_policy: Domain-specific policies
            global_user_sim_guidelines: Global user simulator guidelines

        Returns:
            List of messages to send to the LLM grader
        """
        # Extract user scenario from task
        user_scenario = cls._format_user_scenario(task)

        # Build system prompt with configuration
        system_prompt = cls._get_system_prompt(
            agent_instruction=agent_instruction or "Not provided",
            domain_policy=domain_policy or "Not provided",
            global_user_sim_guidelines=global_user_sim_guidelines or "Not provided",
            user_scenario=user_scenario,
        )

        # Build user prompt with task and trajectory
        user_prompt = cls._format_grading_request(task, full_trajectory)

        return [
            SystemMessage(role="system", content=system_prompt),
            UserMessage(role="user", content=user_prompt),
        ]

    @classmethod
    def _get_system_prompt(
        cls,
        agent_instruction: str = "Not provided",
        domain_policy: str = "Not provided",
        global_user_sim_guidelines: str = "Not provided",
        user_scenario: str = "Not provided",
    ) -> str:
        """
        Get the system prompt that defines the LLM grader's role.

        Args:
            agent_instruction: Instructions given to the agent
            domain_policy: Domain-specific policies
            global_user_sim_guidelines: Global user simulator guidelines
            user_scenario: User scenario for this specific task

        Returns:
            System prompt string formatted with configuration
        """
        return cls.SYSTEM_PROMPT_TEMPLATE.format(
            agent_instruction=agent_instruction,
            domain_policy=domain_policy,
            global_user_sim_guidelines=global_user_sim_guidelines,
            user_scenario=user_scenario,
        )

    @classmethod
    def _format_user_scenario(cls, task: Task) -> str:
        """
        Extract and format the user scenario from the task.

        Args:
            task: The task definition

        Returns:
            Formatted user scenario string
        """
        if not task.user_scenario:
            return "Not provided"

        parts = []

        # Add persona if available
        if task.user_scenario.persona:
            parts.append(f"Persona: {task.user_scenario.persona}")

        # Add instructions
        instructions = task.user_scenario.instructions
        if isinstance(instructions, str):
            parts.append(f"Instructions: {instructions}")
        else:
            # StructuredUserInstructions
            if hasattr(instructions, 'domain'):
                parts.append(f"Domain: {instructions.domain}")
            if hasattr(instructions, 'reason_for_call'):
                parts.append(f"Reason for call: {instructions.reason_for_call}")
            if hasattr(instructions, 'known_info') and instructions.known_info:
                parts.append(f"Known info: {instructions.known_info}")
            if hasattr(instructions, 'unknown_info') and instructions.unknown_info:
                parts.append(f"Unknown info: {instructions.unknown_info}")
            if hasattr(instructions, 'task_instructions'):
                parts.append(f"Task instructions: {instructions.task_instructions}")

        return "\n".join(parts) if parts else "Not provided"

    @classmethod
    def _format_grading_request(
        cls,
        task: Task,
        full_trajectory: list[Message],
    ) -> str:
        """
        Format the grading request with task details and trajectory.

        Args:
            task: The task definition
            full_trajectory: The conversation trajectory

        Returns:
            Formatted prompt string using USER_PROMPT_TEMPLATE
        """
        # Format task as JSON-like string for clarity
        task_str = cls._format_task(task)

        # Format conversation trajectory
        trajectory_str = cls._format_trajectory(full_trajectory)

        # Use the template
        return cls.USER_PROMPT_TEMPLATE.format(
            task=task_str,
            conversation_trajectory=trajectory_str,
        )

    @classmethod
    def _format_task(cls, task: Task) -> str:
        """
        Format the task definition for the LLM.

        Args:
            task: The task definition

        Returns:
            Formatted task string
        """
        parts = []

        # Task ID
        parts.append(f"Task ID: {task.id}")

        # Description
        if task.description:
            if task.description.purpose:
                parts.append(f"\nPurpose: {task.description.purpose}")
            if task.description.relevant_policies:
                parts.append(f"\nRelevant Policies: {task.description.relevant_policies}")
            if task.description.notes:
                parts.append(f"\nNotes: {task.description.notes}")

        # Evaluation Criteria
        if task.evaluation_criteria:
            parts.append("\n--- Evaluation Criteria ---")

            # Actions
            if task.evaluation_criteria.actions:
                parts.append(f"\nExpected Actions ({len(task.evaluation_criteria.actions)}):")
                for i, action in enumerate(task.evaluation_criteria.actions, 1):
                    parts.append(f"  {i}. {action.requestor}: {action.name}({action.arguments})")
                    if action.info:
                        parts.append(f"     Info: {action.info}")

            # Communication requirements
            if task.evaluation_criteria.communicate_info:
                parts.append(f"\nCommunication Requirements ({len(task.evaluation_criteria.communicate_info)}):")
                for i, comm in enumerate(task.evaluation_criteria.communicate_info, 1):
                    parts.append(f"  {i}. {comm}")

            # NL Assertions
            if task.evaluation_criteria.nl_assertions:
                parts.append(f"\nNatural Language Assertions ({len(task.evaluation_criteria.nl_assertions)}):")
                for i, assertion in enumerate(task.evaluation_criteria.nl_assertions, 1):
                    parts.append(f"  {i}. {assertion}")

            # Environment assertions
            if task.evaluation_criteria.env_assertions:
                parts.append(f"\nEnvironment Assertions ({len(task.evaluation_criteria.env_assertions)}):")
                for i, env_assert in enumerate(task.evaluation_criteria.env_assertions, 1):
                    parts.append(f"  {i}. {env_assert.func_name}({env_assert.arguments}) should be {env_assert.assert_value}")

        return "\n".join(parts)

    @classmethod
    def _format_trajectory(cls, trajectory: list[Message]) -> str:
        """
        Format the conversation trajectory for the LLM grader.

        Args:
            trajectory: List of messages in the conversation

        Returns:
            Formatted trajectory string
        """
        # TODO: Implement more sophisticated formatting
        lines = []
        for i, message in enumerate(trajectory, 1):
            role = message.role.upper()
            content = message.content or ""

            # Add tool call information for assistant messages
            if hasattr(message, 'tool_calls') and message.tool_calls:
                tool_info = ", ".join([f"{tc.name}()" for tc in message.tool_calls])
                lines.append(f"[{i}] {role}: {content}")
                lines.append(f"    Tools called: {tool_info}")
            else:
                lines.append(f"[{i}] {role}: {content}")

        return "\n".join(lines)

    @classmethod
    def _parse_grading_response(cls, response: Message) -> tuple[float, dict]:
        """
        Parse the LLM's grading response into a reward score and detailed result.

        Expected JSON format from LLM:
        {
            "success": true/false,
            "confidence": 0.0-1.0,
            "reasoning": "explanation",
            "criteria_met": [...],
            "criteria_not_met": [...],
            "criteria_partially_met": [...],
            ...
        }

        Args:
            response: The LLM's response message

        Returns:
            Tuple of (reward, result_dict) where reward is 0.0 or 1.0 and result_dict contains all grading details
        """
        import json

        content = response.content or ""

        try:
            # Try to extract JSON from the response
            # LLMs sometimes wrap JSON in markdown code blocks
            json_content = content

            if "```json" in content:
                # Extract from ```json ... ```
                json_content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                # Extract from ``` ... ```
                json_content = content.split("```")[1].split("```")[0].strip()

            # Parse the JSON
            result = json.loads(json_content)

            # Extract success status
            success = result.get("success", False)
            reward = 1.0 if success else 0.0

            # Build detailed feedback from the JSON structure
            feedback_parts = []

            # Add reasoning
            if "reasoning" in result:
                feedback_parts.append(f"Reasoning: {result['reasoning']}")

            # Add confidence if available
            if "confidence" in result:
                feedback_parts.append(f"Confidence: {result['confidence']}")

            # Add criteria met
            if "criteria_met" in result and result["criteria_met"]:
                criteria_met = result["criteria_met"]
                if isinstance(criteria_met, dict):
                    feedback_parts.append(f"Criteria Met: {', '.join(criteria_met.keys())}")
                elif isinstance(criteria_met, list):
                    feedback_parts.append(f"Criteria Met: {', '.join(str(c) for c in criteria_met)}")

            # Add criteria not met
            if "criteria_not_met" in result and result["criteria_not_met"]:
                criteria_not_met = result["criteria_not_met"]
                if isinstance(criteria_not_met, dict):
                    feedback_parts.append(f"Criteria Not Met: {', '.join(criteria_not_met.keys())}")
                elif isinstance(criteria_not_met, list):
                    feedback_parts.append(f"Criteria Not Met: {', '.join(str(c) for c in criteria_not_met)}")

            # Add criteria partially met
            if "criteria_partially_met" in result and result["criteria_partially_met"]:
                criteria_partial = result["criteria_partially_met"]
                if isinstance(criteria_partial, dict):
                    feedback_parts.append(f"Criteria Partially Met: {', '.join(criteria_partial.keys())}")
                elif isinstance(criteria_partial, list):
                    feedback_parts.append(f"Criteria Partially Met: {', '.join(str(c) for c in criteria_partial)}")

            feedback = " | ".join(feedback_parts) if feedback_parts else result.get("reasoning", "No detailed feedback")

            # Add the feedback string to the result dict for backward compatibility
            result["grading_feedback"] = feedback

            logger.debug(f"Successfully parsed LLM grading response: success={success}, confidence={result.get('confidence', 'N/A')}")

            return reward, result

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.warning(f"Failed to parse JSON from LLM response: {e}")
            logger.debug(f"Response content: {content[:500]}...")

            # Fallback to simple keyword-based parsing
            content_lower = content.lower()

            if "success" in content_lower and "failure" not in content_lower:
                reward = 1.0
                success = True
            elif "failure" in content_lower or "fail" in content_lower:
                reward = 0.0
                success = False
            else:
                # Default to failure if unclear
                logger.warning(f"Unclear LLM grading response, defaulting to failure")
                reward = 0.0
                success = False

            feedback = content.strip()[:500]  # Truncate long responses

            # Return a dict with fallback information
            return reward, {
                "success": success,
                "grading_feedback": feedback,
                "parse_error": str(e),
                "note": "Failed to parse structured JSON response, used fallback parsing"
            }



