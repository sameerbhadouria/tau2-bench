from enum import Enum

from tau2.data_model.simulation import RewardInfo, SimulationRun, TerminationReason
from tau2.data_model.tasks import RewardType, Task
from tau2.evaluator.evaluator_action import ActionEvaluator
from tau2.evaluator.evaluator_communicate import CommunicateEvaluator
from tau2.evaluator.evaluator_env import EnvironmentEvaluator
from tau2.evaluator.evaluator_llm_grader import LLMGraderEvaluator
from tau2.evaluator.evaluator_nl_assertions import NLAssertionsEvaluator
from tau2.registry import registry


class EvaluationType(str, Enum):
    ENV = "env"
    COMMUNICATE = "communicate"
    ACTION = "action"
    ALL = "all"
    NL_ASSERTIONS = "nl_assertions"  # WIP
    ALL_WITH_NL_ASSERTIONS = "all_with_nl_assertions"  # WIP
    WEIGHTED_ALL = "weighted_all"
    WEIGHTED_ALL_WITH_NL_ASSERTIONS = "weighted_all_with_nl_assertions"
    LLM_GRADER = "llm_grader"


def calculate_reward_weights(task: Task) -> dict[RewardType, float]:
    """
    Calculate weights for each reward component based on the count of checks/assertions.

    Weight calculation:
    - DB: 1 if present (single check)
    - ENV_ASSERTION: count of env_assertions
    - ACTION: count of actions
    - COMMUNICATE: count of communicate_info
    - NL_ASSERTION: count of nl_assertions

    Returns normalized weights that sum to 1.0 for active components.
    """
    if task.evaluation_criteria is None:
        return {}

    eval_criteria = task.evaluation_criteria
    counts = {}

    # Count DB checks (always 1 if DB is in reward_basis)
    if RewardType.DB in eval_criteria.reward_basis:
        counts[RewardType.DB] = 1

    # Count environment assertions
    if RewardType.ENV_ASSERTION in eval_criteria.reward_basis:
        num_env_assertions = len(eval_criteria.env_assertions) if eval_criteria.env_assertions else 0
        if num_env_assertions > 0:
            counts[RewardType.ENV_ASSERTION] = num_env_assertions

    # Count actions
    if RewardType.ACTION in eval_criteria.reward_basis:
        num_actions = len(eval_criteria.actions) if eval_criteria.actions else 0
        if num_actions > 0:
            counts[RewardType.ACTION] = num_actions

    # Count communicate info
    if RewardType.COMMUNICATE in eval_criteria.reward_basis:
        num_communicate = len(eval_criteria.communicate_info) if eval_criteria.communicate_info else 0
        if num_communicate > 0:
            counts[RewardType.COMMUNICATE] = num_communicate

    # Count NL assertions
    if RewardType.NL_ASSERTION in eval_criteria.reward_basis:
        num_nl_assertions = len(eval_criteria.nl_assertions) if eval_criteria.nl_assertions else 0
        if num_nl_assertions > 0:
            counts[RewardType.NL_ASSERTION] = num_nl_assertions

    # Normalize to sum to 1.0
    total_count = sum(counts.values())
    if total_count == 0:
        return {}

    weights = {component: count / total_count for component, count in counts.items()}
    return weights


def evaluate_all_components(
    simulation: SimulationRun,
    task: Task,
    domain: str,
    solo_mode: bool,
    include_nl_assertions: bool,
    weights: dict[RewardType, float] | None = None,
) -> RewardInfo:
    """
    Evaluate all reward components and combine them.

    Args:
        simulation: The simulation run to evaluate
        task: The task definition
        domain: The domain name
        solo_mode: Whether running in solo mode
        include_nl_assertions: Whether to include NL assertions in evaluation
        weights: Optional weights for combining components. If None, uses multiplicative
                 combination (product). If provided, uses weighted average.

    Returns:
        RewardInfo with combined reward and detailed breakdown
    """
    # Evaluate all components
    env_reward_info = EnvironmentEvaluator.calculate_reward(
        environment_constructor=registry.get_env_constructor(domain),
        task=task,
        full_trajectory=simulation.messages,
        solo_mode=solo_mode,
    )
    action_reward_info = ActionEvaluator.calculate_reward(
        task=task,
        full_trajectory=simulation.messages,
    )
    communicate_reward_info = CommunicateEvaluator.calculate_reward(
        task=task,
        full_trajectory=simulation.messages,
    )
    nl_reward_info = None
    if include_nl_assertions:
        nl_reward_info = NLAssertionsEvaluator.calculate_reward(
            task=task,
            full_trajectory=simulation.messages,
        )

    # Define component groupings
    env_bases = {RewardType.DB, RewardType.ENV_ASSERTION}
    action_bases = {RewardType.ACTION}
    nl_bases = {RewardType.NL_ASSERTION}
    comm_bases = {RewardType.COMMUNICATE}
    task_reward_basis = set(task.evaluation_criteria.reward_basis)

    reward_breakdown = {}

    # Combine rewards based on whether weights are provided
    if weights is None:
        # Multiplicative combination (original ALL behavior)
        reward = 1.0

        if task_reward_basis & env_bases:
            if env_reward_info.reward_breakdown is not None:
                reward_breakdown.update(env_reward_info.reward_breakdown)
            reward *= env_reward_info.reward

        if task_reward_basis & action_bases:
            if action_reward_info.reward_breakdown is not None:
                reward_breakdown.update(action_reward_info.reward_breakdown)
            reward *= action_reward_info.reward

        if task_reward_basis & nl_bases:
            if not include_nl_assertions:
                raise ValueError(
                    "NL assertions are part of the reward basis, but they are not being evaluated."
                )
            if nl_reward_info.reward_breakdown is not None:
                reward_breakdown.update(nl_reward_info.reward_breakdown)
            reward *= nl_reward_info.reward

        if task_reward_basis & comm_bases:
            if communicate_reward_info.reward_breakdown is not None:
                reward_breakdown.update(communicate_reward_info.reward_breakdown)
            reward *= communicate_reward_info.reward
    else:
        # Weighted average combination (WEIGHTED_ALL behavior)
        reward = 0.0

        if task_reward_basis & env_bases:
            if env_reward_info.reward_breakdown is not None:
                reward_breakdown.update(env_reward_info.reward_breakdown)
            # For env, we may have both DB and ENV_ASSERTION
            for reward_type in env_bases:
                if reward_type in weights:
                    component_reward = env_reward_info.reward_breakdown.get(reward_type, 0.0) if env_reward_info.reward_breakdown else 0.0
                    reward += weights[reward_type] * component_reward

        if task_reward_basis & action_bases:
            if action_reward_info.reward_breakdown is not None:
                reward_breakdown.update(action_reward_info.reward_breakdown)
            if RewardType.ACTION in weights:
                reward += weights[RewardType.ACTION] * action_reward_info.reward

        if task_reward_basis & nl_bases:
            if not include_nl_assertions:
                raise ValueError(
                    "NL assertions are part of the reward basis, but they are not being evaluated."
                )
            if nl_reward_info.reward_breakdown is not None:
                reward_breakdown.update(nl_reward_info.reward_breakdown)
            if RewardType.NL_ASSERTION in weights:
                reward += weights[RewardType.NL_ASSERTION] * nl_reward_info.reward

        if task_reward_basis & comm_bases:
            if communicate_reward_info.reward_breakdown is not None:
                reward_breakdown.update(communicate_reward_info.reward_breakdown)
            if RewardType.COMMUNICATE in weights:
                reward += weights[RewardType.COMMUNICATE] * communicate_reward_info.reward

    return RewardInfo(
        reward=reward,
        db_check=env_reward_info.db_check,
        env_assertions=env_reward_info.env_assertions,
        action_checks=action_reward_info.action_checks,
        nl_assertions=(
            nl_reward_info.nl_assertions if nl_reward_info is not None else None
        ),
        communicate_checks=communicate_reward_info.communicate_checks,
        reward_basis=task.evaluation_criteria.reward_basis,
        reward_breakdown=reward_breakdown,
        reward_weights=weights,
        info={
            "env": env_reward_info.info,
            "nl": nl_reward_info.info if nl_reward_info is not None else None,
            "communicate": communicate_reward_info.info,
            "action": action_reward_info.info,
        },
    )


def evaluate_simulation(
    simulation: SimulationRun,
    task: Task,
    evaluation_type: EvaluationType,
    solo_mode: bool,
    domain: str,
    agent_instruction: str | None = None,
    domain_policy: str | None = None,
    global_user_sim_guidelines: str | None = None,
    llm_grader_model: str | None = None,
    llm_grader_args: dict | None = None,
) -> RewardInfo:
    """
    Evaluate the simulation based on the evaluation type.

    Args:
        simulation: The simulation run to evaluate
        task: The task definition
        evaluation_type: Type of evaluation to perform
        solo_mode: Whether running in solo mode
        domain: Domain name
        agent_instruction: Instructions given to agent (for LLM_GRADER)
        domain_policy: Domain policies (for LLM_GRADER)
        global_user_sim_guidelines: User simulator guidelines (for LLM_GRADER)
        llm_grader_model: Model to use for LLM grading
        llm_grader_args: Additional args for LLM grader

    Returns:
        RewardInfo with evaluation results
    """
    if simulation.termination_reason in {
        TerminationReason.TOO_MANY_ERRORS,
        TerminationReason.MAX_STEPS,
    }:
        return RewardInfo(
            reward=0.0,
            info={
                "note": f"Simulation terminated prematurely. Termination reason: {simulation.termination_reason}"
            },
        )
    if task.evaluation_criteria is None:
        return RewardInfo(
            reward=1.0,
            info={"note": "No evaluation criteria"},
        )
    if evaluation_type == EvaluationType.ENV:
        reward_info = EnvironmentEvaluator.calculate_reward(
            environment_constructor=registry.get_env_constructor(domain),
            task=task,
            full_trajectory=simulation.messages,
            solo_mode=solo_mode,
        )
    elif evaluation_type == EvaluationType.NL_ASSERTIONS:
        reward_info = NLAssertionsEvaluator.calculate_reward(
            task=task,
            full_trajectory=simulation.messages,
        )
    elif evaluation_type == EvaluationType.COMMUNICATE:
        reward_info = CommunicateEvaluator.calculate_reward(
            task=task,
            full_trajectory=simulation.messages,
        )
    elif evaluation_type == EvaluationType.ACTION:
        reward_info = ActionEvaluator.calculate_reward(
            task=task,
            full_trajectory=simulation.messages,
        )
    elif evaluation_type in {EvaluationType.ALL, EvaluationType.ALL_WITH_NL_ASSERTIONS}:
        # Use multiplicative combination (product) - weights=None
        include_nl = evaluation_type == EvaluationType.ALL_WITH_NL_ASSERTIONS
        reward_info = evaluate_all_components(
            simulation=simulation,
            task=task,
            domain=domain,
            solo_mode=solo_mode,
            include_nl_assertions=include_nl,
            weights=None,
        )
    elif evaluation_type in {EvaluationType.WEIGHTED_ALL, EvaluationType.WEIGHTED_ALL_WITH_NL_ASSERTIONS}:
        # Use weighted average combination
        include_nl = evaluation_type == EvaluationType.WEIGHTED_ALL_WITH_NL_ASSERTIONS
        weights = calculate_reward_weights(task)
        reward_info = evaluate_all_components(
            simulation=simulation,
            task=task,
            domain=domain,
            solo_mode=solo_mode,
            include_nl_assertions=include_nl,
            weights=weights,
        )
    elif evaluation_type == EvaluationType.LLM_GRADER:
        # Use LLM-based grading
        reward_info = LLMGraderEvaluator.calculate_reward(
            task=task,
            full_trajectory=simulation.messages,
            model=llm_grader_model,
            llm_args=llm_grader_args,
            agent_instruction=agent_instruction,
            domain_policy=domain_policy,
            global_user_sim_guidelines=global_user_sim_guidelines,
        )
    else:
        raise ValueError(f"Unknown evaluation type: {evaluation_type}")
    return reward_info
