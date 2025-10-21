import math
import re

import pandas as pd
from loguru import logger
from pydantic import BaseModel

from tau2.data_model.simulation import Results
from tau2.data_model.message import AssistantMessage


def is_successful(reward: float) -> bool:
    """
    Check if the reward is successful.
    """
    return (1 - 1e-6) <= reward <= (1 + 1e-6)


class DerivedComplexityStats(BaseModel):
    """
    Statistics for derived/inferred task complexity scores.
    """
    min: float
    max: float
    mean: float
    percentile_90: float
    percentile_95: float
    percentile_99: float


class AgentMetrics(BaseModel):
    avg_reward: float
    avg_reward_sem: float
    avg_reward_ci_95: tuple[float, float]
    success_rate: float
    derived_complexity_stats: DerivedComplexityStats
    success_rate_by_complexity_quartile: dict[str, float]
    pass_hat_ks: dict[int, float]
    avg_num_tool_calls: float
    avg_agent_cost: float

    def as_dict(self) -> dict:
        data = {
            "avg_reward": self.avg_reward,
            "avg_reward_sem": self.avg_reward_sem,
            "avg_reward_ci_95_lower": self.avg_reward_ci_95[0],
            "avg_reward_ci_95_upper": self.avg_reward_ci_95[1],
            "success_rate": self.success_rate,
            "derived_complexity_min": self.derived_complexity_stats.min,
            "derived_complexity_max": self.derived_complexity_stats.max,
            "derived_complexity_mean": self.derived_complexity_stats.mean,
            "derived_complexity_p90": self.derived_complexity_stats.percentile_90,
            "derived_complexity_p95": self.derived_complexity_stats.percentile_95,
            "derived_complexity_p99": self.derived_complexity_stats.percentile_99,
            "avg_num_tool_calls": self.avg_num_tool_calls,
            "avg_agent_cost": self.avg_agent_cost,
        }
        for quartile, rate in self.success_rate_by_complexity_quartile.items():
            data[f"success_rate_complexity_{quartile}"] = rate
        for k, v in self.pass_hat_ks.items():
            data[f"pass_hat_{k}"] = v
        return data


def pass_hat_k(num_trials: int, success_count: int, k: int) -> float:
    """
    Compute the pass^k metric for the given number of trials, success count, and k.
    from https://arxiv.org/pdf/2406.12045
    Args:
        num_trials: The number of trials.
        success_count: The number of successful trials.
        k: The number of trials to consider.
    Returns:
        The pass^k metric.
    """
    if num_trials < k:
        raise ValueError(f"Number of trials {num_trials} is less than k {k}.")
    return math.comb(success_count, k) / math.comb(num_trials, k)


def get_metrics_df(results: Results) -> tuple[pd.DataFrame, int]:
    """
    Convert the results to a dataframe and add a column for success.
    Checks that all simulations have the same number of trials.
    Returns the maximum number of trials that can be used for pass^k metrics.
    """
    df = results.to_df()
    df["success"] = df.reward.apply(is_successful)
    if len(df.info_num_trials.unique()) > 1:
        logger.warning(
            f"All simulations must have the same number of trials. Found {df.info_num_trials.unique()}"
        )
    max_k = df.info_num_trials.max()

    task_ids_counts = [(tid, count) for tid, count in df.task_id.value_counts().items()]
    task_ids_counts.sort(key=lambda x: x[1])
    min_k = task_ids_counts[0][1]
    if min_k < max_k:
        logger.warning(
            f"The minimum number of trials for a task is {min_k}, which is less than the expected number of trials {max_k}. Setting max k to {min_k}."
        )
        max_k = min_k
    return df, max_k


def get_tasks_pass_hat_k(results: Results) -> pd.DataFrame:
    """
    Compute the pass^k for each k from 1 to the maximum number of trials.
    """
    df, max_k = get_metrics_df(results)
    dfs = []
    for k in range(1, max_k + 1):
        res = df.groupby("task_id")["success"].apply(
            lambda df: pass_hat_k(len(df), df.sum(), k)
        )
        res.name = f"pass^{k}"
        dfs.append(res)
    df_pass_hat_k = pd.concat(dfs, axis=1)
    task_columns = [
        "task_num_agent_actions",
        "task_num_user_actions",
        "task_num_actions",
    ]
    df_task_infos = df.groupby("task_id").first()[task_columns]
    df_pass_hat_k = df_task_infos.join(df_pass_hat_k)
    return df_pass_hat_k


def count_agent_tool_calls(results: Results) -> dict[str, int]:
    """
    Count the actual number of tool calls made by the agent (assistant) in each simulation.
    Returns a dict mapping simulation_id to number of tool calls.
    """
    tool_call_counts = {}
    for sim in results.simulations:
        num_tool_calls = 0
        for message in sim.messages:
            if isinstance(message, AssistantMessage) and message.tool_calls is not None:
                num_tool_calls += len(message.tool_calls)
        tool_call_counts[sim.id] = num_tool_calls
    return tool_call_counts


def compute_derived_task_complexity(df: pd.DataFrame) -> pd.Series:
    """
    Compute derived/inferred task complexity as an empirical z-score based on:
    - task_num_agent_actions: Expected number of agent tool calls defined in task (task-intrinsic)
    - num_messages: Actual number of messages exchanged during simulation (empirical)
    - duration: Actual time taken to complete the task (empirical)

    This is an empirical measure derived from actual execution, not a predefined task property.

    Returns a Series with task_id as index and complexity z-score as values.
    """
    # Aggregate metrics at task level
    # Note: Combines task-defined expectations (task_num_agent_actions) with
    # empirical execution metrics (num_messages, duration) averaged across trials
    task_metrics = df.groupby("task_id").agg({
        "task_num_agent_actions": "first",
        "num_messages": "mean",
        "duration": "mean",
    })

    # Standardize each metric (z-score)
    task_metrics["task_num_agent_actions_std"] = (
        (task_metrics["task_num_agent_actions"] - task_metrics["task_num_agent_actions"].mean()) /
        task_metrics["task_num_agent_actions"].std()
    )
    task_metrics["num_messages_std"] = (
        (task_metrics["num_messages"] - task_metrics["num_messages"].mean()) /
        task_metrics["num_messages"].std()
    )
    task_metrics["duration_std"] = (
        (task_metrics["duration"] - task_metrics["duration"].mean()) /
        task_metrics["duration"].std()
    )

    # Linear combination of standardized values (average of z-scores)
    derived_complexity = (
        task_metrics["task_num_agent_actions_std"] +
        task_metrics["num_messages_std"] +
        task_metrics["duration_std"]
    ) / 3

    return derived_complexity


def prepare_dfs(results: Results) -> tuple[pd.DataFrame, pd.DataFrame]:
    df, max_k = get_metrics_df(results)
    df_pass_hat_k = get_tasks_pass_hat_k(results)
    df_pass_hat_k["num_actions"] = df.groupby("task_id").first()["task_num_actions"]
    df_pass_hat_k = df_pass_hat_k.sort_values(by="num_actions")
    return df, df_pass_hat_k


def compute_metrics(results: Results) -> AgentMetrics:
    """
    Compute metrics for the agent.
    - average reward
    - average reward SEM (Standard Error of the Mean)
    - average reward 95% confidence interval
    - success rate
    - derived task complexity statistics (min, max, mean, 90th, 95th, 99th percentiles)
    - success rate by derived complexity quartile
    - average number of tool calls (actual agent tool calls)
    - pass^k
    """
    df, df_pass_hat_k = prepare_dfs(results)
    avg_reward = df.reward.mean()
    success_rate = df.success.sum() / len(df)

    agent_tool_call_counts = count_agent_tool_calls(results)
    df["num_agent_tool_calls"] = df["simulation_id"].map(agent_tool_call_counts)
    avg_num_tool_calls = df["num_agent_tool_calls"].mean()

    # Reference: SEM and 95% CI for avg_reward: https://arxiv.org/pdf/2411.00640
    n = len(df)
    reward_std = df.reward.std(ddof=1)  # Sample standard deviation
    avg_reward_sem = reward_std / math.sqrt(n)

    ci_margin = 1.96 * avg_reward_sem
    avg_reward_ci_95 = (
        max(0.0, avg_reward - ci_margin),  # Lower bound, clipped to [0, 1]
        min(1.0, avg_reward + ci_margin),  # Upper bound, clipped to [0, 1]
    )

    # Compute derived task complexity and calculate statistics
    derived_complexity = compute_derived_task_complexity(df)
    df = df.join(derived_complexity.rename("derived_complexity"), on="task_id")

    # Calculate complexity statistics
    derived_complexity_stats = DerivedComplexityStats(
        min=float(derived_complexity.min()),
        max=float(derived_complexity.max()),
        mean=float(derived_complexity.mean()),
        percentile_90=float(derived_complexity.quantile(0.90)),
        percentile_95=float(derived_complexity.quantile(0.95)),
        percentile_99=float(derived_complexity.quantile(0.99)),
    )

    # Calculate quartile boundaries
    quartile_labels = ["Q1", "Q2", "Q3", "Q4"]
    df["complexity_quartile"] = pd.qcut(
        df["derived_complexity"],
        q=4,
        labels=quartile_labels,
        duplicates="drop"
    )

    # Calculate success rate for each quartile
    success_rate_by_complexity_quartile = {}
    for quartile in quartile_labels:
        quartile_df = df[df["complexity_quartile"] == quartile]
        if len(quartile_df) > 0:
            success_rate_by_complexity_quartile[quartile] = (
                quartile_df["success"].sum() / len(quartile_df)
            )
        else:
            success_rate_by_complexity_quartile[quartile] = 0.0

    pass_hat_ks = {}
    for column in df_pass_hat_k.columns:
        if match := re.match(r"pass\^(\d+)", column):
            k = int(match.group(1))
            pass_hat_ks[k] = df_pass_hat_k[column].mean()
    avg_agent_cost = df.agent_cost.mean()
    return AgentMetrics(
        avg_reward=avg_reward,
        avg_reward_sem=avg_reward_sem,
        avg_reward_ci_95=avg_reward_ci_95,
        success_rate=success_rate,
        derived_complexity_stats=derived_complexity_stats,
        success_rate_by_complexity_quartile=success_rate_by_complexity_quartile,
        pass_hat_ks=pass_hat_ks,
        avg_num_tool_calls=avg_num_tool_calls,
        avg_agent_cost=avg_agent_cost,
    )


def display_metrics(metrics: AgentMetrics) -> None:
    print(f"🏆 Average reward: {metrics.avg_reward:.4f}")
    print(f"   SEM: {metrics.avg_reward_sem:.4f}")
    print(f"   95% CI: [{metrics.avg_reward_ci_95[0]:.4f}, {metrics.avg_reward_ci_95[1]:.4f}]")
    print(f"✅ Success rate: {metrics.success_rate:.2%}")
    print("📊 Derived Task Complexity (empirical):")
    stats = metrics.derived_complexity_stats
    print(f"   Min: {stats.min:.4f}, Max: {stats.max:.4f}, Mean: {stats.mean:.4f}")
    print(f"   90th percentile: {stats.percentile_90:.4f}, 95th percentile: {stats.percentile_95:.4f}, 99th percentile: {stats.percentile_99:.4f}")
    print("📊 Success rate by complexity quartile:")
    for quartile in ["Q1", "Q2", "Q3", "Q4"]:
        if quartile in metrics.success_rate_by_complexity_quartile:
            rate = metrics.success_rate_by_complexity_quartile[quartile]
            print(f"   {quartile} (least complex → most complex): {rate:.2%}")
    print("📈 Pass^k")
    for k, pass_hat_k in metrics.pass_hat_ks.items():
        print(f"  k={k}: {pass_hat_k}")
    print(f"🔧 Average number of tool calls: {metrics.avg_num_tool_calls:.2f}")
    print(f"💰 Average agent cost: {metrics.avg_agent_cost}")


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True)
    args = parser.parse_args()
    results = Results.load(Path(args.results))
    metrics = compute_metrics(results)
    display_metrics(metrics)
