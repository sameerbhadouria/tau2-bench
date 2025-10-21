#!/usr/bin/env python3
"""
Script to recompute rewards in LLM grader simulation results using stored confidence scores.

This script reads an existing simulation results file and updates the reward values
to use the confidence scores instead of binary 0/1 values.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any


def recompute_reward(simulation: Dict[str, Any]) -> tuple[float, float]:
    """
    Recompute the reward for a simulation using confidence score.

    Args:
        simulation: The simulation data dictionary

    Returns:
        Tuple of (old_reward, new_reward)
    """
    reward_info = simulation['reward_info']
    old_reward = reward_info['reward']

    # Extract success and confidence from info
    info = reward_info.get('info', {})
    success = info.get('success', False)
    confidence = info.get('confidence', 1.0)

    # Compute new reward: confidence if success, 0.0 otherwise
    new_reward = confidence if success else 0.0

    return old_reward, new_reward


def update_simulation_rewards(file_path: Path, backup: bool = True) -> None:
    """
    Update rewards in the simulation results file.

    Args:
        file_path: Path to the simulation results JSON file
        backup: Whether to create a backup of the original file
    """
    print(f"Loading simulation results from: {file_path}")

    # Load the data
    with open(file_path, 'r') as f:
        data = json.load(f)

    print(f"Found {len(data['simulations'])} simulations")

    # Create backup if requested
    if backup:
        backup_path = file_path.with_suffix('.json.backup')
        print(f"Creating backup at: {backup_path}")
        with open(backup_path, 'w') as f:
            json.dump(data, f, indent=2)

    # Recompute rewards
    changes = []
    for i, simulation in enumerate(data['simulations']):
        old_reward, new_reward = recompute_reward(simulation)

        if old_reward != new_reward:
            changes.append({
                'sim_id': simulation['id'],
                'task_id': simulation['task_id'],
                'trial': simulation['trial'],
                'old_reward': old_reward,
                'new_reward': new_reward
            })

        # Update the reward
        simulation['reward_info']['reward'] = new_reward

        # Also update reward_breakdown if it uses NL_ASSERTION
        if 'reward_breakdown' in simulation['reward_info']:
            if 'NL_ASSERTION' in simulation['reward_info']['reward_breakdown']:
                simulation['reward_info']['reward_breakdown']['NL_ASSERTION'] = new_reward

    # Print summary
    print(f"\nReward Changes Summary:")
    print(f"  Total simulations: {len(data['simulations'])}")
    print(f"  Simulations with changed rewards: {len(changes)}")
    print(f"  Simulations unchanged: {len(data['simulations']) - len(changes)}")

    if changes:
        print(f"\nSample of changed rewards (first 10):")
        for change in changes[:10]:
            print(f"  Task {change['task_id']}, Trial {change['trial']}: "
                  f"{change['old_reward']:.2f} -> {change['new_reward']:.2f}")

        if len(changes) > 10:
            print(f"  ... and {len(changes) - 10} more")

    # Calculate statistics
    old_rewards = [c['old_reward'] for c in changes] + [s['reward_info']['reward'] for s in data['simulations'] if s['reward_info']['reward'] not in [c['old_reward'] for c in changes]]
    new_rewards = [s['reward_info']['reward'] for s in data['simulations']]

    print(f"\nReward Statistics:")
    print(f"  Old rewards - Min: {min([r['old_reward'] for r in changes] or [0]):.2f}, "
          f"Max: {max([r['old_reward'] for r in changes] or [1]):.2f}, "
          f"Mean: {sum(old_rewards) / len(old_rewards):.3f}")
    print(f"  New rewards - Min: {min(new_rewards):.2f}, "
          f"Max: {max(new_rewards):.2f}, "
          f"Mean: {sum(new_rewards) / len(new_rewards):.3f}")

    # Save updated data
    print(f"\nSaving updated results to: {file_path}")
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

    print("✓ Done!")


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        file_path = Path(sys.argv[1])
    else:
        # Default to the known file
        file_path = Path("data/simulations/llm_grader_grok_airline_all.json")

    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    # Ask for confirmation
    response = input(f"This will update rewards in {file_path}. Continue? [y/N]: ")
    if response.lower() != 'y':
        print("Cancelled.")
        sys.exit(0)

    update_simulation_rewards(file_path, backup=True)


if __name__ == "__main__":
    main()

