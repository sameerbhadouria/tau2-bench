from collections import defaultdict
import json
from typing import Dict, List, Set, Optional
from tau2.data_model.simulation import Results, SimulationRun
from tau2.data_model.tasks import Task


class TaskConversationFetcher:

    def __init__(self, results: Results):
        self._tasks_by_id: Dict[str, Task] = {task.id: task for task in results.tasks}

        # Map: task_id -> trial -> list of SimulationRuns
        self._task_id_to_trial_simulations: Dict[str, Dict[int, List[SimulationRun]]] = defaultdict(lambda: defaultdict(list))

        for sim in results.simulations:
            self._task_id_to_trial_simulations[sim.task_id][sim.trial].append(sim)

    def get_all_simulations(self, task_id: int, trial_number: int) -> List[SimulationRun]:
        """
        Get all simulation runs for a specific task and trial (handles duplicates).

        Args:
            task_id: The task ID
            trial_number: The trial number

        Returns:
            List of SimulationRun objects (empty if none found)
        """
        task_id_str = str(task_id)
        return self._task_id_to_trial_simulations.get(task_id_str, {}).get(trial_number, [])

    def get_simulation(self, task_id: int, trial_number: int, index: int = -1) -> Optional[SimulationRun]:
        """
        Get a specific simulation run for a task and trial.

        Args:
            task_id: The task ID
            trial_number: The trial number
            index: Which simulation to get if multiple exist (default: -1 for most recent)

        Returns:
            SimulationRun object or None if not found
        """
        simulations = self.get_all_simulations(task_id, trial_number)
        if not simulations:
            return None
        return simulations[index]

    # Note you can also use the `tau2 view` cli command to view the simulation run details for a given task and trial number
    # Vibe coded by Cursor and verified using the tau2 view cli command
    def print_simulation_run_details(self, task_id: int, trial_number: int, index: int = -1):
        """
        Retrieve and display task details and conversation for a specific task and trial.

        Args:
            task_id: The task ID (e.g., 0, 1, 2, ...)
            trial_number: The trial number (e.g., 0, 1, 2, ...)
            index: Which simulation to display if multiple exist (default: -1 for most recent)
        """


        # Use pre-computed lookup - O(1)
        task_id_str = str(task_id)
        task = self._tasks_by_id.get(task_id_str)

        if task is None:
            print(f"❌ Task {task_id} not found")
            return

        # Get all simulations for this task/trial
        simulations = self.get_all_simulations(task_id, trial_number)

        if not simulations:
            print(f"❌ Simulation for task {task_id}, trial {trial_number} not found")
            return

        # Warn if multiple simulations exist
        if len(simulations) > 1:
            print(f"⚠️  Found {len(simulations)} simulations for task {task_id}, trial {trial_number}")
            print(f"    Displaying simulation {index} (most recent: -1)")
            print(f"    Simulation IDs: {[s.id for s in simulations]}\n")

        simulation = simulations[index]

        # Display Task Details
        print("=" * 80)
        print("📋 TASK DETAILS")
        print("=" * 80)
        print(f"Task ID: {task.id}")

        if task.description and task.description.purpose:
            print(f"\nPurpose: {task.description.purpose}")

        print(f"\n📝 User Instructions:")
        if isinstance(task.user_scenario.instructions, str):
            print(task.user_scenario.instructions)
        else:
            print(f"  Domain: {task.user_scenario.instructions.domain}")
            print(f"  Reason for call: {task.user_scenario.instructions.reason_for_call}")
            if task.user_scenario.instructions.known_info:
                print(f"  Known info: {task.user_scenario.instructions.known_info}")
            print(f"  Task instructions: {task.user_scenario.instructions.task_instructions}")

        if task.evaluation_criteria:
            if task.evaluation_criteria.actions:
                print(f"\n✅ Expected Actions ({len(task.evaluation_criteria.actions)}):")
                for i, action in enumerate(task.evaluation_criteria.actions):
                    print(f"  {i+1}. {action.name}")
                    if action.arguments:
                        for key, value in action.arguments.items():
                            print(f"     - {key}: {value}")

        # Display Simulation Details
        print("\n" + "=" * 80)
        print("🎭 SIMULATION DETAILS")
        print("=" * 80)
        print(f"Task ID: {simulation.task_id} | Trial: {simulation.trial}")
        print(f"Duration: {simulation.duration:.2f}s | Termination: {simulation.termination_reason}")

        if simulation.agent_cost:
            print(f"Agent Cost: ${simulation.agent_cost:.4f}", end="")
        if simulation.user_cost:
            print(f" | User Cost: ${simulation.user_cost:.4f}")
        else:
            print()

        if simulation.reward_info:
            marker = "✅" if simulation.reward_info.reward >= 1.0 else "❌"
            print(f"\nReward: {marker} {simulation.reward_info.reward:.2f}")

            if simulation.reward_info.action_checks:
                print(f"\n📊 Action Checks:")
                for i, check in enumerate(simulation.reward_info.action_checks):
                    status = "✅" if check.action_match else "❌"
                    print(f"  {status} {check.action.name}")

        # Display Conversation
        print("\n" + "=" * 80)
        print("💬 CONVERSATION")
        print("=" * 80)

        for i, msg in enumerate(simulation.messages):
            # Skip system messages for brevity
            if msg.role == "system":
                continue

            print(f"\n[Turn {msg.turn_idx}] {msg.role.upper()}:")

            if msg.content:
                # Truncate very long messages
                content = msg.content
                if len(content) > 500:
                    content = content[:500] + "... [truncated]"
                print(f"  {content}")

            # Show tool calls
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool in msg.tool_calls:
                    print(f"  🔧 Tool: {tool.name}")
                    args_str = json.dumps(tool.arguments, indent=4)
                    # Truncate long arguments
                    if len(args_str) > 300:
                        args_str = args_str[:300] + "... [truncated]"
                    print(f"     Args: {args_str}")

            # Show tool results (for tool messages)
            if msg.role == "tool":
                if msg.error:
                    print(f"  ⚠️ ERROR in response")

        print("\n" + "=" * 80)

