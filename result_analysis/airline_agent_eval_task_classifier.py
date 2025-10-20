import json
from pathlib import Path
from typing import Dict, List, Optional
from tau2.data_model.tasks import Task

from llm_runner import LLMRunner


class AirlineAgentEvalTaskClassifier:

    SYSTEM_PROMPT_TEMPLATE = """
    You are an expert classifier for airline customer service agent tasks. Your job is to categorize
    evaluation tasks into specific categories based on what aspect of agent behavior they're testing.

    ## Domain Context

    Airline agents help users with:
    - Booking new flight reservations
    - Modifying existing reservations (flights, cabin, baggage, passengers)
    - Cancelling reservations with appropriate refunds
    - Handling complaints and issuing compensation

    Agents must strictly follow the Airline Agent Policy defined below:
    <policy>
    {airline_policy}
    </policy>

    \n\n

    Classify each task into one of the following base categories:
    - Booking Creation
    - Flight Modification
    - Cancellation Policy
    - Compensation/Complaints
    - Membership & Benefits
    - Insurance Issues
    - Complex Multi-Transaction

    Further if the task is about user tricking the agent append " Tricking" to the categories above.
    If the task can't be classified into any of the base categories append " Other" to the categories above.

    Each task has a purpose, reason for call, instructions, action names, and nl assertions.

    Output the classification as a JSON object with the following fields:
    - category: the category of the task
    - score: the score of the classification

    Example output:
    {{
        "category": "Booking Creation",
        "score": 0.95
    }}

    """

    USER_PROMPT_TEMPLATE = """
    Classify the following airline agent task into a evaluation category:

    <purpose>
    {purpose}
    </purpose>

    <reason_for_call>
    {reason_for_call}
    </reason_for_call>

    <instructions>
    {instructions}
    </instructions>

    <action_names>
    {action_names}
    </action_names>

    <nl_assertions>
    {nl_assertions}
    </nl_assertions>

    """

    def __init__(self, domain_dir: str):
        assert Path(domain_dir).is_dir()

        self.system_prompt = self._create_system_prompt(domain_dir)
        self._task_file_path = f"{domain_dir}/tasks.json"
        self._grok_runner = LLMRunner()
        self._task_id_category: Dict[str, str] = {}


    def _create_system_prompt(self, domain_dir: Path) -> str:
        airline_policy_path =  f"{domain_dir}/policy.md"
        with open(airline_policy_path, "r") as file:
            policy = file.read()
        return self.SYSTEM_PROMPT_TEMPLATE.format(airline_policy=policy)


    def _create_user_prompt(self, task: Task) -> str:
        purpose = task.description.purpose
        reason_for_call = task.user_scenario.instructions.reason_for_call
        instructions = task.user_scenario.instructions.task_instructions

        action_names = [action.name for action in task.evaluation_criteria.actions]
        expected_actions = ", ".join(action_names) if action_names else "None"

        nl_assertions = task.evaluation_criteria.nl_assertions
        expected_behaviors = "\n- ".join(nl_assertions) if nl_assertions else "None specified"
        if nl_assertions:
            expected_behaviors = "- " + expected_behaviors

        return self.USER_PROMPT_TEMPLATE.format(
            purpose=purpose,
            reason_for_call=reason_for_call,
            instructions=instructions,
            action_names=expected_actions,
            nl_assertions=expected_behaviors
        )

    def _read_tasks_from_json(self) -> List[Task]:
        with open(self._task_file_path, "r") as file:
            tasks = json.load(file)
        return [Task.model_validate(task) for task in tasks]

    def run(self, save_to_path: Optional[str] = None) -> Dict[str, str]:
        tasks: List[Task] = self._read_tasks_from_json()
        for task in tasks:
            if task.id not in self._task_id_category:
                user_prompt = self._create_user_prompt(task)
                response: str = self._grok_runner.run_completion(
                    system_message=self.system_prompt,
                    user_message=user_prompt,
                )

                resp = json.loads(response)

                # print(resp)
                self._task_id_category[task.id] = resp["category"]

        if save_to_path:
            Path(save_to_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_to_path, "w") as f:
                json.dump(self._task_id_category, f, indent=4)

        return self._task_id_category

    @property
    def task_id_category(self):
        return self._task_id_category

