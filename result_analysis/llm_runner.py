

import os

from dotenv import load_dotenv

from xai_sdk import Client
from xai_sdk.chat import system, user


res = load_dotenv()
if not res:
    print("No .env file found")


class LLMRunner:

    def __init__(self) -> None:
        self._client = Client(
            api_key=os.getenv("XAI_API_KEY"),
            timeout=3600, # Override default timeout with longer timeout for reasoning models
        )


    def run_completion(
        self,
        system_message: str,
        user_message: str,
        model: str = "grok-4-fast-reasoning"
    ) -> str:
        chat = self._client.chat.create(
            model=model,
            messages=[
                system(system_message),
                user(user_message)
            ],
        )

        response = chat.sample()

        return response.content


    # print("Final Response:")
    # print(response.content)

    # print("Number of completion tokens:")
    # print(response.usage.completion_tokens)

    # print("Number of reasoning tokens:")
    # print(response.usage.reasoning_tokens)
