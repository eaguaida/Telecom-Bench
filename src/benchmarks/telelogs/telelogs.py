import sys
from pathlib import Path
from textwrap import dedent

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from inspect_ai import Task, task  # noqa: E402
from inspect_ai.agent import Agent, agent, react, AgentPrompt  # noqa: E402
from inspect_ai.dataset import FieldSpec  # noqa: E402
from inspect_ai.scorer import pattern  # noqa: E402
from scripts.utils import load_env, load_huggingface_dataset  # type: ignore # noqa: E402

# Load environment variables
load_env()

INSTRUCTIONS = dedent(
    """
    You are a telecommunications operator analysing system logs.
    Think step by step and keep reasoning concise.
    When you reach a conclusion, call {submit}() with only the final answer formatted as \\boxed{C<value>} where <value> is the numeric identifier.
    """
)


@agent
def telelogs_agent(attempts: int = 1) -> Agent:
    return react(
        description="Telecommunications log analyst.",
        prompt=AgentPrompt(
            instructions=INSTRUCTIONS,
        ),
        tools=[],
        attempts=attempts,
    )


@task
def telelogs() -> Task:
    dataset = load_huggingface_dataset(
        "netop/TeleLogs",
        sample_fields=FieldSpec(
            input="question",
            target="answer",
        )
    )

    return Task(
        dataset=dataset,
        solver=telelogs_agent(),
        scorer=pattern(r"\\boxed\{\{?(.+?)\}?\}"),
    )

