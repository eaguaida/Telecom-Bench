import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from inspect_ai import Task, task  # noqa: E402
from inspect_ai.dataset import Sample  # noqa: E402
from inspect_ai.scorer import choice  # noqa: E402
from inspect_ai.solver import multiple_choice  # noqa: E402
from scripts.utils import load_env, load_huggingface_dataset  # type: ignore # noqa: E402

# Load environment variables
load_env()


def teleqna_record_to_sample(record):
    """Convert TeleQnA record to Inspect Sample with choices."""
    return Sample(
        input=record["question"],
        choices=record["choices"],
        target=chr(65 + record["answer"])
    )


@task
def teleqna() -> Task:
    """TeleQnA benchmark - Multiple choice telecommunications knowledge assessment."""
    dataset = load_huggingface_dataset(
        "netop/TeleQnA",
        sample_fields=teleqna_record_to_sample
    )
    
    return Task(
        dataset=dataset,
        solver=multiple_choice(cot=True),
        scorer=choice()
    )

