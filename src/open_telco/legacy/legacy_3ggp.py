import re
from typing import Any

from dotenv import load_dotenv
from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import generate

load_dotenv()


def parse_working_group(text: str) -> str:
    """
    Parses the working group from the text.
    It looks for patterns like SA1, RAN2, CT3, etc.
    """
    if not text:
        return ""
    # Look for common 3GPP WG patterns: 2-4 letters followed by digits
    # e.g., SA1, RAN2, CT3, GERAN, SA3-LI
    match = re.search(r"([A-Z]+[\d]+(?:-[A-Z]+)?)", text.upper())
    if match:
        return match.group(1)
    
    # Fallback: just return clean text if no specific pattern found, 
    # assuming the model outputs just the WG name or similar.
    return text.strip()


@scorer(metrics=[accuracy(), stderr()])
def tsg_scorer() -> Any:
    """
    Scorer that evaluates if the predicted working group matches the ground truth.
    """
    async def score(state: Any, target: Target) -> Score:
        ans = parse_working_group(state.output.completion)
        gt = parse_working_group(target.text)
        is_correct = ans.lower() == gt.lower()
        return Score(
            value=1.0 if is_correct else 0.0,
            answer=ans,
            explanation=f"Predicted: {ans}, Ground Truth: {gt}. Match: {is_correct}"
        )
    
    return score


def record_to_sample(record: dict[str, Any]) -> Sample:
    return Sample(
        input=record["question"],
        target=record["answer"],
    )


@task
def old_three_gpp() -> Task:
    dataset = hf_dataset(
        "otellm/gsma-sample-data",
        name="3gpp_tsg",
        sample_fields=record_to_sample,
        split="test",
    )

    return Task(
        dataset=dataset,
        solver=[generate()],
        scorer=tsg_scorer(),
        metrics=[accuracy(), stderr()]
    )
