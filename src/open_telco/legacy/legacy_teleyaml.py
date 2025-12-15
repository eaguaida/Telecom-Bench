import json
import re
from textwrap import dedent
from typing import Any

from dotenv import load_dotenv
from inspect_ai import Task, task
from inspect_ai.dataset import FieldSpec, hf_dataset
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, Model, get_model
from inspect_ai.scorer import Score, Target, mean, scorer, stderr
from inspect_ai.solver import generate, system_message

load_dotenv()


TELEYAML_JUDGE_SYSTEM_PROMPT = dedent("""
You are a meticulous and impartial AI evaluator specializing in network engineering and technical explanations. 
Your ONLY task is to assess an AI's response against the provided question and ground truth.
EVALUATION CRITERIA - Each criterion must be scored on a scale of 1 to 10:
- overall_score: Overall quality of the response (1 = very poor, 10 = excellent)
- confidence: Your confidence in this evaluation (1 = uncertain, 10 = highly certain)
- factual_correctness: Accuracy compared to the ground truth (1 = false, 10 = perfectly accurate)
- relevance: Relevance to the given question (1 = irrelevant, 10 = perfectly relevant)
- completeness: Coverage of necessary details (1 = missing key info, 10 = fully sufficient)
- conciseness: Appropriate length (1 = far too brief, 5 = balanced, 10 = far too wordy)
- is_safe: Boolean - True if response is safe and free from harmful/biased content, False otherwise
- reasoning: A short explanation (1–3 sentences) justifying your scores
RESPONSE FORMAT: Return ONLY a valid JSON object with this exact schema:
{
  "overall_score": <integer 1–10>,
  "confidence": <integer 1–10>,
  "factual_correctness": <integer 1–10>,
  "relevance": <integer 1–10>,
  "completeness": <integer 1–10>,
  "conciseness": <integer 1–10>,
  "is_safe": <true/false>,
  "reasoning": "<string>"
}
Do not include markdown, code fences, or any extra text.
""")

SYSTEM_PROMPT = dedent("""You are a network engineering expert. Analyze the provided context and question, and provide a technically accurate, concise, and complete answer.""")


def parse_score(output: str) -> dict[str, Any]:
    """Parse JSON score from judge output."""
    try:
        match = re.search(r"\{.*\}", output, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(output)
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return {"overall_score": 0, "reasoning": f"Failed to parse JSON: {output}"}


@scorer(metrics=[mean(), stderr()])
def legacy_judge_scorer(model: str | Model) -> Any:
    """Scorer that uses an LLM to judge the output."""

    async def score(state: Any, target: Target) -> Score:
        result = await get_model(model).generate([
            ChatMessageSystem(content=TELEYAML_JUDGE_SYSTEM_PROMPT),
            ChatMessageUser(content=f"\n[Question]\n{state.input}\n\n[Ground Truth]\n{target.text}\n\n[Model Response]\n{state.output.completion}\n")
        ])

        return Score(
            value=(parsed := parse_score(result.completion)).get("overall_score", 0),
            answer=result.completion,
            explanation=parsed.get("reasoning", str(parsed)),
            metadata=parsed
        )

    return score


@task
def legacy_teleyaml() -> Task:
    """Defines the legacy teleyaml task."""
    dataset = hf_dataset(
        "otellm/gsma-sample-data",
        name="teleyaml",
        split="test",
        sample_fields=FieldSpec(
            input="Question",
            target="Answer",
            metadata=["Main Category", "Category", "Context"],
        ),
    )

    return Task(
        dataset=list(dataset),
        solver=[system_message(SYSTEM_PROMPT), generate()],
        scorer=legacy_judge_scorer(model="openrouter/openai/gpt-oss-120b"),
        metrics=[mean(), stderr()],
    )
