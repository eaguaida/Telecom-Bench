from inspect_ai import Task, task
from inspect_ai.dataset import FieldSpec
from inspect_ai.scorer import accuracy, stderr, model_graded_fact
from inspect_ai.solver import system_message, generate
from open_telco.scripts.utils import load_env, load_huggingface_dataset
from textwrap import dedent
from pathlib import Path

load_env()

AGENT_SYSTEM_PROMPT = dedent(""" 
You are an expert 5G Core Network Engineer and Configuration Specialist.
You are assisting a user with {Main Category} by converting requests into
server-side YAML configurations for {Category}.

<context>
{Context}
</context>

Your response must be valid YAML.
""")


def get_rubric(category: str, rubrics_dir: Path) -> str:
    """Load rubric content based on category."""
    if category == "AMF Configuration":
        rubric_path = rubrics_dir / "amf_configuration.txt"
    elif category == "Slice Deployment":
        rubric_path = rubrics_dir / "slice_deployment.txt"
    elif category == "UE Provisioning":
        rubric_path = rubrics_dir / "ue_provisioning.txt"
    else:
        return "NO RUBRIC FOUND"

    return rubric_path.read_text() if rubric_path.exists() else "NO RUBRIC FOUND"


@task
def teleyaml() -> Task:
    dataset = load_huggingface_dataset(
        "otellm/gsma-sample-data",
        name="teleyaml",
        sample_fields=FieldSpec(
            input="Question",
            target="Answer",
            metadata=["Main Category", "Category", "Context"],
        ),
    )

    rubrics_dir = Path(__file__).parent / "judge_rubrics"

    dataset_samples = list(dataset)
    for sample in dataset_samples:
        category = sample.metadata.get("Category")
        sample.metadata["rubric"] = get_rubric(category, rubrics_dir)

    template = dedent("""
        You are comparing a submitted answer to an expert answer on a given question. Here is the data:
        [BEGIN DATA]
        ************
        [Task]: {question}
        ************
        [Rubric]: {rubric}
        ************
        [Submission]: {answer}
        ************
        [END DATA]

        Evaluate the submission based on the rubric.

        {instructions}
    """).strip()

    instructions = dedent("""
        Based on the rubric, reply with 'Score: $NUMBER' (without quotes) where NUMBER is the final score. Then, end with your answer formatted as 'GRADE: $LETTER' (without quotes) where LETTER is one of C{partial_letter}I.  Please choose ONE option for the grade: either "C" for correct answers, {partial_prompt}or "I" for incorrect answers.
        For example, after reviewing a correct answer you might write 'GRADE: C' or after reviewing an incorrect answer you might write 'GRADE: I'.

        First, write out in a step by step manner your reasoning about the criterion to be sure that your conclusion is correct. Avoid simply stating the correct answers at the outset. Then, end with your answer formatted as 'GRADE: $LETTER' (without quotes) where LETTER is one of C{partial_letter}I. 
    """).strip()

    return Task(
        dataset=dataset_samples,
        solver=[system_message(AGENT_SYSTEM_PROMPT), generate()],
        scorer=model_graded_fact(
            template=template,
            instructions=instructions,
            partial_credit=True,
            model="openrouter/openai/gpt-4o",
        ),
        metrics=[accuracy(), stderr()],
    )