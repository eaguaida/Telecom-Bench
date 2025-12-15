from textwrap import dedent

from dotenv import load_dotenv 
from inspect_ai import Task, task
from inspect_ai.dataset import FieldSpec, hf_dataset
from inspect_ai.scorer import accuracy, stderr
from inspect_ai.solver import system_message, generate

from open_telco.teleyaml.judge import judge, assign_rubrics  

load_dotenv()


SYSTEM_PROMPT = dedent(""" 
You are an expert 5G Core Network Engineer and Configuration Specialist.
You are assisting a user with {Main Category} by converting requests into
server-side YAML configurations for {Category}.

<context>
{Context}
</context>

Your response must be valid YAML.
""")


@task
def teleyaml() -> Task:
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

    judge_models = [
        "openrouter/openai/gpt-oss-120b",
    ]

    return Task(
        dataset=assign_rubrics(list(dataset)),
        solver=[system_message(SYSTEM_PROMPT), generate()],
        scorer=judge(model=judge_models),
        metrics=[accuracy(), stderr()],
    )