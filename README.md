# TeleMath Benchmark

Evaluation benchmark for telecommunications mathematical problems using the [netop/TeleMath](https://huggingface.co/datasets/netop/TeleMath) dataset.

## Overview

This benchmark uses the Inspect AI framework to evaluate language models on telecommunications domain problems. The evaluation uses a ReAct agent equipped with bash and python tools to solve mathematical and technical problems.

## Tasks

The benchmark provides three tasks:

- `telemath`: TeleMath benchmark with configurable difficulty levels (basic, intermediate, advanced, or full)
- `teleqna`: TeleQnA benchmark for multiple choice telecommunications knowledge assessment
- `telelogs`: TeleLogs benchmark for log interpretation and incident response

## Usage

Run the complete TeleMath benchmark:

```bash
inspect eval src/benchmarks/telemath/telemath.py
```

Run TeleMath with a specific difficulty level:

```bash
inspect eval src/benchmarks/telemath/telemath.py -T difficulty=basic
inspect eval src/benchmarks/telemath/telemath.py -T difficulty=intermediate
inspect eval src/benchmarks/telemath/telemath.py -T difficulty=advanced
```

Run TeleQnA benchmark:

```bash
inspect eval src/benchmarks/teleqna/teleqna.py
```

Run TeleLogs benchmark:

```bash
inspect eval src/benchmarks/telelogs/telelogs.py
```

With a specific model:

```bash
inspect eval src/benchmarks/telemath/telemath.py --model openai/gpt-4o
```

Limit samples for testing:

```bash
inspect eval src/benchmarks/telemath/telemath.py --limit 10
```

## Components

### Agent

The `telecom_agent` is a ReAct agent configured with:
- **Tools**: `bash()` and `python()` for computational tasks
- **Attempts**: 3 attempts to solve each problem
- **Description**: Telecommunication Operator

### Dataset

Loaded from Hugging Face: `netop/TeleMath` (test split)

**Field mapping**:
- `input`: question
- `target`: answer
- `metadata`: category, tags, difficulty

### Scorer

Uses `match()` scorer for exact answer matching.

## References

- [Inspect AI Documentation](https://inspect.aisi.org.uk/)
- [Inspect AI Datasets](https://inspect.aisi.org.uk/datasets.html)
- [Inspect AI ReAct Agent](https://inspect.aisi.org.uk/react-agent.html)
- [TeleMath Dataset](https://huggingface.co/datasets/netop/TeleMath)

