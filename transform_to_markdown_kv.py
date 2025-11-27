#!/usr/bin/env python3
"""Transform TeleLogs dataset to Markdown-KV format.

This script reads the processed TeleLogs dataset (JSON format) and transforms
the question field from pipe-delimited tables to Markdown-KV format for improved
LLM comprehension.

Usage:
    python transform_to_markdown_kv.py [--input INPUT_PATH] [--output OUTPUT_PATH]

The script:
1. Reads the JSON dataset (default: telelogs/telelogs_test.json)
2. Transforms each question from pipe-delimited to Markdown-KV format
3. Saves the transformed dataset to a new JSON file
4. Optionally validates the transformation quality
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Dict, List
from huggingface_hub import HfApi, login

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from telecom_bench.prompt_transformer import transform_prompt, validate_transformation


def transform_dataset(input_path: str, output_path: str, validate: bool = True) -> Dict[str, any]:
    """Transform dataset from pipe-delimited to Markdown-KV format.

    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSON file
        validate: Whether to validate transformations

    Returns:
        Dictionary with transformation statistics
    """
    print(f"Loading dataset from: {input_path}")

    # Read input JSON
    with open(input_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} samples")

    # Transform each sample
    transformed_dataset = []
    validation_results = []

    print("\nTransforming samples...")
    for i, sample in enumerate(dataset):
        original_question = sample['question']

        # Transform the question field
        transformed_question = transform_prompt(original_question)

        # Create transformed sample
        transformed_sample = {
            'question': transformed_question,
            'answer': sample['answer'],
            'choices': sample['choices']
        }

        transformed_dataset.append(transformed_sample)

        # Validate if requested
        if validate:
            checks = validate_transformation(original_question, transformed_question)
            validation_results.append(checks)

        # Progress indicator
        if (i + 1) % 100 == 0:
            print(f"  Transformed {i + 1}/{len(dataset)} samples")

    print(f"  Transformed {len(dataset)}/{len(dataset)} samples")

    # Save transformed dataset
    print(f"\nSaving transformed dataset to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(transformed_dataset, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(transformed_dataset)} samples")

    # Compute validation statistics
    stats = {
        'total_samples': len(dataset),
        'transformed_samples': len(transformed_dataset),
    }

    if validate and validation_results:
        print("\n=== Validation Results ===")
        validation_summary = {}

        for check_name in validation_results[0].keys():
            passed = sum(1 for result in validation_results if result[check_name])
            percentage = (passed / len(validation_results)) * 100
            validation_summary[check_name] = {
                'passed': passed,
                'failed': len(validation_results) - passed,
                'percentage': percentage
            }
            print(f"  {check_name}: {passed}/{len(validation_results)} ({percentage:.1f}%)")

        stats['validation'] = validation_summary

    return stats


def upload_to_huggingface(folder_path: str, repo_id: str = "eaguaida/telelogs_markdown") -> bool:
    """Upload transformed dataset to HuggingFace.

    Args:
        folder_path: Path to the folder containing the dataset files
        repo_id: HuggingFace repository ID (default: eaguaida/telelogs_markdown)

    Returns:
        True if upload succeeded, False otherwise
    """
    # Get HuggingFace token from environment
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Warning: HF_TOKEN environment variable not set.")
        print("Please set it with your HuggingFace token to upload:")
        print("  export HF_TOKEN=your_token_here")
        return False

    try:
        # Login to HuggingFace
        print(f"\nLogging in to HuggingFace...")
        login(token=hf_token, add_to_git_credential=False)

        # Initialize API
        api = HfApi(token=hf_token)

        print(f"Uploading to {repo_id}...")
        api.upload_folder(
            folder_path=folder_path,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Upload Markdown-KV transformed TeleLogs dataset"
        )

        print(f"\n✓ Successfully uploaded to https://huggingface.co/datasets/{repo_id}")
        return True

    except Exception as e:
        print(f"\n✗ Error uploading to HuggingFace: {e}")
        return False


def show_sample_comparison(input_path: str, sample_index: int = 0):
    """Show a side-by-side comparison of original and transformed samples.

    Args:
        input_path: Path to input JSON file
        sample_index: Index of sample to display
    """
    # Read input JSON
    with open(input_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    if sample_index >= len(dataset):
        print(f"Error: Sample index {sample_index} out of range (max: {len(dataset) - 1})")
        return

    sample = dataset[sample_index]
    original_question = sample['question']
    transformed_question = transform_prompt(original_question)

    print("=" * 80)
    print(f"SAMPLE #{sample_index}")
    print("=" * 80)

    print("\n--- ORIGINAL (Pipe-Delimited) ---")
    print(original_question[:1000])
    if len(original_question) > 1000:
        print(f"\n... (truncated, total length: {len(original_question)} chars)")

    print("\n--- TRANSFORMED (Markdown-KV) ---")
    print(transformed_question[:1000])
    if len(transformed_question) > 1000:
        print(f"\n... (truncated, total length: {len(transformed_question)} chars)")

    print("\n--- VALIDATION ---")
    checks = validate_transformation(original_question, transformed_question)
    for check_name, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {check_name}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Transform TeleLogs dataset to Markdown-KV format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transform default dataset
  python transform_to_markdown_kv.py

  # Transform and upload to HuggingFace
  python transform_to_markdown_kv.py --upload

  # Transform with custom repository
  python transform_to_markdown_kv.py --upload --repo-id myuser/my-telelogs

  # Transform specific input/output
  python transform_to_markdown_kv.py --input data.json --output data_mkv.json

  # Show sample comparison
  python transform_to_markdown_kv.py --sample 0

  # Skip validation (faster)
  python transform_to_markdown_kv.py --no-validate
        """
    )

    parser.add_argument(
        '--input',
        default='telelogs/telelogs_test.json',
        help='Input JSON file path (default: telelogs/telelogs_test.json)'
    )
    parser.add_argument(
        '--output',
        default='telelogs/telelogs_test_mkv.json',
        help='Output JSON file path (default: telelogs/telelogs_test_mkv.json)'
    )
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip validation checks (faster processing)'
    )
    parser.add_argument(
        '--sample',
        type=int,
        metavar='INDEX',
        help='Show comparison for a specific sample index (does not transform dataset)'
    )
    parser.add_argument(
        '--upload',
        action='store_true',
        help='Upload transformed dataset to HuggingFace after transformation'
    )
    parser.add_argument(
        '--repo-id',
        default='eaguaida/telelogs_markdown',
        help='HuggingFace repository ID for upload (default: eaguaida/telelogs_markdown)'
    )

    args = parser.parse_args()

    # If --sample is specified, show comparison and exit
    if args.sample is not None:
        show_sample_comparison(args.input, args.sample)
        return

    # Validate input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        print("\nPlease run 'python extract_telelogs.py' first to download the dataset.")
        sys.exit(1)

    # Create output directory if needed
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Transform dataset
    try:
        stats = transform_dataset(
            args.input,
            args.output,
            validate=not args.no_validate
        )

        print("\n=== Transformation Complete ===")
        print(f"Input:  {args.input}")
        print(f"Output: {args.output}")
        print(f"Samples: {stats['transformed_samples']}/{stats['total_samples']}")

        if 'validation' in stats:
            # Check for any failing validations
            failed_checks = [
                check for check, data in stats['validation'].items()
                if data['percentage'] < 100
            ]

            if failed_checks:
                print(f"\n⚠ Warning: {len(failed_checks)} validation checks have failures")
                print("  Run with --sample <index> to inspect specific samples")
            else:
                print("\n✓ All validation checks passed!")

        # Upload to HuggingFace if requested
        if args.upload:
            output_dir = Path(args.output).parent
            upload_success = upload_to_huggingface(
                folder_path=str(output_dir),
                repo_id=args.repo_id
            )
            if not upload_success:
                print("\n⚠ Upload failed. You can manually upload later with:")
                print(f"  python transform_to_markdown_kv.py --upload --repo-id {args.repo_id}")

    except Exception as e:
        print(f"\nError during transformation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
