"""
Script to extract and transform the TeleLogs dataset from HuggingFace
"""
import os
import re
import sys
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import login
import json
import pandas as pd
import argparse

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from telecom_bench.prompt_transformer import transform_prompt

# Get HuggingFace token from environment variable
# Only check when running as main script, not when importing functions
HF_TOKEN = os.environ.get("HF_TOKEN")

def ensure_hf_login():
    """Ensure HuggingFace authentication is set up."""
    if not HF_TOKEN:
        raise ValueError(
            "HF_TOKEN environment variable not set. "
            "Please set it with your HuggingFace token:\n"
            "export HF_TOKEN=your_token_here"
        )

    # Login to HuggingFace
    try:
        login(token=HF_TOKEN, add_to_git_credential=False)
        print("Successfully logged in to HuggingFace")
    except Exception as e:
        print(f"Warning: Could not login to HuggingFace: {e}")

def process_split(split_data, split_name: str, apply_markdown_kv: bool = False, suffix: str = ""):
    """
    Process a single split of the dataset

    Args:
        split_data: The dataset split to process
        split_name: Name of the split (e.g., 'train', 'test')
        apply_markdown_kv: If True, transform questions to Markdown-KV format
        suffix: Additional suffix for output files (e.g., '_plain')

    Returns:
        Processed dataframe
    """
    print(f"\n{'='*60}")
    print(f"Processing {split_name} split")
    print(f"{'='*60}")
    print(f"{split_name.capitalize()} split size: {len(split_data)}")

    # Convert to pandas for easier manipulation
    df = split_data.to_pandas()
    print(f"Original columns: {df.columns.tolist()}")

    # Show a sample of the original data
    print(f"\n=== Original data sample ({split_name}) ===")
    print(f"Question (first 500 chars): {df['question'].iloc[0][:500]}...")
    if 'answer' in df.columns:
        print(f"Answer: {df['answer'].iloc[0]}")
    if 'label' in df.columns:
        print(f"Label: {df['label'].iloc[0]}")
    if 'choices' in df.columns:
        print(f"Choices: {df['choices'].iloc[0]}")

    # Optionally transform to Markdown-KV format (only transform question column)
    if apply_markdown_kv:
        print(f"\n=== Transforming {split_name} questions to Markdown-KV format ===")
        df['question'] = df['question'].apply(transform_prompt)
        print("   Markdown-KV transformation applied!")

        # Verify transformation quality
        print("   Verifying transformation...")
        sample_transformed = df['question'].iloc[0]
        checks = {
            'has_domain_rules': '# Domain Rules' in sample_transformed,
            'has_drive_test': '# Drive Test Data' in sample_transformed,
            'has_engineering': '# Engineering Parameters' in sample_transformed,
            'has_relationships': '# Data Relationships' in sample_transformed,
            'has_code_blocks': '```' in sample_transformed
        }

        for check_name, passed in checks.items():
            status = "✓" if passed else "✗"
            print(f"   {status} {check_name}")

        if not all(checks.values()):
            print("   ⚠ WARNING: Transformation may have failed!")
            print(f"   Sample length: {len(sample_transformed)} chars")
            print(f"   First 500 chars:\n{sample_transformed[:500]}")

    # Show a sample after transformation (if applied)
    print(f"\n=== Final data sample ({split_name}) ===")
    print(f"Question (first 500 chars): {df['question'].iloc[0][:500]}...")
    for col in ['answer', 'label', 'choices']:
        if col in df.columns:
            print(f"{col.capitalize()}: {df[col].iloc[0]}")

    # Save to different formats
    print(f"\n=== Saving {split_name} split ===")

    # Create output directory if it doesn't exist
    os.makedirs("telelogs_markdown", exist_ok=True)

    # Save as CSV
    csv_path = f"telelogs_markdown/telelogs_{split_name}{suffix}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved to {csv_path}")

    # Save as JSON
    json_path = f"telelogs_markdown/telelogs_{split_name}{suffix}.json"
    df.to_json(json_path, orient='records', indent=2)
    print(f"Saved to {json_path}")

    # Save as parquet (more efficient for large datasets)
    parquet_path = f"telelogs_markdown/telelogs_{split_name}{suffix}.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"Saved to {parquet_path}")

    print(f"\n{split_name.capitalize()} split processed: {len(df)} samples")
    return df

def process_dataset(apply_markdown_kv: bool = False, source_dataset: str = "netop/TeleLogs", splits: list = None):
    """
    Load and process the TeleLogs dataset

    Args:
        apply_markdown_kv: If True, transform questions to Markdown-KV format
        source_dataset: HuggingFace dataset to load from (default: netop/TeleLogs)
        splits: List of splits to process (default: ['train', 'test'])

    Returns:
        Dictionary of processed dataframes by split name
    """
    if splits is None:
        splits = ['train', 'test']

    # Ensure authentication is set up
    ensure_hf_login()

    print("Loading dataset from HuggingFace...")
    print(f"Source: {source_dataset}")
    print(f"Splits to process: {splits}")
    try:
        # Load the dataset with authentication
        ds = load_dataset(source_dataset, token=HF_TOKEN)
        print(f"Dataset loaded successfully!")
        print(f"Available splits: {list(ds.keys())}")

        # Add suffix for non-markdown format
        suffix = "" if apply_markdown_kv else "_plain"

        # Process each requested split
        results = {}
        for split_name in splits:
            if split_name not in ds:
                print(f"\n⚠ WARNING: Split '{split_name}' not found in dataset. Skipping...")
                continue

            split_data = ds[split_name]
            results[split_name] = process_split(split_data, split_name, apply_markdown_kv, suffix)

        print("\n" + "="*60)
        print("=== Processing complete! ===")
        print("="*60)
        for split_name, df in results.items():
            print(f"{split_name.capitalize()} samples: {len(df)}")
            print(f"Columns: {df.columns.tolist()}")
        if apply_markdown_kv:
            print(f"Format: Markdown-KV (improved LLM comprehension)")
        print("\nNote: README.md with dataset card already exists in telelogs_markdown/")
        print("Upload to HuggingFace with: python upload_telelogs.py")

        return results

    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract and transform TeleLogs dataset from HuggingFace to Markdown-KV format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract both train and test splits from netop/TeleLogs with Markdown-KV (default)
  python extract_telelogs.py

  # Extract only train split
  python extract_telelogs.py --splits train

  # Extract only test split
  python extract_telelogs.py --splits test

  # Extract from a different source dataset
  python extract_telelogs.py --source eaguaida/telelogs

  # Extract in standard pipe-delimited format (not recommended)
  python extract_telelogs.py --no-markdown-kv

The Markdown-KV format achieves ~60% accuracy vs ~41-44% for pipe-delimited
format in LLM table understanding benchmarks. This is now the default.
        """
    )

    parser.add_argument(
        '--no-markdown-kv',
        action='store_true',
        help='Skip Markdown-KV transformation (keeps original pipe-delimited format)'
    )

    parser.add_argument(
        '--source',
        default='netop/TeleLogs',
        help='HuggingFace dataset to load from (default: netop/TeleLogs)'
    )

    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'test'],
        help='Dataset splits to process (default: train test)'
    )

    args = parser.parse_args()
    results = process_dataset(
        apply_markdown_kv=not args.no_markdown_kv,
        source_dataset=args.source,
        splits=args.splits
    )
