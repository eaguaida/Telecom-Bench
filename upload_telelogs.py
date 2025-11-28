"""
Script to upload the processed TeleLogs dataset to HuggingFace
"""
import os
import argparse
from huggingface_hub import HfApi
from pathlib import Path

def upload_dataset(plain_format: bool = False, repo_id: str = "eaguaida/telelogs", splits: list = None):
    """Upload the processed TeleLogs dataset to HuggingFace (JSON + README only).

    Args:
        plain_format: If True, upload the plain pipe-delimited version (_plain files)
        repo_id: HuggingFace repository ID to upload to
        splits: List of splits to upload (default: ['train', 'test'])
    """
    if splits is None:
        splits = ['train', 'test']

    # Get HuggingFace token from environment variable
    HF_TOKEN = os.environ.get("HF_TOKEN")
    if not HF_TOKEN:
        raise ValueError(
            "HF_TOKEN environment variable not set. "
            "Please set it with your HuggingFace token:\n"
            "export HF_TOKEN=your_token_here"
        )

    # Initialize HuggingFace API
    api = HfApi(token=HF_TOKEN)

    # Path to the processed dataset folder
    dataset_folder = Path("telelogs_markdown")

    if not dataset_folder.exists():
        raise FileNotFoundError(
            f"Dataset folder '{dataset_folder}' not found. "
            "Please run extract_telelogs.py first to create the processed dataset."
        )

    # Determine which files to upload based on format
    suffix = "_plain" if plain_format else ""
    readme_file = dataset_folder / "README.md"

    # Collect all JSON files to upload
    json_files = []
    for split in splits:
        json_file = dataset_folder / f"telelogs_{split}{suffix}.json"
        if json_file.exists():
            json_files.append((f"telelogs_{split}{suffix}.json", str(json_file)))
        else:
            print(f"⚠ Warning: JSON file not found: {json_file}")
            print(f"   Run: python extract_telelogs.py --splits {split}{' --no-markdown-kv' if plain_format else ''}")

    if not json_files:
        raise FileNotFoundError(
            f"No JSON files found for splits: {splits}\n"
            f"Please run: python extract_telelogs.py --splits {' '.join(splits)}{' --no-markdown-kv' if plain_format else ''}"
        )

    if not readme_file.exists():
        print(f"Warning: README.md not found at {readme_file}")
        print("Dataset card will not be displayed properly on HuggingFace.")

    format_name = "Plain pipe-delimited" if plain_format else "Markdown-KV"
    print("=" * 60)
    print(f"Uploading TeleLogs dataset to HuggingFace ({format_name} format)")
    print("=" * 60)
    print(f"Source folder: {dataset_folder.absolute()}")
    print(f"Target repository: {repo_id}")
    print(f"Repository type: dataset")
    print(f"\nFiles to upload:")
    for filename, _ in json_files:
        print(f"  - {filename} (data)")
    if readme_file.exists():
        print(f"  - README.md (dataset card with YAML metadata)")
    print()

    try:
        # Upload JSON files and README
        files_to_upload = json_files.copy()
        if readme_file.exists():
            files_to_upload.append(("README.md", str(readme_file)))

        for path_in_repo, local_path in files_to_upload:
            if Path(local_path).exists():
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=path_in_repo,
                    repo_id=repo_id,
                    repo_type="dataset",
                    commit_message=f"Upload {path_in_repo} ({format_name} format)"
                )
                print(f"✅ Uploaded: {path_in_repo}")

        print()
        print("✅ Dataset uploaded successfully!")
        print()
        print(f"View your dataset at: https://huggingface.co/datasets/{repo_id}")
        print()
        print("The dataset card should now display with:")
        print("  - Proper YAML metadata")
        print("  - Dataset description and structure")
        print("  - Usage examples")
        if not plain_format:
            print("  - Markdown-KV formatted questions for improved LLM comprehension")
        print(f"\nUploaded splits: {', '.join(splits)}")

    except Exception as e:
        print(f"❌ Error uploading dataset: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload TeleLogs Markdown-KV dataset to HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload both train and test splits in Markdown-KV format (default, recommended)
  python upload_telelogs.py

  # Upload only train split
  python upload_telelogs.py --splits train

  # Upload only test split
  python upload_telelogs.py --splits test

  # Upload plain pipe-delimited format (not recommended)
  python upload_telelogs.py --plain

  # Upload to custom repository
  python upload_telelogs.py --repo-id myuser/my-telelogs
        """
    )

    parser.add_argument(
        '--plain',
        action='store_true',
        help='Upload the plain pipe-delimited version instead of Markdown-KV'
    )

    parser.add_argument(
        '--repo-id',
        default='eaguaida/telelogs',
        help='HuggingFace repository ID (default: eaguaida/telelogs)'
    )

    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'test'],
        help='Dataset splits to upload (default: train test)'
    )

    args = parser.parse_args()
    upload_dataset(plain_format=args.plain, repo_id=args.repo_id, splits=args.splits)
