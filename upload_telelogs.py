"""
Script to upload the processed TeleLogs dataset to HuggingFace
"""
import os
from huggingface_hub import HfApi
from pathlib import Path

# Get HuggingFace token from environment variable
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN environment variable not set. "
        "Please set it with your HuggingFace token:\n"
        "export HF_TOKEN=your_token_here"
    )

def upload_dataset():
    """Upload the processed TeleLogs dataset to HuggingFace."""

    # Initialize HuggingFace API
    api = HfApi(token=HF_TOKEN)

    # Path to the processed dataset folder
    dataset_folder = Path("telelogs")

    if not dataset_folder.exists():
        raise FileNotFoundError(
            f"Dataset folder '{dataset_folder}' not found. "
            "Please run extract_telelogs.py first to create the processed dataset."
        )

    # Check if required files exist
    required_files = ["telelogs_test.parquet", "telelogs_test.json", "telelogs_test.csv"]
    missing_files = [f for f in required_files if not (dataset_folder / f).exists()]

    if missing_files:
        print(f"Warning: Missing files: {missing_files}")
        print("Make sure extract_telelogs.py completed successfully.")

    print("=" * 60)
    print("Uploading TeleLogs dataset to HuggingFace")
    print("=" * 60)
    print(f"Source folder: {dataset_folder.absolute()}")
    print(f"Target repository: eaguaida/telelogs")
    print(f"Repository type: dataset")
    print()

    try:
        # Upload the entire folder to HuggingFace
        api.upload_folder(
            folder_path=str(dataset_folder),
            repo_id="eaguaida/telelogs",
            repo_type="dataset",
            commit_message="Upload processed TeleLogs dataset with MCQ format"
        )

        print("✅ Dataset uploaded successfully!")
        print()
        print("View your dataset at: https://huggingface.co/datasets/eaguaida/telelogs")
        print()
        print("Files uploaded:")
        for file in dataset_folder.iterdir():
            if file.is_file():
                print(f"  - {file.name}")

    except Exception as e:
        print(f"❌ Error uploading dataset: {e}")
        raise

if __name__ == "__main__":
    upload_dataset()
