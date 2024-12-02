from huggingface_hub import HfApi, HfFolder
import os


def upload_to_huggingface(
    model_path="best_emotion_model.pth",
    repo_name="SentimentSound",
    organization=None,
    commit_message="Upload speech emotion recognition model",
):
    """
    Upload model and related files to Hugging Face

    Args:
        model_path (str): Path to the model weights
        repo_name (str): Name of the Hugging Face repository
        organization (str, optional): Organization to upload to
        commit_message (str): Commit message for the upload
    """
    # Ensure you're logged in
    try:
        # Check if logged in
        token = HfFolder.get_token()
        if not token:
            print("Please log in first. Run `huggingface-cli login` in your terminal.")
            return

        # Create API instance
        api = HfApi()

        # Full repo name (with org if specified)
        full_repo_name = (
            f"{organization}/{repo_name}"
            if organization
            else "Vishal-Padia/" + repo_name
        )

        # Create repository if it doesn't exist
        # try:
        #     api.create_repo(repo_id=full_repo_name, repo_type="model", exist_ok=True)
        # except Exception as e:
        #     print(f"Error creating repository: {e}")
        #     return

        # Files to upload
        files_to_upload = [
            model_path,  # Model weights
            "README.md",  # Readme
            "requirements.txt",  # Dependencies
            "confusion_matrix.png",  # Visualization
            "main.py",  # Main script
            "emotion_predictor.py",  # Prediction script
        ]

        # Upload each file
        for file_path in files_to_upload:
            if os.path.exists(file_path):
                print(f"Uploading {file_path}")
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=os.path.basename(file_path),
                    repo_id=full_repo_name,
                    repo_type="model",
                    commit_message=commit_message,
                )
            else:
                print(f"Warning: {file_path} not found. Skipping.")

        print(f"Successfully uploaded to {full_repo_name}")

    except Exception as e:
        print(f"An error occurred: {e}")


def main():
    # Prompt for optional organization
    org_choice = input(
        "Enter organization name (leave blank for personal account): "
    ).strip()
    organization = org_choice if org_choice else None

    # Upload
    upload_to_huggingface(organization=organization)


if __name__ == "__main__":
    main()
