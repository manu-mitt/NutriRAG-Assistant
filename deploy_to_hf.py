"""
Deploy NutriChat to Hugging Face Spaces.
Author: Manu M

Usage:
    python deploy_to_hf.py

This script will:
1. Prompt for your HF token (if not already logged in)
2. Create the Space repository on Hugging Face
3. Upload app.py, requirements.txt, README.md, and the embeddings CSV
"""

import os
import sys
import shutil

# Configure stdout encoding to UTF-8
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

# Ensure huggingface_hub is available
try:
    from huggingface_hub import HfApi, login, upload_folder
except ImportError:
    print("[WARN] huggingface_hub not installed. Installing...")
    os.system(f"{sys.executable} -m pip install huggingface_hub")
    from huggingface_hub import HfApi, login, upload_folder

# -- Configuration ---------------------------------------------------
SPACE_NAME = "nutrichat-rag"  # Will become {username}/nutrichat-rag
SPACE_SDK = "gradio"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HF_SPACE_DIR = os.path.join(SCRIPT_DIR, "hf_space")
EMBEDDINGS_CSV = os.path.join(SCRIPT_DIR, "text_chunks_and_embeddings_df.csv")


def main():
    print("=" * 60)
    print("  NutriChat - Deploy to Hugging Face Spaces")
    print("=" * 60)

    # Step 1: Authenticate
    api = HfApi()
    try:
        user_info = api.whoami()
        username = user_info["name"]
        print(f"\n[OK] Already logged in as: {username}")
    except Exception:
        print("\n[WARN] Not logged in to Hugging Face.")
        print("    Get your token from: https://huggingface.co/settings/tokens")
        print("    (Create a token with 'write' permissions)\n")
        token = input("Paste your HF token: ").strip()
        if not token:
            print("[ERROR] No token provided. Aborting.")
            sys.exit(1)
        login(token=token)
        user_info = api.whoami()
        username = user_info["name"]
        print(f"[OK] Logged in as: {username}")

    repo_id = f"{username}/{SPACE_NAME}"
    print(f"\n[INFO] Target Space: https://huggingface.co/spaces/{repo_id}")

    # Step 2: Create the Space (if it doesn't exist)
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk=SPACE_SDK,
            exist_ok=True,
            private=False,
        )
        print(f"[OK] Space '{repo_id}' ready.")
    except Exception as e:
        print(f"[ERROR] Failed to create Space: {e}")
        sys.exit(1)

    # Step 3: Copy embeddings CSV into the upload folder
    csv_dest = os.path.join(HF_SPACE_DIR, "text_chunks_and_embeddings_df.csv")
    if not os.path.exists(csv_dest):
        print(f"[INFO] Copying embeddings CSV ({os.path.getsize(EMBEDDINGS_CSV) / 1e6:.1f} MB)...")
        shutil.copy2(EMBEDDINGS_CSV, csv_dest)
        print("[OK] Embeddings CSV copied.")
    else:
        print("[OK] Embeddings CSV already in place.")

    # Step 4: Upload the folder
    print(f"\n[INFO] Uploading files to HF Spaces...")
    print(f"    Files: app.py, requirements.txt, README.md, text_chunks_and_embeddings_df.csv")
    print(f"    This may take a few minutes for the 21 MB CSV...\n")

    upload_folder(
        folder_path=HF_SPACE_DIR,
        repo_id=repo_id,
        repo_type="space",
    )

    url = f"https://huggingface.co/spaces/{repo_id}"
    print("\n" + "=" * 60)
    print(f"  [OK] DEPLOYMENT SUCCESSFUL!")
    print(f"  [INFO] Your app is live at:")
    print(f"      {url}")
    print(f"\n  Note: First launch takes 3-5 minutes as HF downloads")
    print(f"  the models (~1.5 GB). Subsequent loads are cached.")
    print("=" * 60)


if __name__ == "__main__":
    main()
