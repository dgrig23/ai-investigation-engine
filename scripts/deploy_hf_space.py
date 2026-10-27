"""Deploy the app to a Hugging Face Space (Docker SDK, free CPU tier).

What it does:
1. creates the Space if it does not exist (public, Docker),
2. copies the secrets it needs from your local .env into the Space's *secrets*
   (encrypted on Hugging Face, never written to the repository),
3. uploads only what the container needs: app/, Dockerfile, pyproject.toml, README.md.

Usage:
  # in .env: HF_TOKEN=hf_...   (a Hugging Face token with *write* permission)
  python -m scripts.deploy_hf_space --space <your-hf-username>/ai-investigation-engine
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SECRETS = ["QDRANT_URL", "QDRANT_API_KEY", "GROQ_API_KEY", "CEREBRAS_API_KEY", "MISTRAL_API_KEY",
           "GEMINI_API_KEY", "ADMIN_TOKEN"]
VARIABLES = ["LLM_PROVIDERS", "GROQ_MODEL", "CEREBRAS_MODEL", "MISTRAL_MODEL", "GEMINI_MODEL"]


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                if value.strip():
                    values[key.strip()] = value.strip()
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", required=True, help="<username>/<space-name>")
    args = parser.parse_args()

    from huggingface_hub import HfApi

    env = {**read_env(ROOT / ".env"), **{k: v for k, v in os.environ.items() if k in SECRETS + VARIABLES + ["HF_TOKEN"]}}
    token = env.get("HF_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN (a Hugging Face write token) in .env first.")
    if not env.get("QDRANT_URL"):
        raise SystemExit("QDRANT_URL is empty: the Space needs Qdrant Cloud, because its own disk is not persistent.")

    api = HfApi(token=token)
    api.create_repo(repo_id=args.space, repo_type="space", space_sdk="docker", exist_ok=True)
    for key in SECRETS:
        if env.get(key):
            api.add_space_secret(args.space, key, env[key])
            print(f"secret set: {key}")
    for key in VARIABLES:
        if env.get(key):
            api.add_space_variable(args.space, key, env[key])
            print(f"variable set: {key}={env[key]}")
    api.upload_folder(
        folder_path=str(ROOT), repo_id=args.space, repo_type="space",
        allow_patterns=["app/**", "Dockerfile", "pyproject.toml", "README.md"],
        ignore_patterns=["**/__pycache__/**", "**/*.pyc"],
        commit_message="Deploy from local repository",
    )
    user, name = args.space.split("/")
    print(f"Space: https://huggingface.co/spaces/{args.space}")
    print(f"App:   https://{user.lower()}-{name.lower().replace('_', '-')}.hf.space  (first build takes a few minutes)")


if __name__ == "__main__":
    main()
