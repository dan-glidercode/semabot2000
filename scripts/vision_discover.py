"""Vision LLM Class Discovery — Claude Vision analyzes game frames.

Sends a sample of recorded gameplay frames to Claude Vision and asks
it to identify every distinct object type, producing an ontology.json
file that Grounding DINO can use for bounding-box labeling.

Usage:
    python scripts/vision_discover.py --dataset datasets/steal_a_brainrot
    python scripts/vision_discover.py --dataset datasets/steal_a_brainrot --samples 20
    python scripts/vision_discover.py --dataset datasets/steal_a_brainrot --model claude-sonnet-4-20250514

Requires ANTHROPIC_API_KEY in .env or environment.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import sys
from pathlib import Path

# Load .env before importing anthropic
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # dotenv is optional if env var is already set


def get_api_key() -> str:
    """Read ANTHROPIC_API_KEY from environment."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key or key.startswith("sk-ant-your-key"):
        print("ERROR: Set ANTHROPIC_API_KEY in .env or environment.")
        print("  cp .env.template .env  # then edit with your real key")
        sys.exit(1)
    return key


def sample_frames(images_dir: Path, n: int, seed: int = 42) -> list[Path]:
    """Pick *n* evenly-spaced frames from the recording."""
    all_frames = sorted(images_dir.glob("*.png"))
    if not all_frames:
        print(f"ERROR: No .png files in {images_dir}")
        sys.exit(1)

    if len(all_frames) <= n:
        return all_frames

    # Evenly spaced selection for maximum diversity
    step = len(all_frames) / n
    selected = [all_frames[int(i * step)] for i in range(n)]
    return selected


def encode_image(path: Path) -> dict:
    """Encode an image as a base64 content block for the API."""
    data = base64.standard_b64encode(path.read_bytes()).decode("ascii")
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": data,
        },
    }


DISCOVERY_PROMPT = """You are analyzing screenshots from a Roblox game.

Look at ALL the provided game screenshots carefully. Identify every distinct
type of visual object, character, element, or UI component you can see.

For EACH distinct object type, provide:
1. A short class name (snake_case, e.g. "player_character", "enemy_npc")
2. A natural language description that a zero-shot object detector could use
   to find this object (e.g. "blocky roblox character with colored outfit")

Focus on objects that are relevant for a game-playing bot:
- Player characters and NPCs
- Enemies or targets
- Collectible items, coins, or pickups
- Interactive elements (shops, doors, buttons)
- Obstacles or hazards
- Important UI elements (health bars, score displays)

Do NOT include:
- Background terrain/sky (not detectable as distinct objects)
- Generic "ground" or "wall" (too vague for detection)

Output ONLY valid JSON — no markdown, no code fences, no explanation.
Format: {"class_name": "detection_prompt", ...}

Example output:
{"player": "blocky roblox character with legs and colored outfit",
 "enemy": "red colored enemy character",
 "coin": "floating yellow coin or token",
 "shop": "shop booth or stand with sign"}"""


def discover_classes(
    frames: list[Path],
    model: str,
    api_key: str,
) -> dict[str, str]:
    """Send frames to Claude Vision and parse the ontology response."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    # Build content: images + prompt
    content: list[dict] = []
    for frame in frames:
        content.append(encode_image(frame))

    content.append({"type": "text", "text": DISCOVERY_PROMPT})

    print(f"  Sending {len(frames)} frames to {model}...")
    print("  (this may take 30-60 seconds)")

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        messages=[{"role": "user", "content": content}],
    )

    raw_text = response.content[0].text.strip()

    # Try to parse JSON — handle markdown code fences if present
    json_text = raw_text
    if json_text.startswith("```"):
        # Strip code fences
        lines = json_text.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        json_text = "\n".join(lines)

    try:
        ontology = json.loads(json_text)
    except json.JSONDecodeError:
        print("WARNING: Could not parse JSON response. Raw output:")
        print(raw_text)
        print("\nSaving raw response for manual review.")
        return {"_raw_response": raw_text}

    return ontology


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Discover game object classes using Claude Vision",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to recorded dataset (with images/ subdir)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=15,
        help="Number of frames to send (default: 15)",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-20250514",
        help="Claude model to use",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for ontology.json (default: <dataset>/ontology.json)",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    images_dir = dataset_dir / "images"
    if not images_dir.exists():
        print(f"ERROR: {images_dir} not found")
        sys.exit(1)

    output_path = Path(args.output) if args.output else dataset_dir / "ontology.json"

    print("=" * 60)
    print("  Vision LLM Class Discovery")
    print("=" * 60)
    print(f"  Dataset: {dataset_dir}")
    print(f"  Model:   {args.model}")
    print(f"  Samples: {args.samples}")

    api_key = get_api_key()
    frames = sample_frames(images_dir, args.samples)
    print(f"  Selected {len(frames)} frames")

    ontology = discover_classes(frames, args.model, api_key)

    # Save ontology
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(ontology, f, indent=2)

    print(f"\n  Ontology saved to: {output_path}")
    print(f"  Classes discovered: {len(ontology)}")
    print()
    for cls, prompt in ontology.items():
        if cls.startswith("_"):
            continue
        print(f"    {cls:25s} -> {prompt}")
    print()
    print("  Review the ontology and edit if needed before running")
    print("  autodistill_label.py with Grounding DINO.")


if __name__ == "__main__":
    main()
