"""Download Qwen's public checkpoint without modifying other models."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models-dir', type=Path, default=Path.home() / 'Models')
    args = parser.parse_args()
    from modelscope import snapshot_download
    target = args.models_dir.expanduser().resolve() / 'Qwen3-1.7B'
    target.parent.mkdir(parents=True, exist_ok=True)
    print(snapshot_download('Qwen/Qwen3-1.7B', local_dir=str(target)))


if __name__ == '__main__':
    main()
