import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description='Verify the official Qwen3-1.7B files.')
parser.add_argument('--model-dir', type=Path, default=Path.home() / 'Models' / 'Qwen3-1.7B')
root = parser.parse_args().model_dir
manifest = Path(__file__).resolve().parent.parent / 'model-files.json'
files = json.loads(manifest.read_text())['Data']['Files']
for item in files:
    if item['Type'] != 'blob':
        continue
    path = root / item['Path']
    assert path.is_file(), f'Missing: {path}'
    assert path.stat().st_size == item['Size'], f'Size mismatch: {path}'
    with path.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    assert digest == item['Sha256'], f'SHA256 mismatch: {path}'
    print('SHA256 OK:', item['Path'], flush=True)
print('ALL MODEL FILES VERIFIED')
