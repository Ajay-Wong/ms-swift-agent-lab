"""Evaluate scripted, base, or LoRA policies by actually executing tools."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from .trajectory import GenerationLimitError, dumps, expert_episode, run_episode


class SwiftPolicy:
    def __init__(self, model, adapter=None, max_tokens=256, max_length=2048):
        import torch
        from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig
        if not Path(model).is_dir():
            raise FileNotFoundError(model)
        if adapter and not (Path(adapter) / 'adapter_config.json').is_file():
            raise FileNotFoundError(f'Not a LoRA checkpoint directory: {adapter}')
        self.engine = TransformersEngine(model, adapters=[adapter] if adapter else [],
                                         torch_dtype=torch.bfloat16, device_map='cuda', attn_impl='sdpa',
                                         template_type='qwen3', max_batch_size=1)
        self.engine.template.enable_thinking = False
        self.engine.template.max_length = max_length
        self.engine.template.truncation_strategy = 'raise'
        self.InferRequest = InferRequest
        self.config = RequestConfig(max_tokens=max_tokens, temperature=0, seed=42)

    def __call__(self, messages, tools):
        response = self.engine.infer([self.InferRequest(messages=messages, tools=tools)],
                                     self.config, use_tqdm=False)[0]
        if isinstance(response, Exception):
            raise response
        choice = response.choices[0]
        if choice.finish_reason == 'length':
            raise GenerationLimitError('generation_token_limit')
        return choice.message.content


def evaluate(tasks_path, output, backend='scripted', model='/models/Qwen3-1.7B', adapter=None,
             limit=None, max_tokens=256, overwrite=False):
    tasks_path, output = Path(tasks_path), Path(output)
    if backend not in {'scripted', 'model'}:
        raise ValueError('Unknown backend')
    if adapter and backend != 'model':
        raise ValueError('--adapter requires --backend model')
    if limit is not None and limit < 1:
        raise ValueError('limit must be positive')
    data = tasks_path.read_bytes()
    tasks = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
    if limit is not None:
        tasks = tasks[:limit]
    if not tasks:
        raise ValueError('Empty task set')
    if not overwrite and any((output / name).exists() for name in ('episodes.jsonl', 'summary.json')):
        raise FileExistsError('Evaluation exists; choose a new output directory or --overwrite')
    output.mkdir(parents=True, exist_ok=True)
    policy = SwiftPolicy(model, adapter, max_tokens) if backend == 'model' else None
    started, totals, cases = time.monotonic(), Counter(), {}
    with (output / 'episodes.jsonl').open('w') as log:
        for i, task in enumerate(tasks):
            # Inference failures abort the run instead of silently becoming model failures.
            episode = expert_episode(task) if policy is None else run_episode(task, policy)
            grade = episode['grade']
            for key in ('success', 'answer_correct', 'format_valid', 'grounded', 'tool_calls', 'valid_tool_calls'):
                totals[key] += grade[key]
            bucket = cases.setdefault(task['case'], {'count': 0, 'successes': 0})
            bucket['count'] += 1
            bucket['successes'] += int(grade['success'])
            log.write(dumps({'task_id': task['task_id'], 'case': task['case'], **episode}) + '\n')
            log.flush()
            print(f'{i+1}/{len(tasks)} {task["task_id"]}: success={grade["success"]}', flush=True)
    n = len(tasks)
    summary = {'backend': backend, 'model': model if policy else None, 'adapter': adapter,
               'tasks_sha256': hashlib.sha256(data).hexdigest(), 'evaluated_task_ids': [t['task_id'] for t in tasks],
               'count': n, 'success_rate': totals['success'] / n,
               'answer_accuracy': totals['answer_correct'] / n, 'format_valid_rate': totals['format_valid'] / n,
               'grounded_rate': totals['grounded'] / n, 'mean_tool_calls': totals['tool_calls'] / n,
               'valid_tool_call_rate': (totals['valid_tool_calls'] / totals['tool_calls']
                                        if totals['tool_calls'] else None),
               'cases': cases, 'elapsed_seconds': round(time.monotonic() - started, 2)}
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', dest='tasks_path', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=['scripted', 'model'], default='scripted')
    parser.add_argument('--model', default='/models/Qwen3-1.7B')
    parser.add_argument('--adapter')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--max-tokens', type=int, default=256)
    parser.add_argument('--overwrite', action='store_true')
    summary = evaluate(**vars(parser.parse_args()))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
