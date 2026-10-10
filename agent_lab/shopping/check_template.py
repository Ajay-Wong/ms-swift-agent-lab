"""Verify real SWIFT tokenization, tool-call targets, and masked tool observations."""
import argparse
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--model', default='/models/Qwen3-1.7B')
    parser.add_argument('--max-length', type=int, default=2048)
    parser.add_argument('--limit', type=int, default=0, help='0 checks all rows')
    args = parser.parse_args()
    from swift.model import get_model_processor
    from swift.template import get_template
    _, processor = get_model_processor(args.model, load_model=False)
    tokenizer = getattr(processor, 'tokenizer', processor)
    template = get_template(processor, template_type='qwen3', agent_template='hermes',
                            enable_thinking=False, max_length=args.max_length, truncation_strategy='raise',
                            loss_scale='default+ignore_empty_think')
    template.set_mode('train')
    lengths, trained = [], []
    with args.data.open() as source:
        for index, line in enumerate(source):
            if args.limit and index >= args.limit:
                break
            row = json.loads(line)
            encoded = template.encode(row)
            ids, labels = encoded['input_ids'], encoded['labels']
            supervised = tokenizer.decode([i for i, label in zip(ids, labels) if label != -100])
            if not 0 < sum(label != -100 for label in labels) < len(ids):
                raise AssertionError(f'Invalid loss mask at row {index}')
            if 'search_products' not in supervised:
                raise AssertionError(f'Search tool call missing from labels: {index}')
            if any(m['role'] == 'assistant' and m.get('tool_calls', [{}])[0].get('function', {}).get('name') == 'get_stock'
                   for m in row['messages']) and 'get_stock' not in supervised:
                raise AssertionError(f'Stock tool call missing from labels: {index}')
            if re.search(r'"(?:products|stock)"\s*:', supervised):
                raise AssertionError(f'Tool observation is being trained: {index}')
            if row['messages'][-1]['content'] not in supervised:
                raise AssertionError(f'Final answer missing from labels: {index}')
            lengths.append(len(ids))
            trained.append(sum(label != -100 for label in labels))
    if not lengths:
        raise ValueError('No examples checked')
    print(json.dumps({'rows': len(lengths), 'max_tokens': max(lengths),
                      'mean_tokens': round(sum(lengths)/len(lengths), 1),
                      'mean_supervised_tokens': round(sum(trained)/len(trained), 1),
                      'tool_calls_supervised': True, 'tool_results_masked': True}, indent=2))


if __name__ == '__main__':
    main()
