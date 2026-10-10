"""Shared strict protocol for demonstration generation and model execution."""
import json
import re
from copy import deepcopy

from .environment import SYSTEM, TOOLS, ShoppingEnv, judge_episode


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def initial_messages(task):
    return [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': task['query']}]


def append_call(messages, name, arguments, result, number):
    call_id = f'call_{number}'
    messages.append({'role': 'assistant', 'content': '', 'tool_calls': [{
        'id': call_id, 'type': 'function', 'function': {'name': name, 'arguments': dumps(arguments)}}]})
    messages.append({'role': 'tool', 'tool_call_id': call_id, 'name': name, 'content': dumps(result)})


def expert_episode(task):
    """Teacher selects using actual tool results, independently of the grading oracle."""
    env, messages = ShoppingEnv(task), initial_messages(task)
    args = {'category': task['category'], 'max_price': task['max_price']}
    result = env.step('search_products', args)
    append_call(messages, 'search_products', args, result, 1)
    products = result['products']
    final = {'product_id': None, 'price': None}
    if products:
        args = {'product_ids': [p['product_id'] for p in products]}
        result = env.step('get_stock', args)
        append_call(messages, 'get_stock', args, result, 2)
        stock = {s['product_id']: s['quantity'] for s in result['stock']}
        available = [p for p in products if stock[p['product_id']] > 0]
        if available:
            best = min(available, key=lambda p: (p['price'], p['product_id']))
            final = {'product_id': best['product_id'], 'price': best['price']}
    messages.append({'role': 'assistant', 'content': dumps(final)})
    grade = judge_episode(task, env.events, final)
    if not grade['success']:
        raise AssertionError(f'Teacher failed: {task["task_id"]}')
    return {'messages': messages, 'events': env.events, 'final': final, 'grade': grade}


def sft_row(task):
    episode = expert_episode(task)
    # JSON string avoids Arrow trying to unify different tool-argument schemas.
    return {'messages': episode['messages'], 'tools': dumps(TOOLS)}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


class GenerationLimitError(ValueError):
    pass


def _reject_constant(value):
    raise ValueError('Non-finite JSON')


def parse_action(text):
    if not isinstance(text, str):
        raise ValueError('Output must be text')
    text = text.strip()
    text = re.sub(r'^<think>\s*</think>\s*', '', text, count=1)
    match = re.fullmatch(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.DOTALL)
    payload = json.loads(match.group(1) if match else text, object_pairs_hook=_unique_object,
                         parse_constant=_reject_constant)
    if match:
        if not isinstance(payload, dict) or set(payload) != {'name', 'arguments'}:
            raise ValueError('Tool call must contain name and arguments only')
        if not isinstance(payload['name'], str) or not isinstance(payload['arguments'], dict):
            raise ValueError('Invalid tool call types')
        return 'call', payload
    return 'final', payload


def run_episode(task, generate, max_calls=3):
    """generate sees only messages/tools, never task.catalog or the oracle answer."""
    env, messages, raw_outputs = ShoppingEnv(task, max_calls), initial_messages(task), []
    final, error = None, None
    for turn in range(max_calls + 1):
        try:
            text = generate(deepcopy(messages), deepcopy(TOOLS))
        except GenerationLimitError:
            error = 'generation_token_limit'
            break
        raw_outputs.append(text)
        try:
            kind, value = parse_action(text)
        except (ValueError, TypeError) as exc:
            error = f'invalid_output: {exc}'
            break
        if kind == 'final':
            final = value
            messages.append({'role': 'assistant', 'content': text})
            break
        if turn == max_calls:
            error = 'tool_call_budget_exceeded'
            break
        result = env.step(value['name'], value['arguments'])
        append_call(messages, value['name'], value['arguments'], result, turn + 1)
    else:
        error = 'no_final_answer'
    if final is None and error is None:
        error = 'no_final_answer'
    return {'messages': messages, 'events': env.events, 'raw_outputs': raw_outputs, 'final': final,
            'grade': judge_episode(task, env.events, final, error, max_calls)}
