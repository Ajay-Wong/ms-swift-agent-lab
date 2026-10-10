"""Dependency-free environment. Prices are integer yuan; stock is episode-local."""
from copy import deepcopy

SYSTEM = '''你是商品查询助手。所有价格以元计，均为整数。
必须先用 search_products 按用户指定的品类和预算查询。没有候选商品时可以直接回答无匹配。
有候选商品时，必须用 get_stock 查询全部候选商品的库存，再选择有货且最便宜的商品。
同价时选择 product_id 按字符串升序排列最靠前的商品。不要猜测库存、商品 ID 或价格。
每次只调用一个工具，等待真实返回结果后再继续，最多调用三次工具。
最终只输出 JSON：{"product_id":"商品ID","price":价格}。
没有符合条件的商品时输出 {"product_id":null,"price":null}。不要输出解释或其他字段。'''
CATEGORIES = ('机械键盘', '鼠标', '耳机')
TOOLS = [
    {'type': 'function', 'function': {
        'name': 'search_products', 'description': '按品类和预算查询商品，不返回库存。',
        'parameters': {'type': 'object', 'properties': {
            'category': {'type': 'string', 'enum': list(CATEGORIES)},
            'max_price': {'type': 'integer', 'minimum': 0, 'description': '预算上限，单位元，包含此价格。'},
        }, 'required': ['category', 'max_price'], 'additionalProperties': False},
    }},
    {'type': 'function', 'function': {
        'name': 'get_stock', 'description': '查询本轮 search_products 已返回的商品库存。',
        'parameters': {'type': 'object', 'properties': {
            'product_ids': {'type': 'array', 'items': {'type': 'string'}, 'minItems': 1, 'maxItems': 20},
        }, 'required': ['product_ids'], 'additionalProperties': False},
    }},
]


def correct_answer(task):
    """Independent oracle; never included in the model's observation."""
    eligible = [p for p in task['catalog'] if p['category'] == task['category']
                and p['price'] <= task['max_price'] and p['stock'] > 0]
    if not eligible:
        return {'product_id': None, 'price': None}
    best = min(eligible, key=lambda p: (p['price'], p['product_id']))
    return {'product_id': best['product_id'], 'price': best['price']}


class ShoppingEnv:
    def __init__(self, task, max_calls=3):
        self.task = deepcopy(task)
        self.max_calls = max_calls
        self.events = []
        self.discovered = set()
        self.stock_observed = set()
        self.searched_exact = False
        self.products = {p['product_id']: p for p in self.task['catalog']}
        if len(self.products) != len(self.task['catalog']):
            raise ValueError('Duplicate product IDs')
        if max_calls < 1:
            raise ValueError('max_calls must be positive')

    def _execute(self, name, args):
        if not isinstance(args, dict):
            raise ValueError('arguments must be an object')
        if name == 'search_products':
            if set(args) != {'category', 'max_price'}:
                raise ValueError('search_products requires category and max_price only')
            if args['category'] not in CATEGORIES or type(args['max_price']) is not int or args['max_price'] < 0:
                raise ValueError('Invalid category or integer budget')
            rows = [{'product_id': p['product_id'], 'name': p['name'], 'price': p['price']}
                    for p in self.task['catalog'] if p['category'] == args['category']
                    and p['price'] <= args['max_price']]
            rows.sort(key=lambda p: p['product_id'])
            self.discovered.update(p['product_id'] for p in rows)
            self.searched_exact |= (args['category'] == self.task['category']
                                    and args['max_price'] == self.task['max_price'])
            return {'products': rows}
        if name == 'get_stock':
            if set(args) != {'product_ids'}:
                raise ValueError('get_stock requires product_ids only')
            ids = args['product_ids']
            if not isinstance(ids, list) or not 1 <= len(ids) <= 20 or not all(isinstance(x, str) for x in ids):
                raise ValueError('product_ids must contain 1 to 20 strings')
            if len(set(ids)) != len(ids) or not set(ids) <= self.discovered:
                raise ValueError('IDs must be unique and returned by search_products in this episode')
            self.stock_observed.update(ids)
            return {'stock': [{'product_id': x, 'quantity': self.products[x]['stock']} for x in ids]}
        raise ValueError('Unknown tool')

    def step(self, name, arguments):
        """Only registered read-only functions can run; no eval/exec or network."""
        try:
            if len(self.events) >= self.max_calls:
                raise ValueError('Tool call budget exceeded')
            result = self._execute(name, arguments)
            ok = True
        except (ValueError, TypeError) as exc:
            result, ok = {'error': str(exc)}, False
        self.events.append(deepcopy({'name': name, 'arguments': arguments, 'result': result, 'ok': ok}))
        return deepcopy(result)


def judge_episode(task, events, final, error=None, max_calls=3):
    """Binary RL-ready reward. Replay tools so fabricated observations cannot pass."""
    replay = ShoppingEnv(task, max_calls=max_calls)
    authentic = True
    for event in events:
        result = replay.step(event.get('name'), event.get('arguments'))
        authentic &= result == event.get('result') and replay.events[-1]['ok'] == event.get('ok')
    valid_calls = sum(e['ok'] for e in replay.events)
    format_valid = (isinstance(final, dict) and set(final) == {'product_id', 'price'}
                    and ((final['product_id'] is None and final['price'] is None)
                         or (isinstance(final['product_id'], str) and type(final['price']) is int
                             and final['price'] >= 0)))
    answer_correct = bool(format_valid and final == correct_answer(task))
    candidate_ids = {p['product_id'] for p in task['catalog']
                     if p['category'] == task['category'] and p['price'] <= task['max_price']}
    grounded = replay.searched_exact and candidate_ids <= replay.stock_observed
    success = bool(answer_correct and grounded and authentic and not error
                   and len(events) <= max_calls and valid_calls == len(events))
    return {'success': success, 'reward': float(success), 'answer_correct': answer_correct,
            'format_valid': bool(format_valid), 'grounded': bool(grounded),
            'authentic_observations': bool(authentic), 'tool_calls': len(events),
            'valid_tool_calls': valid_calls, 'error': error}
