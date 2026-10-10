"""Deterministic task generation with independent train/val/test worlds."""
import hashlib
import random

from .environment import CATEGORIES

CASES = ('normal', 'no_candidates', 'sold_out', 'budget_boundary', 'price_tie', 'cheapest_unavailable')


def make_task(seed, split, index):
    if split not in {'train', 'val', 'test'} or index < 0:
        raise ValueError('Invalid split or index')
    digest = hashlib.sha256(f'{seed}:{split}:{index}'.encode()).hexdigest()
    rng = random.Random(int(digest, 16))
    category = rng.choice(CATEGORIES)
    budget = rng.randrange(150, 501, 10)
    case = CASES[index % len(CASES)]
    offers = [(rng.randint(30, budget), rng.choice([0, 1, 2, 5])) for _ in range(rng.randint(3, 5))]
    if case == 'normal':
        offers[0] = (offers[0][0], 2)
    elif case == 'no_candidates':
        offers = [(budget + rng.randint(1, 100), 3) for _ in offers]
    elif case == 'sold_out':
        offers = [(price, 0) for price, _ in offers]
    elif case == 'budget_boundary':
        offers = [(budget, 1)] + [(budget + j + 1, 3) for j in range(len(offers) - 1)]
    elif case == 'price_tie':
        low = rng.randint(40, budget - 30)
        offers = [(low, 2), (low, 1), (low - 10, 0), (low + 20, 5)]
    elif case == 'cheapest_unavailable':
        low = rng.randint(40, budget - 30)
        offers = [(low, 0), (low + 10, 1), (low + 20, 2)]
    rng.shuffle(offers)
    rows = []
    for j, (price, stock) in enumerate(offers):
        rows.append({'product_id': f'P{digest[:12]}-{j:02d}', 'name': f'{category}{rng.randint(10,99)}号',
                     'category': category, 'price': price, 'stock': stock})
    for other in CATEGORIES:
        if other != category:
            rows.append({'product_id': f'P{digest[:12]}-{len(rows):02d}', 'name': f'{other}基础款',
                         'category': other, 'price': 10, 'stock': 9})
    rng.shuffle(rows)
    prompts = [f'帮我找一款不超过{budget}元、有货的{category}，选最便宜的。',
               f'我想买{category}，预算上限{budget}元。请选择最便宜的现货。',
               f'查询{category}：价格≤{budget}元，库存大于0，选择价格最低的一款。',
               f'预算{budget}元（含），买有库存的{category}，哪款最便宜？']
    return {'task_id': f'{split}-{index:06d}-{digest[:12]}', 'split': split, 'case': case,
            'category': category, 'max_price': budget, 'query': rng.choice(prompts), 'catalog': rows}
