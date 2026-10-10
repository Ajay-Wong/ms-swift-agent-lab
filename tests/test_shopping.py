import json
from pathlib import Path
import tempfile
import unittest

from agent_lab.shopping.environment import ShoppingEnv, correct_answer, judge_episode
from agent_lab.shopping.generate import generate_dataset
from agent_lab.shopping.tasks import CASES, make_task
from agent_lab.shopping.trajectory import (GenerationLimitError, dumps, expert_episode, parse_action,
                                           run_episode)
from agent_lab.shopping.evaluate import evaluate


class ShoppingTests(unittest.TestCase):
    def task(self, case=0):
        return make_task(42, 'test', case)

    def test_teacher_and_oracle_agree_across_worlds(self):
        for seed in range(15):
            for index in range(24):
                with self.subTest(seed=seed, case=CASES[index % 6]):
                    task = make_task(seed, 'test', index)
                    episode = expert_episode(task)
                    self.assertTrue(episode['grade']['success'])
                    self.assertEqual(episode['final'], correct_answer(task))
                    self.assertLessEqual(len(episode['events']), 2)

    def test_scenario_semantics(self):
        self.assertIsNone(correct_answer(self.task(1))['product_id'])
        self.assertIsNone(correct_answer(self.task(2))['product_id'])
        self.assertEqual(correct_answer(self.task(3))['price'], self.task(3)['max_price'])
        task = self.task(4)
        available = [p for p in task['catalog'] if p['category'] == task['category'] and p['stock'] > 0]
        low = min(p['price'] for p in available)
        tied = [p['product_id'] for p in available if p['price'] == low]
        self.assertEqual(len(tied), 2)
        self.assertEqual(correct_answer(task)['product_id'], min(tied))
        task = self.task(5)
        prices = [p['price'] for p in task['catalog'] if p['category'] == task['category']]
        self.assertGreater(correct_answer(task)['price'], min(prices))

    def test_search_hides_stock_and_filters_category(self):
        task = self.task()
        env = ShoppingEnv(task)
        result = env.step('search_products', {'category': task['category'], 'max_price': task['max_price']})
        for p in result['products']:
            self.assertEqual(set(p), {'product_id', 'name', 'price'})
            self.assertLessEqual(p['price'], task['max_price'])
        self.assertNotIn('stock', dumps(result))
        result['products'].clear()
        self.assertTrue(env.events[0]['result']['products'])

    def test_tool_argument_validation(self):
        invalid = [('shell', {}), ('search_products', {'category': '机械键盘', 'max_price': True}),
                   ('search_products', {'category': '机械键盘', 'max_price': 1.5}),
                   ('search_products', {'category': '机械键盘', 'max_price': -1}),
                   ('search_products', {'category': '机械键盘', 'max_price': 200, 'extra': 1}),
                   ('get_stock', {'product_ids': []}), ('get_stock', {'product_ids': ['unknown']}),
                   ('get_stock', {'product_ids': [None]})]
        for name, args in invalid:
            with self.subTest(name=name, args=args):
                env = ShoppingEnv(self.task())
                self.assertIn('error', env.step(name, args))
                self.assertFalse(env.events[-1]['ok'])

    def test_stock_requires_search_and_unique_ids(self):
        task, env = self.task(), ShoppingEnv(self.task())
        pid = task['catalog'][0]['product_id']
        self.assertIn('error', env.step('get_stock', {'product_ids': [pid]}))
        result = env.step('search_products', {'category': task['category'], 'max_price': task['max_price']})
        pid = result['products'][0]['product_id']
        self.assertIn('error', env.step('get_stock', {'product_ids': [pid, pid]}))

    def test_state_is_isolated(self):
        task = self.task()
        env = ShoppingEnv(task)
        task['catalog'].clear()
        self.assertTrue(env.task['catalog'])
        other = ShoppingEnv(self.task())
        self.assertFalse(other.discovered)
        args = {'category': self.task()['category'], 'max_price': self.task()['max_price']}
        env.step('search_products', args)
        self.assertFalse(other.discovered)

    def test_lucky_correct_answer_without_tools_fails(self):
        task = self.task()
        grade = judge_episode(task, [], correct_answer(task))
        self.assertTrue(grade['answer_correct'])
        self.assertEqual(grade['reward'], 0)

    def test_partial_stock_evidence_fails(self):
        task = self.task(4)
        env = ShoppingEnv(task)
        env.step('search_products', {'category': task['category'], 'max_price': task['max_price']})
        env.step('get_stock', {'product_ids': [correct_answer(task)['product_id']]})
        self.assertFalse(judge_episode(task, env.events, correct_answer(task))['success'])

    def test_forged_observation_fails(self):
        task = self.task()
        episode = expert_episode(task)
        episode['events'][0]['result'] = {'products': []}
        grade = judge_episode(task, episode['events'], episode['final'])
        self.assertFalse(grade['authentic_observations'])
        self.assertEqual(grade['reward'], 0)

    def test_wrong_price_and_extra_fields_fail(self):
        task = self.task()
        episode = expert_episode(task)
        for final in ({**episode['final'], 'price': True}, {**episode['final'], 'extra': 1},
                      {**episode['final'], 'price': episode['final']['price'] + 1}):
            self.assertFalse(judge_episode(task, episode['events'], final)['success'])

    def test_strict_output_parser(self):
        good = '<tool_call>{"name":"get_stock","arguments":{"product_ids":["P1"]}}</tool_call>'
        self.assertEqual(parse_action(good)[0], 'call')
        self.assertEqual(parse_action('<think>\n\n</think>\n{"product_id":null,"price":null}')[0], 'final')
        for bad in (good + good, 'Here is the answer: {}', '{"a":1,"a":2}', '{"price":NaN}',
                    '<tool_call>{"name":"get_stock","arguments":[]}</tool_call>', None):
            with self.subTest(bad=bad), self.assertRaises((ValueError, TypeError)):
                parse_action(bad)

    def test_runner_executes_teacher_actions_without_catalog_leak(self):
        task = self.task(4)
        teacher = expert_episode(task)
        actions = [f'<tool_call>{dumps({"name": e["name"], "arguments": e["arguments"]})}</tool_call>'
                   for e in teacher['events']] + [dumps(teacher['final'])]
        def generate(messages, tools):
            self.assertNotIn('catalog', dumps(messages))
            self.assertNotIn('stock', messages[1]['content'])
            if len(actions) == 2:
                self.assertEqual(messages[-1]['role'], 'tool')
                self.assertIn('products', json.loads(messages[-1]['content']))
            return actions.pop(0)
        self.assertTrue(run_episode(task, generate)['grade']['success'])

    def test_runner_bounds_and_generation_limit(self):
        task = self.task()
        call = '<tool_call>' + dumps({'name': 'search_products', 'arguments': {
            'category': task['category'], 'max_price': task['max_price']}}) + '</tool_call>'
        result = run_episode(task, lambda *_: call)
        self.assertEqual(result['grade']['error'], 'tool_call_budget_exceeded')
        self.assertEqual(len(result['events']), 3)
        def limited(*_):
            raise GenerationLimitError()
        self.assertEqual(run_episode(task, limited)['grade']['error'], 'generation_token_limit')

    def test_dataset_determinism_and_no_split_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp)/'a', Path(tmp)/'b'
            manifest = generate_dataset(a, 12, 6, 6, 123)
            self.assertEqual(manifest, generate_dataset(b, 12, 6, 6, 123))
            worlds = []
            for split in ('train', 'val', 'test'):
                rows = [json.loads(x) for x in (a/f'{split}.tasks.jsonl').read_text().splitlines()]
                worlds.append({p['product_id'] for t in rows for p in t['catalog']})
                sft = [json.loads(x) for x in (a/f'{split}.sft.jsonl').read_text().splitlines()]
                self.assertTrue(all(set(row) == {'messages', 'tools'} for row in sft))
            self.assertFalse(worlds[0] & worlds[1] or worlds[0] & worlds[2] or worlds[1] & worlds[2])
            with self.assertRaises(FileExistsError):
                generate_dataset(a, 12, 6, 6, 123)
            summary = evaluate(a/'test.tasks.jsonl', Path(tmp)/'eval', limit=6)
            self.assertEqual(summary['backend'], 'scripted')
            self.assertEqual(summary['success_rate'], 1)
            with self.assertRaises(FileExistsError):
                evaluate(a/'test.tasks.jsonl', Path(tmp)/'eval')


if __name__ == '__main__':
    unittest.main()
