"""Show a real scripted tool interaction and its outcome reward."""
import argparse
import json

from .tasks import make_task
from .trajectory import expert_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()
    task = make_task(args.seed, 'test', args.index)
    episode = expert_episode(task)
    print('用户：', task['query'])
    for event in episode['events']:
        print('调用：', event['name'], json.dumps(event['arguments'], ensure_ascii=False))
        print('返回：', json.dumps(event['result'], ensure_ascii=False))
    print('答案：', json.dumps(episode['final'], ensure_ascii=False))
    print('判定：', json.dumps(episode['grade'], ensure_ascii=False))


if __name__ == '__main__':
    main()
