import os
import faulthandler
faulthandler.dump_traceback_later(120, repeat=True)
os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')

def main():
    from vllm import LLM, SamplingParams
    print('Creating vLLM engine...', flush=True)
    llm = LLM(
        model='/models/Qwen3-1.7B', dtype='bfloat16',
        max_model_len=512, max_num_seqs=2,
        gpu_memory_utilization=0.30, enforce_eager=True,
        enable_prefix_caching=False,
    )
    print('vLLM engine ready.', flush=True)
    tokenizer = llm.get_tokenizer()
    prompt = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': '你好，请用一句话介绍自己。'}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    print('Generating...', flush=True)
    outputs = llm.generate([prompt], SamplingParams(max_tokens=32, temperature=0))
    text = outputs[0].outputs[0].text
    assert text.strip(), 'Empty generation'
    print('vLLM rollout inference PASS:', text, flush=True)
    faulthandler.cancel_dump_traceback_later()

if __name__ == '__main__':
    main()
