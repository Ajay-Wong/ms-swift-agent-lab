import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

path = '/models/Qwen3-1.7B'
tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    path, local_files_only=True, dtype=torch.bfloat16,
    attn_implementation='sdpa', device_map='cuda',
)
model = get_peft_model(model, LoraConfig(
    task_type='CAUSAL_LM', r=16, lora_alpha=32,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
))
model.print_trainable_parameters()
batch = tokenizer('用户：查询有货的键盘。助手：我先查询商品列表。', return_tensors='pt').to('cuda')
model.train()
loss = model(**batch, labels=batch['input_ids']).loss
loss.backward()
assert torch.isfinite(loss)
assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad)
print('LoRA forward/backward PASS; loss:', loss.item(), flush=True)
model.eval()
prompt = tokenizer.apply_chat_template(
    [{'role': 'user', 'content': '你好，请用一句话介绍自己。'}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
with torch.no_grad():
    result = model.generate(**inputs, max_new_tokens=48, do_sample=False)
print(tokenizer.decode(result[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True))
print('Peak CUDA allocated GiB:', round(torch.cuda.max_memory_allocated()/2**30, 2))
