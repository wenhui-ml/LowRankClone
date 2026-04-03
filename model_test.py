import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers.models.llama.modeling_llama as modeling_llama
import time

# ================= 🐒 猴子补丁 (Monkey Patch) 最终究极版 =================
orig_init = modeling_llama.LlamaAttention.__init__

def patched_init(self, config, layer_idx=None):
    orig_init(self, config, layer_idx)
    if hasattr(config, "head_dim"):
        self.head_dim = config.head_dim
        
        # 【核心修复】欺骗 HF 的 reshape 逻辑！
        # 让 LlamaAttention 内部以为 hidden_size 是 4096，这样 .view() 就不会崩溃
        self.hidden_size = self.num_heads * self.head_dim 
        
        # 重新初始化线性层
        # 注意：q, k, v 的输入端仍然接收来自残差流的 1536 (config.hidden_size)
        self.q_proj = torch.nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = torch.nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = torch.nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        # o_proj 的输入是 4096，输出端降维回到 1536 (config.hidden_size)
        self.o_proj = torch.nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        
        # 修复 RoPE (你上一步报错解决的地方)
        self.rotary_emb = modeling_llama.LlamaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

modeling_llama.LlamaAttention.__init__ = patched_init
# ================= 🐒 猴子补丁 (Monkey Patch) 结束 =================

# 指定你的模型路径
model_path = "/home/ubuntu/wenhui/LowRankClone/ckpts/LRC-2.7B-Converted"
prompt = "The capital of France is"

print(f"正在从 {model_path} 加载 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

print(f"正在加载模型到 CPU ...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="cpu",
    torch_dtype=torch.bfloat16 
)
print("✅ 模型成功加载！没有报错！")

inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=20, 
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
print("\n" + "="*40)
print("🎯 最终生成结果:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
print("="*40 + "\n")