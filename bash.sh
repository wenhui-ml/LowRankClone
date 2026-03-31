1.数据整理
cd /home/ubuntu/wenhui/LowRankClone
mkdir -p datasets

huggingface-cli download HuggingFaceFW/fineweb-edu --local-dir datasets/fineweb-edu-dedup --repo-type dataset --include "sample/10BT/*.parquet"
huggingface-cli download teknium/OpenHermes-2.5 --local-dir datasets/OpenHermes-2.5 --repo-type dataset
huggingface-cli download HuggingFaceH4/ultrachat_200k --local-dir datasets/ultrachat_200k --repo-type dataset
huggingface-cli download mlfoundations/dclm-baseline-1.0-parquet --local-dir datasets/dclm-baseline-1.0-parquet --repo-type dataset --include "*.parquet"


mkdir -p models
# 下载 Llama-3-8B-Instruct 作为分词器来源和 Distillation 的教师模型
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --local-dir models/Meta-Llama-3-8B-Instruct

python data/generate_general_data_parallel.py \
  --version v5.1 \
  --tkn_path models/Meta-Llama-3-8B-Instruct \
  --data_max_len 2048 \
  --num_workers 128



2.模型训练
wandb login --relogin
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

accelerate launch --main_process_port 12231 --config_file "configs/accel_ds_8h800_gas1.yaml" hf_trainer.py \
  --log_steps 100 \
  --max_grad_norm 1.0 \
  --learning-rate 1e-4 \
  --gradient_accumulation_steps 1 \
  --max_steps 208000 \
  --dataset_name mix_general_llama3_tokenized_v5.1/train.jsonl \
  --batch-size 1 \
  --data-max-len 2048 \
  --save_steps 20000 \
  --check_data_cls_loss False \
  --target_hidden_size 1536 \
  --kl_temperature 40 \
  --warmup-ratio 0.005 \
  --raw-model-name models/Meta-Llama-3-8B-Instruct \
  --extra_tags general_train,8h800,arch,try_sota,all_ffn,all_attn \
  --use_accelerate True \
  --output_dir ckpts \
  --str_ban_losses no \
  --tie_word_emb_proj 1 \
  --use_all_attn 1 \
  --aux_loss_scale_factor 0.2 \
  2>&1 | tee accel_ds_8h800_gas1_20260331_1.log



1. 合并 DeepSpeed 权重 (Checkpoint Consolidation)
由于使用了 DeepSpeed 训练，得到的权重是分片存储的（位于 global_step208000 文件夹下）。需要先将它们合并成一个完整的 model.safetensors 文件。

# 进入你的工作目录
cd /home/ubuntu/wenhui/LowRankClone
# 使用生成的 zero_to_fp32.py 脚本合并权重
# 第一个参数是 checkpoint 的父目录，第二个参数是输出目录
python ckpts/train.jsonl-v4-general_train-8h800-arch-try_sota-all_ffn-all_attn/032808/zero_to_fp32.py \
  ckpts/train.jsonl-v4-general_train-8h800-arch-try_sota-all_ffn-all_attn/032808/ \
  ckpts/train.jsonl-v4-general_train-8h800-arch-try_sota-all_ffn-all_attn/032808/ \
  --safe_serialization \
  --tag global_step208000 \
  --max_shard_size 100GB

2. 转换为标准 Hugging Face 模型 (Checkpoint Conversion)
接下来，使用 convert_ckpt.py 脚本将 LRC 结构转换为标准的 Hugging Face 模型格式，以便后续使用 transformers 或 vLLM 加载。
注意：参数必须与你训练时使用的参数严格一致（target-hidden-size 1536, use-all-attn 1, tie-word-emb-proj 1）。

python convert_ckpt.py \
  --ckpt-path ckpts/train.jsonl-v4-general_train-8h800-arch-try_sota-all_ffn-all_attn/032808/model.safetensors \
  --target-hidden-size 1536 \
  --raw-model-name models/Meta-Llama-3-8B-Instruct \
  --save-path ckpts/LRC-2.7B-Converted \
  --use-all-attn 1 \
  --tie-word-emb-proj 1

3. Evaluation
lm_eval --model hf \
  --tasks "sciq,piqa,winogrande,arc_easy" \
  --model_args pretrained=ckpts/LRC-2.7B-Converted

4. SFT


