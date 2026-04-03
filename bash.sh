1.数据整理
cd /home/ubuntu/wenhui/LowRankClone
mkdir -p datasets

hf download HuggingFaceTB/smollm-corpus --repo-type dataset --local-dir datasets --include "fineweb-edu-dedup/*.parquet"
hf download HuggingFaceTB/smollm-corpus --repo-type dataset --local-dir datasets --include "cosmopedia-v2/*"
hf download teknium/OpenHermes-2.5 --repo-type dataset --local-dir datasets/OpenHermes-2.5
hf download HuggingFaceH4/ultrachat_200k --repo-type dataset --local-dir datasets/ultrachat_200k
# 全量下载命令（需要极大的硬盘空间，慎用！）
hf download togethercomputer/RedPajama-Data-1T --repo-type dataset --local-dir datasets/RedPajama-Data-1T
hf download togethercomputer/RedPajama-Data-1T-Sample --repo-type dataset --local-dir datasets/RedPajama-Data-1T-Sample
hf download mlfoundations/dclm-baseline-1.0-parquet --local-dir datasets/dclm-baseline-1.0-parquet --repo-type dataset --include "*.parquet"


mkdir -p models
# 下载 Llama-3-8B-Instruct 作为分词器来源和 Distillation 的教师模型
hf download meta-llama/Meta-Llama-3-8B-Instruct --local-dir models/Meta-Llama-3-8B-Instruct
hf download meta-llama/Llama-3.2-3B-Instruct --local-dir models/Llama-3.2-3B-Instruct

python data/generate_general_data_parallel.py \
  --version v5.1 \
  --tkn_path models/Llama-3.2-3B-Instruct \
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
  --extra_tags general_train,8h100,arch,try_sota,all_ffn,all_attn \
  --use_accelerate True \
  --output_dir ckpts \
  --str_ban_losses no \
  --tie_word_emb_proj 1 \
  --use_all_attn 1 \
  --aux_loss_scale_factor 0.2 \
  2>&1 | tee accel_ds_8h800_gas1_20260331_1.log


accelerate launch --main_process_port 12231 --config_file "configs/accel_ds_8h800_gas1.yaml" hf_trainer.py \
  --log_steps 100 \
  --max_grad_norm 1.0 \
  --learning-rate 1e-4 \
  --gradient_accumulation_steps 1 \
  --max_steps 208000 \
  --dataset_name mix_general_llama3_tokenized_v5.1_10b/train.jsonl \
  --batch-size 3 \
  --data-max-len 2048 \
  --save_steps 20000 \
  --check_data_cls_loss False \
  --target_hidden_size 1536 \
  --kl_temperature 40 \
  --warmup-ratio 0.005 \
  --raw-model-name models/Llama-3.2-3B-Instruct \
  --extra_tags general_train,8h100,arch,try_sota,all_ffn,all_attn \
  --use_accelerate True \
  --output_dir ckpts \
  --str_ban_losses no \
  --tie_word_emb_proj 1 \
  --use_all_attn 1 \
  --aux_loss_scale_factor 0.2 \
  2>&1 | tee accel_ds_8h800_gas1_20260401.log



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

python convert_ckpt.py \
  --ckpt-path ckpts/train.jsonl-v4-general_train-8h100-arch-try_sota-all_ffn-all_attn/040112/model.safetensors \
  --target-hidden-size 1536 \
  --raw-model-name models/Llama-3.2-3B-Instruct \
  --save-path ckpts/LRC-1.5B-Converted \
  --use-all-attn 1 \
  --tie-word-emb-proj 1

3. Evaluation
lm_eval --model hf \
  --tasks "sciq,piqa,winogrande,arc_easy" \
  --model_args pretrained=ckpts/LRC-2.7B-Converted

lm_eval --model hf \
  --tasks "sciq,piqa,winogrande,arc_easy" \
  --model_args pretrained=ckpts/LRC-1.5B-Converted


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes 8 \
  -m lm_eval \
  --model hf \
  --tasks "arc_easy,arc_challenge,logiqa,commonsense_qa,piqa,winogrande,boolq,sciq" \
  --num_fewshot 0 \
  --batch_size auto \
  --model_args pretrained=ckpts/LRC-1.5B-Converted

lm_eval --model hf \
  --tasks "mmlu" \
  --model_args pretrained=ckpts/LRC-1.5B-Converted

hf (pretrained=ckpts/LRC-1.5B-Converted), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 1
|                 Tasks                 |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|---------------------------------------|------:|------|-----:|------|---|-----:|---|-----:|
|mmlu                                   |      2|none  |      |acc   |↑  |0.4966|±  |0.0041|
| - humanities                          |      2|none  |      |acc   |↑  |0.4529|±  |0.0069|
|  - formal_logic                       |      1|none  |     0|acc   |↑  |0.2937|±  |0.0407|
|  - high_school_european_history       |      1|none  |     0|acc   |↑  |0.6606|±  |0.0370|
|  - high_school_us_history             |      1|none  |     0|acc   |↑  |0.6863|±  |0.0326|
|  - high_school_world_history          |      1|none  |     0|acc   |↑  |0.7131|±  |0.0294|
|  - international_law                  |      1|none  |     0|acc   |↑  |0.6364|±  |0.0439|
|  - jurisprudence                      |      1|none  |     0|acc   |↑  |0.5648|±  |0.0479|
|  - logical_fallacies                  |      1|none  |     0|acc   |↑  |0.6196|±  |0.0381|
|  - moral_disputes                     |      1|none  |     0|acc   |↑  |0.5434|±  |0.0268|
|  - moral_scenarios                    |      1|none  |     0|acc   |↑  |0.2380|±  |0.0142|
|  - philosophy                         |      1|none  |     0|acc   |↑  |0.5659|±  |0.0282|
|  - prehistory                         |      1|none  |     0|acc   |↑  |0.5679|±  |0.0276|
|  - professional_law                   |      1|none  |     0|acc   |↑  |0.3722|±  |0.0123|
|  - world_religions                    |      1|none  |     0|acc   |↑  |0.6140|±  |0.0373|
| - other                               |      2|none  |      |acc   |↑  |0.5433|±  |0.0088|
|  - business_ethics                    |      1|none  |     0|acc   |↑  |0.5100|±  |0.0502|
|  - clinical_knowledge                 |      1|none  |     0|acc   |↑  |0.5358|±  |0.0307|
|  - college_medicine                   |      1|none  |     0|acc   |↑  |0.4971|±  |0.0381|
|  - global_facts                       |      1|none  |     0|acc   |↑  |0.2700|±  |0.0446|
|  - human_aging                        |      1|none  |     0|acc   |↑  |0.5605|±  |0.0333|
|  - management                         |      1|none  |     0|acc   |↑  |0.6311|±  |0.0478|
|  - marketing                          |      1|none  |     0|acc   |↑  |0.7393|±  |0.0288|
|  - medical_genetics                   |      1|none  |     0|acc   |↑  |0.6100|±  |0.0490|
|  - miscellaneous                      |      1|none  |     0|acc   |↑  |0.5709|±  |0.0177|
|  - nutrition                          |      1|none  |     0|acc   |↑  |0.6209|±  |0.0278|
|  - professional_accounting            |      1|none  |     0|acc   |↑  |0.3652|±  |0.0287|
|  - professional_medicine              |      1|none  |     0|acc   |↑  |0.5257|±  |0.0303|
|  - virology                           |      1|none  |     0|acc   |↑  |0.4518|±  |0.0387|
| - social sciences                     |      2|none  |      |acc   |↑  |0.5720|±  |0.0087|
|  - econometrics                       |      1|none  |     0|acc   |↑  |0.3246|±  |0.0440|
|  - high_school_geography              |      1|none  |     0|acc   |↑  |0.6566|±  |0.0338|
|  - high_school_government_and_politics|      1|none  |     0|acc   |↑  |0.6736|±  |0.0338|
|  - high_school_macroeconomics         |      1|none  |     0|acc   |↑  |0.4564|±  |0.0253|
|  - high_school_microeconomics         |      1|none  |     0|acc   |↑  |0.4874|±  |0.0325|
|  - high_school_psychology             |      1|none  |     0|acc   |↑  |0.7266|±  |0.0191|
|  - human_sexuality                    |      1|none  |     0|acc   |↑  |0.5649|±  |0.0435|
|  - professional_psychology            |      1|none  |     0|acc   |↑  |0.4575|±  |0.0202|
|  - public_relations                   |      1|none  |     0|acc   |↑  |0.5545|±  |0.0476|
|  - security_studies                   |      1|none  |     0|acc   |↑  |0.5714|±  |0.0317|
|  - sociology                          |      1|none  |     0|acc   |↑  |0.7264|±  |0.0315|
|  - us_foreign_policy                  |      1|none  |     0|acc   |↑  |0.7200|±  |0.0451|
| - stem                                |      2|none  |      |acc   |↑  |0.4421|±  |0.0086|
|  - abstract_algebra                   |      1|none  |     0|acc   |↑  |0.3100|±  |0.0465|
|  - anatomy                            |      1|none  |     0|acc   |↑  |0.4963|±  |0.0432|
|  - astronomy                          |      1|none  |     0|acc   |↑  |0.5592|±  |0.0404|
|  - college_biology                    |      1|none  |     0|acc   |↑  |0.5833|±  |0.0412|
|  - college_chemistry                  |      1|none  |     0|acc   |↑  |0.4000|±  |0.0492|
|  - college_computer_science           |      1|none  |     0|acc   |↑  |0.4600|±  |0.0501|
|  - college_mathematics                |      1|none  |     0|acc   |↑  |0.3200|±  |0.0469|
|  - college_physics                    |      1|none  |     0|acc   |↑  |0.3235|±  |0.0466|
|  - computer_security                  |      1|none  |     0|acc   |↑  |0.5900|±  |0.0494|
|  - conceptual_physics                 |      1|none  |     0|acc   |↑  |0.4298|±  |0.0324|
|  - electrical_engineering             |      1|none  |     0|acc   |↑  |0.5310|±  |0.0416|
|  - elementary_mathematics             |      1|none  |     0|acc   |↑  |0.3280|±  |0.0242|
|  - high_school_biology                |      1|none  |     0|acc   |↑  |0.6484|±  |0.0272|
|  - high_school_chemistry              |      1|none  |     0|acc   |↑  |0.4680|±  |0.0351|
|  - high_school_computer_science       |      1|none  |     0|acc   |↑  |0.5100|±  |0.0502|
|  - high_school_mathematics            |      1|none  |     0|acc   |↑  |0.3074|±  |0.0281|
|  - high_school_physics                |      1|none  |     0|acc   |↑  |0.4172|±  |0.0403|
|  - high_school_statistics             |      1|none  |     0|acc   |↑  |0.3935|±  |0.0333|
|  - machine_learning                   |      1|none  |     0|acc   |↑  |0.3304|±  |0.0446|

|      Groups      |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|------------------|------:|------|------|------|---|-----:|---|-----:|
|mmlu              |      2|none  |      |acc   |↑  |0.4966|±  |0.0041|
| - humanities     |      2|none  |      |acc   |↑  |0.4529|±  |0.0069|
| - other          |      2|none  |      |acc   |↑  |0.5433|±  |0.0088|
| - social sciences|      2|none  |      |acc   |↑  |0.5720|±  |0.0087|
| - stem           |      2|none  |      |acc   |↑  |0.4421|±  |0.0086|




4. SFT
FORCE_TORCHRUN=1 llamafactory-cli train /home/ubuntu/wenhui/LowRankClone/configs/llama_factory/llama3-sft-full.yaml 2>&1 | tee accel_ds_8h800_gas1_20260403_sft.log










训练数据语料统计
开始统计: mix_general_llama3_tokenized_v5.1_10b/fineweb-edu-dedup.jsonl
mix_general_llama3_tokenized_v5.1_10b/fineweb-edu-dedup.jsonl 统计完成: 47.859 B (47,858,743,296 tokens)

开始统计: mix_general_llama3_tokenized_v5.1_10b/hermes.jsonl
mix_general_llama3_tokenized_v5.1_10b/hermes.jsonl 统计完成: 370.438 M (370,438,144 tokens)

========================================
统计汇总：
 - fineweb-edu-dedup.jsonl: 47.859 B
 - hermes.jsonl: 370.438 M
----------------------------------------
总计 Token 量: 48.229 B (48,229,181,440 tokens)
========================================