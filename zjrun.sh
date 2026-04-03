export WANDB_MODE=disabled
export WANDB_DISABLED=true
# 或者如果你有团队的 wandb name，请使用下面这行：
# export WANDB_ENTITY="your_team_name"

accelerate launch --main_process_port 12231 --config_file "configs/accel_ds_8h800.yaml" hf_zhoujie_trainer.py \
  --log_steps 100 \
  --max_grad_norm 1.0 \
  --learning-rate 1e-4 \
  --gradient_accumulation_steps 1 \
  --max_steps 208000 \
  --dataset_name tokenized_v5.1/train_part_1.jsonl;tokenized_v5.1/train_part_2.jsonl;tokenized_v5.1/train_part_3.jsonl;tokenized_v5.1/train_part_4.jsonl;tokenized_v5.1/train_part_5.jsonl;tokenized_v5.1/train_part_6.jsonl;tokenized_v5.1/train_part_7.jsonl;tokenized_v5.1/train_part_8.jsonl \
  --batch-size 1 \
  --data-max-len 2048 \
  --save_steps 20000 \
  --check_data_cls_loss False \
  --target_hidden_size 1536 \
  --kl_temperature 40 \
  --warmup-ratio 0.005 \
  --raw-model-name models/Meta-Llama-3-8B-Instruct \
  --use_accelerate True \
  --output_dir ckpts \
  --str_ban_losses no \
  --tie_word_emb_proj 1 \
  --use_all_attn 1 \
  --aux_loss_scale_factor 0.2 \
  2>&1 | tee accel_ds_8h800_gas1_20260331_zhoujie_1.log
