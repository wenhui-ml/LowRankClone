import random
import fire
import wandb
import torch
import os
import json
from torch.utils.data import IterableDataset

orig_torch_load = torch.load

def torch_wrapper(*args, **kwargs):
    print("[comfyui-unsafe-torch] I have unsafely patched `torch.load`.  The `weights_only` option of `torch.load` is forcibly disabled.")
    kwargs['weights_only'] = False
    return orig_torch_load(*args, **kwargs)

torch.load = torch_wrapper

NODE_CLASS_MAPPINGS = {}
__all__ = ['NODE_CLASS_MAPPINGS']

from transformers import AutoTokenizer, Trainer, TrainingArguments, LlamaForCausalLM, Qwen2ForCausalLM, set_seed
from datasets import IterableDatasetDict
from tools.global_state import hyper_params
from tools import global_state
from tools.assign_device_map import assign_device_map
from accelerate import Accelerator
from datetime import datetime

torch.set_num_threads(8)
print("init accelerate")
accelerator = Accelerator()
print("init accelerate done.")

def get_current_time_short():
    now = datetime.now()
    time_str = now.strftime('%m%d%H')
    return time_str

# ================= 🚀 208核硬核分布式流式读取器 =================
class DistributedShardedJsonlDataset(IterableDataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.cls_map = {"general": 0, "sft": 1}

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = 1 if worker_info is None else worker_info.num_workers
        worker_id = 0 if worker_info is None else worker_info.id

        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        global_num_workers = world_size * num_workers
        global_worker_id = rank * num_workers + worker_id

        with open(self.file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i % global_num_workers == global_worker_id:
                    try:
                        data = json.loads(line)
                        cls_str = data.get("data_cls", "general")
                        cls_int = self.cls_map.get(cls_str, 0)
                        
                        yield {
                            "input_ids": data["input_ids"],
                            "data_cls": cls_int
                        }
                    except Exception:
                        continue

# ================= 🛡️ 绝对安全的数据批次拼接器 (Collator) =================
def custom_data_collator(features):
    """
    不管 batch_size 是 1 还是 100，严格保证返回 2D 矩阵！
    彻底解决 `expected 3, got 2` 的维度坍缩问题。
    """
    input_ids = [f["input_ids"] for f in features]
    data_cls = [f.get("data_cls", 0) for f in features]
    
    batch = {
        # 强制组装成 [batch_size, seq_len] 的 2D 矩阵
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        # 补齐 Attention Mask 确保模型不会报错
        "attention_mask": torch.ones((len(input_ids), len(input_ids[0])), dtype=torch.long),
        # 语言模型自回归需要 labels
        "labels": torch.tensor(input_ids, dtype=torch.long),
        "data_cls": torch.tensor(data_cls, dtype=torch.long),
    }
    return batch
# =======================================================================

def train_model(
    target_hidden_size=1024,
    raw_model_name="models/Meta-Llama-3-8B-Instruct",
    model_cls = "distill",
    dataset_name="../datasets/squad_v2",
    output_dir="../ckpts",
    num_epochs=1,
    batch_size=4,
    learning_rate=1e-4,
    warmup_ratio=0.005,
    target_rms_norm_eps=1e-5,
    gradient_accumulation_steps=1,
    log_steps=100,
    save_steps=20000,
    max_steps=-1,
    data_max_len=1024,
    project_name="expt-small-llm",
    tie_n=-1,
    tie_word_emb_proj=False,
    max_grad_norm=1.0,
    aux_loss_scale_factor=1.0,
    use_aux_loss=True,
    use_logits_loss=True,
    use_std_like_attn=False,
    student_attn_from_scratch=False,
    del_layers="",
    ban_layers="",
    use_in_out_mlp=False,
    use_all_attn=False,
    use_additional_align=False,
    gpus=1,
    resume_checkpoint=None,
    load_model_weight_path=None,
    check_data_cls_loss=False,
    extra_tags="ordinary",
    kl_temperature=10.0,
    lr_scheduler="linear",
    aux_loss_type="mseloss",
    use_ntp_loss=True,
    str_ban_losses="no",
    use_accelerate=False,
    adam_beta2=0.999,
):  
    hyper_params["gradient_accumulation_steps"] = gradient_accumulation_steps
    hyper_params["aux_loss_scale_factor"] = aux_loss_scale_factor
    
    if "llama" in raw_model_name.lower():
        from modeling.co_train_llama import CoTrainLM, CustomConfig, reinit_weight
    elif "qwen" in raw_model_name.lower():
        from modeling.co_train_qwen import CoTrainLM, CustomConfig, reinit_weight
    else:
        raise ValueError("Could not find corresponding teacher model")
        
    if isinstance(str_ban_losses, str):
        global_state.ban_losses += str_ban_losses.split(',')
    else:
        global_state.ban_losses += str_ban_losses
        
    if isinstance(del_layers, str):
        del_layers = [int(x) for x in del_layers.split(',') if len(x) > 0] 
    if isinstance(ban_layers, str):
        ban_layers = [int(x) for x in ban_layers.split(',') if len(x) > 0]
    global_state.ban_layers += ban_layers

    print(f"(for debug) use-aux-loss: {use_aux_loss}")
    print(f"(for debug) tie word emb proj: {tie_word_emb_proj}")
    print(f"(for debug) use logits/kl loss: {use_logits_loss}")
    set_seed(429)
    
    if "tokenize" not in dataset_name:
        tokenizer = AutoTokenizer.from_pretrained(raw_model_name, use_fast=True) 
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = None

    print(f"🚀 Loading High-Performance Sharded Dataset from: {dataset_name}")
    raw_dataset = DistributedShardedJsonlDataset(dataset_name)
    tokenized_datasets = IterableDatasetDict({"train": raw_dataset})

    config = CustomConfig.from_pretrained(raw_model_name)
    config.set_custom_kwargs(
        target_hidden_size=target_hidden_size, 
        target_rms_norm_eps=target_rms_norm_eps,
        use_aux_loss=use_aux_loss,
        use_std_like_attn=use_std_like_attn,
        check_data_cls_loss=check_data_cls_loss,
        kl_temperature=kl_temperature,
        aux_loss_type=aux_loss_type,
        use_logits_loss=use_logits_loss,
        use_ntp_loss=use_ntp_loss,
        student_attn_from_scratch=student_attn_from_scratch,
        tie_word_emb_proj=tie_word_emb_proj,
        del_layers=del_layers,
        use_in_out_mlp=use_in_out_mlp,
        use_all_attn=use_all_attn,
        use_additional_align=use_additional_align,
    )

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if accelerator.is_main_process:
        print(f"Main Process Local Rank is {local_rank}")

    if model_cls == "distill":
        model: CoTrainLM = (
            CoTrainLM.from_pretrained(
                raw_model_name, config=config, torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2" if not use_std_like_attn else "manual",
                device_map=None if use_accelerate else assign_device_map(raw_model_name, gpus=gpus, local_rank=local_rank),
            )
        )
        if tie_n > 1:
            model.tie_custom_weights(tie_n)
        model.freeze_original_model()
        model.apply(reinit_weight)

        if load_model_weight_path is not None:
            from safetensors.torch import load_file
            sd = load_file(load_model_weight_path)
            missed, unexpected = model.load_state_dict(sd, strict=False)
            assert len(unexpected) == 0
            print("loaded model weights.")

    elif model_cls == "origin":
        config._attn_implementation = "flash_attention_2"
        model = LlamaForCausalLM if "llama" in raw_model_name.lower() else Qwen2ForCausalLM
        model = model.to(dtype=torch.bfloat16, device="cuda:0") 
        for n, p in model.named_parameters():
            assert p.dtype == torch.bfloat16
            
    assert isinstance(extra_tags, str) or isinstance(extra_tags, tuple)
    data_real_name = os.path.split(dataset_name)[-1]
    if data_real_name.endswith("jsonl"):
        data_real_name = os.path.split(data_real_name)[-1]
    tags = [
        data_real_name, 
        "v4",
    ] + (extra_tags.split(",") if isinstance(extra_tags, str) else list(extra_tags))
    
    output_dir = os.path.join(*[output_dir, "-".join(tags), get_current_time_short()])
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)

    lr_scheduler_kwargs = {}
    if lr_scheduler == "warmup_stable_decay":
        lr_scheduler_kwargs = {
            "num_warmup_steps": int(max_steps * warmup_ratio) + 1,
            "num_stable_steps": int(max_steps * (0.9 - warmup_ratio)) + 1,
            "num_decay_steps": max_steps - (int(max_steps * warmup_ratio) + 1) - (int(max_steps * (0.9 - warmup_ratio)) + 1)
        }

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        logging_steps=log_steps,
        save_steps=save_steps,
        max_steps=max_steps,
        save_total_limit=10,
        bf16=True,  
        # 核心算力开关：保持 16 以榨干你的 208 核 CPU！
        dataloader_num_workers=16,
        dataloader_pin_memory=True,
        # 必须设为 False，防止 Trainer 把我们在 Collator 里生成的 data_cls 等关键列丢弃
        remove_unused_columns=False,
        max_grad_norm=max_grad_norm,
        logging_dir="./logs",
        report_to="wandb" if local_rank == 0 else "none",
        lr_scheduler_type=lr_scheduler,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
        adam_beta2=adam_beta2,
    )

    _upload_cfg = training_args.to_dict()
    _upload_cfg.update(config.to_dict())
    if local_rank == 0:
        wandb.init(
            project=project_name, 
            name=f"{extra_tags}-ths{target_hidden_size}lr{learning_rate}kl_t{kl_temperature}",
            config=_upload_cfg,
            tags=tags,
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        # 启用我们无懈可击的手写 Collator
        data_collator=custom_data_collator
    )

    trainer.train(resume_from_checkpoint=resume_checkpoint)
    
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()
    if tokenizer:
        tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    fire.Fire(train_model)