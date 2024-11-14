from dataclasses import dataclass
from codetrace.utils import load_dataset
import os
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
import torch
import argparse
from tqdm import tqdm
from pathlib import Path
from glob import glob
from functools import partial
from accelerate.utils import DataLoaderConfiguration
from accelerate import Accelerator
import datasets 
from typing import List, Dict,Any
from codetrace.parsing_utils import FimObj, get_model_fim

@dataclass
class TrainingArgs:
    batch_size:int
    max_steps:int
    gradient_accumulation_steps:int
    warmup_steps:int
    lr:float
    weight_decay:float
    max_grad_norm:float
    save_strategy:str
    eval_strategy:str
    save_steps:int
    eval_steps:int
    checkpoint_dir:Path

    def __init__(self,**kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

def prepare_tokenizer(model:str):
    tokenizer = AutoTokenizer.from_pretrained(model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer

def save(model, dirpath, accelerator=None):
    if accelerator:
        if accelerator.is_main_process:
            accelerator.save_model(model,dirpath, safe_serialization=False)
            accelerator.unwrap_model(model).config.save_pretrained(dirpath)
        accelerator.wait_for_everyone()
    else:
        model.save_pretrained(dirpath)
        model.config.save_pretrained(dirpath)

def evaluate(model, accelerator, eval_dataloader):
    tot_eval_loss = 0
    for i,batch in tqdm(enumerate(eval_dataloader), desc="Evaluating"):
        with torch.no_grad():
            output = model.forward(**batch)
            loss = output.loss
            log = {
                "eval_loss":loss,
                "eval_step": i,
            }
            tot_eval_loss +=loss
            accelerator.print(log)
    return {"total_eval_loss": tot_eval_loss, "mean_eval_loss": tot_eval_loss / len(eval_dataloader)}

def custom_collate_fn(data:List[Dict[str,Any]], tokenizer:AutoTokenizer, fim_obj:FimObj):
    inputs = [fim_obj.placeholder_to_fim(d["fim_program"]) for d in data]
    labels = [d["fim_type"] for d in data]
    inputs = tokenizer(inputs, return_tensors="pt", padding=True, padding_side="left")
    labels = tokenizer(labels, return_tensors="pt", padding=False, add_special_tokens=False)
    padded_labels = torch.zeros_like(inputs["input_ids"]).fill_(-100)
    for i, label in enumerate(labels["input_ids"]):
        padded_labels[i, -1] = label 

    return {
        "input_ids": inputs["input_ids"],
        "labels": padded_labels
    }

def get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    # https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    if len(grads) == 0:
        # print("Detected none grads")
        # note if using deepspeed this will always be the case
        return torch.tensor(0.)
    first_device = grads[0].device
    grouped_grads = torch.utils._foreach_utils._group_tensors_by_device_and_dtype([grads])

    norms = []
    for ((device, _), ([device_grads], _)) in grouped_grads.items():
        if torch.utils._foreach_utils._has_foreach_support(device_grads, device):
            norms.extend(torch._foreach_norm(device_grads, norm_type))
        else:
            norms.extend([torch.linalg.vector_norm(g, norm_type) for g in device_grads])

    total_norm = torch.linalg.vector_norm(torch.stack([norm.to(first_device) for norm in norms]), norm_type)
    return total_norm.item()

def train(
    model,
    tokenizer,
    train_ds,
    eval_ds,
    accelerator,
    training_args:TrainingArgs,
    fim_obj:FimObj
):
    
    train_dataloader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=training_args.batch_size,
        collate_fn=partial(custom_collate_fn, tokenizer=tokenizer, fim_obj=fim_obj)
    )
    eval_dataloader = torch.utils.data.DataLoader(
        eval_ds,
        batch_size=training_args.batch_size,
        collate_fn=partial(custom_collate_fn, tokenizer=tokenizer, fim_obj=fim_obj)
    )
    optim = torch.optim.AdamW(model.parameters(), lr=training_args.lr, weight_decay=training_args.weight_decay)
    scheduler = get_scheduler(
        name="cosine",
        optimizer=optim,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=training_args.max_steps // training_args.gradient_accumulation_steps,
    )
    
    trainable_params = [k for (k,p) in model.named_parameters() if p.requires_grad]
    accelerator.print(f"Len trainable params: {len(trainable_params)}")
    
    model, optim, train_dataloader, eval_dataloader, scheduler = accelerator.prepare(
        model, 
        optim, 
        train_dataloader,
        eval_dataloader,
        scheduler
    )
    torch.autograd.set_detect_anomaly(True)
    
    for epoch in range(training_args.num_epochs):
        accelerator.print(f"Epoch: {epoch}")
        
        for step,batch in tqdm(enumerate(train_dataloader), total=training_args.max_steps // training_args.num_epochs, desc="Step"):
            if (step * (epoch+1)) > training_args.max_steps:
                break
            
            # model forward + compute loss
            output = model.forward(**batch)
            loss = output.loss
            accelerator.backward(loss)

            # log metrics
            grad_norm = get_grad_norm(model.parameters())
            logs = {"step":step,
                    "train_loss":loss.item(), 
                    "epoch":epoch, 
                    "grad_norm":grad_norm, # will be 0 if using deepspeed, handled internally
                    "lr":optim.param_groups[0]['lr']}
            accelerator.print(logs)
            wandb.log(logs, step=step)
            
            # update when grad acc
            if ((step + 1) % training_args.gradient_accumulation_steps == 0) or (step + 1 == len(train_dataloader)):
                if training_args.max_grad_norm and accelerator.sync_gradients:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), training_args.max_grad_norm)
                optim.step()
                scheduler.step()
                optim.zero_grad()
                
            # eval and save
            if training_args.eval_strategy == "steps" and (step+1) % training_args.eval_steps == 0:
                eval_res = evaluate(model, accelerator, eval_dataloader)
                accelerator.print(eval_res)
                wandb.log(eval_res, step=step)

            if training_args.save_strategy == "steps" and (step+1) % training_args.save_steps == 0:
                save(model,os.path.join(training_args.checkpoint_dir, f"ckpt_{(step+1)*(epoch+1)}"), accelerator)
         
        # epoch eval and save       
        if training_args.eval_strategy == "epoch":
            eval_res = evaluate(model, accelerator, eval_dataloader)
            wandb.log(eval_res, step=step)
            accelerator.print(eval_res)
            
        if training_args.save_strategy == "epoch":
            save(model,os.path.join(training_args.checkpoint_dir, f"ckpt_{(step+1)*(epoch+1)}"), accelerator)
    
    accelerator.wait_for_everyone()
    # final model
    save(model,os.path.join(training_args.checkpoint_dir, f"ckpt_final_{(step+1)*training_args.num_epochs}"), accelerator)
    accelerator.end_training()


def main(args):
    wandb_mode = "online" if args.report_to == "wandb" else "offline"
    wandb.init(name=args.run_name, project="codetrace_ft", mode=wandb_mode, dir=args.logdir_name)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=getattr(torch, args.dtype),
        attn_implementation="flash_attention_2"
    )
    model = model.to(device=args.device)
    model.train()

    # dataloader_config = DataLoaderConfiguration(dispatch_batches=False)
    accelerator = Accelerator(project_dir=args.checkpoint_dir)
                            #   ,dataloader_config=dataloader_config)
    
    tokenizer = prepare_tokenizer(args.model)
    fim_obj = get_model_fim(args.model)
    ds = load_dataset(args.dataset, args.split)
    ds = ds.train_test_split(args.eval_size)
    train_ds,eval_ds = ds["train"], ds["test"]


    if not args.max_steps:
        args.max_steps = ((len(train_ds) // args.batch_size) // torch.cuda.device_count()) * args.num_epochs
    accelerator.print(f"Approximate eval size {len(eval_ds)}, train size: {len(train_ds)}, max train steps {args.max_steps}")
        
    training_args = TrainingArgs(**args.__dict__)
    train(model, tokenizer, train_ds, eval_ds, accelerator, training_args,fim_obj)

if __name__=="__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--model", required=True)
    
    # torch and data load args
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", type=str, choices=["float32","bfloat16"], default="bfloat16")
    parser.add_argument("--max-n", type=int, default=-1)
    parser.add_argument("--max-steps", type=int, default=None)
    
    # save args: everything will be saved under output_dir/run_name
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", required=True) 
    parser.add_argument("--logdir-name", type=str, default=None)
    
    # train args
    parser.add_argument("--save-steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--report-to", type=str, choices=["wandb", None], default=None)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--layer-norm-eps", type=float, default=1e-5)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--eval-strategy", choices=["steps","epoch","no"], default="steps")
    parser.add_argument("--eval-size", type=float, default=0.01)
    parser.add_argument("--save-strategy", type=str, choices=["steps","epoch"], default="steps")
    args = parser.parse_args()
    
    # init output dir
    args.checkpoint_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # init log dir
    if not args.logdir_name:
        args.logdir_name = os.path.join(args.checkpoint_dir, "logs")
    os.makedirs(args.logdir_name, exist_ok=True)
    
    # run accelerate config to set deepspeed, gpus etc.
    main(args)