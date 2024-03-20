from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import datasets
from tqdm import tqdm
from multiprocessing import cpu_count
import wandb
from torch import nn
import pandas as pd
from torch.utils.data.dataloader import default_collate
from codetrace.utils import placeholder_to_std_fmt, STARCODER_FIM
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from collections import Counter
from transformers import get_scheduler
from torch.optim import AdamW
"""
Based on https://huggingface.co/blog/codeparrot
"""

def get_grouped_params(model, args, no_decay=["bias", "LayerNorm.weight"]):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay): params_without_wd.append(p)
        else: params_with_wd.append(p)
    return [{"params": params_with_wd, "weight_decay": args.weight_decay},
            {"params": params_without_wd, "weight_decay": 0.0},]

def evaluate(model, eval_dataloader, is_ood, args):
    """
    Unlike standard validation, this computes accuracy
    """
    if is_ood:
        ood_flag = "ood_"
    else:
        ood_flag = ""
    model.eval()
    losses = []
    correct_predictions = 0
    total_num_predictions = 0
    for step, batch in tqdm(enumerate(eval_dataloader), desc=f"{ood_flag}Eval step", total=len(eval_dataloader)):
        with torch.no_grad():
            outputs = model(batch, labels=batch)
            loss = outputs.loss
        losses.append(loss)
        if args.do_log:
            args.logger.log({f"{ood_flag}eval_loss": loss.item(), 
                           f"{ood_flag}eval_step": step,})
        logits = outputs.logits[:,-2,:] # last token prediction
        output_prediction = torch.argmax(logits, dim=-1)
        labels = batch[:, -1] # last token label
        assert output_prediction.shape == labels.shape, f"Output shape: {output_prediction.shape}, labels shape: {labels.shape}"
        correct_predictions += torch.sum(torch.argmax(logits, dim=-1) == labels).item()
        total_num_predictions += len(labels)
        if args.do_log:
            args.logger.log({f"{ood_flag}eval_accuracy": correct_predictions/total_num_predictions, 
                           f"{ood_flag}eval_step": step,
                           f"{ood_flag}eval_total": total_num_predictions,
                           f"{ood_flag}eval_numcorrect": correct_predictions,})
    loss = torch.mean(torch.stack(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    return loss.item(), perplexity.item()

def train_loop(model, train_dataloader, eval_dataloader, ood_evaldataloader, args):
    optimizer = AdamW(get_grouped_params(model, args), lr=args.learning_rate)
    lr_scheduler = get_scheduler(name=args.lr_scheduler_type, optimizer=optimizer,
                                num_warmup_steps=args.num_warmup_steps,
                                num_training_steps=args.n_epochs * len(train_dataloader))
    
    evaluate(model, eval_dataloader, False, args)
    evaluate(model, ood_evaldataloader, True, args)
    
    model.train()
    total_steps = 0
    for epoch in tqdm(range(args.n_epochs), desc="Epoch", total=args.n_epochs):
        for step, batch in enumerate(train_dataloader, start=1):
            total_steps += 1
            outputs = model(batch, labels=batch, use_cache=False)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            if args.do_log: 
                args.logger.log({"train_loss": loss.item(), 
                            "epoch": epoch, 
                            "step": total_steps,})
            if step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

        # save each epoch
        eval_loss, perplexity = evaluate(model, eval_dataloader, False, args)
        ood_eval_loss, ood_perplexity = evaluate(model, ood_evaldataloader,True, args)
        print(f"Epoch {epoch} eval loss: {eval_loss}, eval perplexity: {perplexity}")
        print(f"Epoch {epoch} ood eval loss: {ood_eval_loss}, ood eval perplexity: {ood_perplexity}")
        model.save_pretrained(args.checkpoints_dir + f"/checkpoint_{epoch}")
        model.train()
    
    # save final model
    model.save_pretrained(args.checkpoints_dir + "/checkpoint_final")
    return model


def main(args):
    if args.do_log:
        args.logger = wandb
        wandb.init(project=args.project_name)
    
    device = f"cuda:{args.gpu}"
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    ds = datasets.load_dataset(args.ds, split="train")
    # filter oom
    ds = ds.filter(lambda x: len(x["fim_program"]) < 8000)
    ds = ds.shuffle(seed=42)
    ood_ds = datasets.load_dataset(args.ood_ds, split="train")
    ood_ds = ood_ds.filter(lambda x: len(x["fim_program"]) < 8000)
    ood_ds = ood_ds.shuffle(seed=42)
    
    # prep items with tokenizer
    prompts = ds[args.items]
    labels = ds[args.labels]
    prompts = [placeholder_to_std_fmt(p, STARCODER_FIM)+l for (p,l) in list(zip(prompts, labels))]
    train_items = tokenizer(prompts, return_tensors="pt", padding=True).input_ids
    
    ood_prompts = ood_ds[args.items]
    ood_labels = ood_ds[args.labels]
    ood_prompts = [placeholder_to_std_fmt(p, STARCODER_FIM)+l for (p,l) in list(zip(ood_prompts, ood_labels))]
    ood_items = tokenizer(ood_prompts, return_tensors="pt", padding=True).input_ids
        
    train_ds, val_ds = train_test_split(train_items, test_size=args.test_size)
    
    if args.max_size > -1:
        train_ds = train_ds[:args.max_size]
        val_ds = val_ds[:args.eval_max_size]
        ood_items = ood_items[:args.eval_max_size]

    # cast to device
    model = model.to(device)
        
    train_ds = torch.utils.data.DataLoader(train_ds, 
                                           batch_size=args.batch_size,
                                           collate_fn=lambda x: default_collate(x).to(device),
                                            shuffle=True)
    val_ds = torch.utils.data.DataLoader(val_ds, 
                                         batch_size=args.batch_size,
                                         collate_fn=lambda x: default_collate(x).to(device),
                                        shuffle=True)
    ood_evaldataloader = torch.utils.data.DataLoader(ood_items,
                                                    batch_size=args.batch_size,
                                                    collate_fn=lambda x: default_collate(x).to(device),
                                                    shuffle=True)
    print("Train size, val size:", len(train_ds), len(val_ds))
    print("OOD eval size:", len(ood_evaldataloader))
    
    train_loop(model, train_ds, val_ds, ood_evaldataloader, args)
    if args.do_log:
        wandb.finish()
        
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="/home/arjun/models/starcoderbase-1b")
    parser.add_argument("--ds", type=str, required=True)
    parser.add_argument("--ood_ds", type=str, required=True)
    parser.add_argument("--labels", type=str, default="fim_type")
    parser.add_argument("--items", type=str, default="fim_program")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--max-size", type=int, required=True)
    parser.add_argument("--eval-max-size", type=int, required=True)
    
    # train hyperparams (from MultiPL-T)
    parser.add_argument("--n_epochs", type=int, default=7)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=10)
    # simulate large batch size
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)

    # saving and logging
    parser.add_argument("--checkpoints_dir", type=str, required=True)
    parser.add_argument("--do-log", action="store_true", default=False)
    parser.add_argument("--project-name", type=str, default=None)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    main(args)
    
