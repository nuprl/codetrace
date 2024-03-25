from argparse import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from copy import deepcopy
import datasets
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from multiprocessing import cpu_count
from codetrace.utils import *
from codetrace.interp_utils import *
import wandb
from torch import nn
import pandas as pd
from torch.utils.data.dataloader import default_collate

do_log = True
if do_log:
    logger = wandb
    wandb.init(project="tuned_lens")


def train_loop(model, train_ds, val_ds, n_epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    total_loss = 0
    total_steps = 0
    for epoch in tqdm(range(n_epochs), desc="Epoch"):
        for batch in train_ds:
            data, target = batch
            model.train()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            if do_log:
                logger.log({"train_loss": loss.item(), 
                           "total_loss": total_loss, 
                           "epoch": epoch, 
                           "step": total_steps,})
            total_steps += 1

        for batch in val_ds:
            model.eval()
            data, target = batch
            output = model(data)
            loss = criterion(output, target)
            if do_log:
                logger.log({"val_loss": loss.item()})
            
        if epoch % 3 == 0:
            torch.save(model.state_dict(), f"models/model_{epoch}.pt")
            
    torch.save(model.state_dict(), "models/model_final.pt")
    return model


"""
TODO: train inputs are activations from a specific layer, eg. 5
model output should be index into vocabulary
"""

def batched_collect_hs(model: LanguageModel,
                       prompts : List[str],
                       layer : int,
                       batch_size : int):
    hidden_states = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Batch collect hs"):
        batch = prompts[i:i+batch_size]
        with torch.no_grad():
            hs = collect_hidden_states_at_tokens(model, batch, token_idx="<fim_middle>",layers=[layer])
        hidden_states.append(hs)

    return hidden_states

def _prepare_data():
    pass

def main(args):
    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    decoder = deepcopy(model.lm_head)
    print(decoder)
    del(model)
    dataset = datasets.load_dataset(args.ds, split="train")
    # filter 1toks
    if "1tok" not in args.ds:
        dataset = dataset.filter(lambda x: len(tokenizer.encode(x[args.labels])) == 1, num_proc=cpu_count())
    # shuffle ds
    dataset = dataset.shuffle(seed=42)
    dataset = datasets.Dataset.from_pandas(pd.DataFrame(dataset[:100]))
    dataset = dataset.map(lambda x: {args.items: placeholder_to_std_fmt(x[args.items], STARCODER_FIM)}, num_proc=cpu_count())
    
    items = dataset[args.items]
    labels = dataset[args.labels]
    # get onehot tensors from tokenizing labels
    labels = [tokenizer.encode(label, return_tensors="pt") for label in labels]
    # turn into one-hot
    labels = [torch.nn.functional.one_hot(label, num_classes=tokenizer.vocab_size).float() for label in labels]
    labels = torch.cat(labels, dim=1)
    print(f"Labels shape:{labels.shape}") #[n_layer, n_prompts, n_vocab]
    
    # collect activations from layer
    model = LanguageModel(args.model, device_map=device)
    items = batched_collect_hs(model, items, args.layer, args.batch_size)
    items = torch.cat(items, dim=1)
    print(f"Train items shape:{items.shape}")
    # 
    del(model)
    
    items = items.squeeze(0)
    labels = labels.squeeze(0)
    items = [i[0] for i in items]
    train_items = list(zip(items, labels))

    train_ds, val_ds = train_test_split(train_items, test_size=args.test_size)
    
    # cast to device
    decoder = decoder.to(device)
    train_ds = torch.utils.data.DataLoader(train_ds, 
                                           batch_size=args.batch_size,
                                           collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)),
                                            shuffle=True)
    val_ds = torch.utils.data.DataLoader(val_ds, 
                                         batch_size=args.batch_size,
                                         collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)),
                                            shuffle=True)
    print("Train size, val size:", len(train_ds), len(val_ds))
    
    train_loop(decoder, train_ds, val_ds, args.n_epochs)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="/home/arjun/models/starcoderbase-1b")
    parser.add_argument("--labels", type=str, default="fim_type")
    parser.add_argument("--items", type=str, default="fim_program")
    parser.add_argument("--ds", type=str)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--layer", type=int)
    args = parser.parse_args()
    main(args)