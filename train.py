"""
Idea: given a prompt where LLM predicts <FILL> incorrectly, train a model CAUSAL_MASKER
top produce a MASK (matrix) that, when applied to LLM components, will produce the correct prediction.

Training data for CAUSAL_MASKER
input: LLM, MASK over components, prompt, incorrect prediction, target prediction
output: MASK over components

Architecture:

CAUSAL_MASKER takes a matrix and produces a new matrix
it has a loss that is computed:

    Loss= LLM_MASK(prompt)[logit_of_incorrect_prediction] - LLM_MASK(prompt)[logit_of_correct_prediction]
    
the goal is to minimize this loss, so that LLM_MASK(prompt) produces the correct prediction

IDEA: use a custom Convolutional Network

SIMPLE VERSION: 1 token prediction

TODO: training loop
model takes input mask, the output mask is then passed to next train item
loss is calculated over batch logits (each same mask thankfully)
mask needs to be applied in a batch 

Modle inp-out are the mask
Loss is derived from a function that uses mask applied to traiuning items
"""

"""
example of a mask for a model with 2 layers, 2 attention heads and an MLP dim of size 6
Note: mlp dim needs to be divisible by attn heads

dimensions are (l, a, (d//a)+1)

TODO: disconnected computation graph somewhere

"""
mask_example = [
    [[0,0,0, 1],
    [1,0,1,0]],
    [[1,1,0, 0],
    [0,0,0,0]]
]

from nnsight import LanguageModel, util
from nnsight.tracing.Proxy import Proxy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import sys
from src.utils import *
import einops
from typing import List
import random
import numpy as np
import datasets
import wandb
wandb_on = True

def kl_divergence_loss(prob_dist1, prob_dist2):
    # Calculate KL divergence between prob_dist1 and prob_dist2
    kl_divergence = F.kl_div(torch.log(prob_dist1), prob_dist2, reduction='batchmean')
    return kl_divergence

def relative_distance_loss(prob_dist1, prob_dist2):
    prob_dist1 = prob_dist1.mean()
    prob_dist2 = prob_dist2.mean()
    return prob_dist1 / prob_dist2


def apply_mask(llm: LanguageModel,
               mask: torch.Tensor,
               prompts: List[str],
               correct_idxs: List[int],
               incorrect_idxs: List[int],
               debug=False) -> List[int]:

    # normalize mask
    mask = z_score_normalize(mask)
    mask = (mask > 0.5).float()
    
    def randomize_values_with_mask(original_tensor, mask):
        torch.random.manual_seed(42)
        # Generate random values for the indices where the mask is equal to 1
        random_values = torch.rand_like(original_tensor) * mask
        
        # Apply the random values to the original tensor
        result_tensor = original_tensor * (1 - mask) + random_values
        
        return result_tensor
    llm.tokenizer.padding_side = "left"
    with llm.generate(max_new_tokens=1, 
                    pad_token_id=llm.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_attentions=True,) as generator:
        llm.tokenizer.pad_token_id = llm.tokenizer.eos_token_id
        with generator.invoke(prompts) as invoker:
            # apply mask

            for layer in range(llm.config.n_layer):
                
                target_shape = llm.transformer.h[layer].attn.c_proj.output.shape
                attn_mask = mask[layer].repeat(1,128).mT.reshape(1,2048)
                
                # apply mask to any attn heads
                llm.transformer.h[layer].attn.c_proj.output= randomize_values_with_mask(llm.transformer.h[layer].attn.c_proj.output, attn_mask)
                
                # target_shape_mlp = llm.transformer.h[layer].mlp.output.shape
                # # shape (1,1,16)
                # mlp_mask = mask[layer][1].repeat(1,128,1).mT.reshape(1,1,2048)

                # # apply mask to any attn heads
                # llm.transformer.h[layer].mlp.output= randomize_values_with_mask(llm.transformer.h[layer].mlp.output, mlp_mask)
                
            final_hs = llm.transformer.h[-1].output[0]
            final_hs = llm.lm_head(llm.transformer.ln_f(final_hs)).save()

        final_logits = final_hs.softmax(dim=-1)
        if debug:
            max_prob_idxs = []
            for i in range(len(prompts)):
                max_prob_idxs.append(final_logits[i][-1].argmax().save())
        else:
            max_prob_idxs = None

        correct_idx_prob = final_logits[:,-1,correct_idxs].save()
        incorrect_idx_prob = final_logits[:,-1,incorrect_idxs].save()

    return correct_idx_prob, incorrect_idx_prob, max_prob_idxs
        
        
class MatrixAutoEncoder(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.conv1 = nn.Conv1d(24, 16, 3, padding=1)  
        self.conv2 = nn.Conv1d(16, 4, 3, padding=1)
        self.conv3 = nn.Conv1d(4, 1, 3, padding=1)
        self.conv4 = nn.Conv1d(1, 1, 3, padding=1)
        self.pool = nn.MaxPool1d(2, 2)
        self.t_conv0 = nn.ConvTranspose1d(1, 1, 2, stride=2)
        self.t_conv1 = nn.ConvTranspose1d(1, 4, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose1d(4, 16, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose1d(16, 24, 2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.relu(self.t_conv0(x))
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = F.sigmoid(self.t_conv3(x))
        return x

def z_score_normalize(matrix):
    mean = matrix.mean()
    std_dev = matrix.std()
    normalized_matrix = (matrix - mean) / std_dev
    return normalized_matrix

def train_loop(llm):
    """
    model not updating?
    """
    torch.random.manual_seed(42)
    if wandb_on:
        wandb.init(project='crazy_interp_mask', name='v0')

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    batch_size = 5 # for OOM
    model = MatrixAutoEncoder()
    model.train()
    
    causal_mask = torch.randint(2, size=(llm.config.n_layer, llm.config.n_head)).float()
    # causal_mask = torch.zeros((llm.config.n_layer, llm.config.n_head)).float()
    dataset = datasets.load_dataset("franlucc/ts_bench_starcoder1b_funcfim_incorrect_uniq", split="train")
    dataset = dataset.filter(lambda x: len(x["prompt"]) < 8000) # for OOM
    prompts = [d["prompt"] for d in dataset]
    correct_idxs = [llm.tokenizer.encode(d["fim_sol"])[0] for d in dataset]
    incorrect_idxs = [llm.tokenizer.encode(d["generated_text"])[0] for d in dataset]
    idxs = list(zip(correct_idxs, incorrect_idxs))
    train_data = list(zip(prompts, idxs))

    
    # shuffle train data
    random.seed(42)
    random.shuffle(train_data)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=1)

    optimizer = torch.optim.AdamW(model.parameters(),  lr=0.001)
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Apply gradient clipping


    print("Training...")
    # number of epochs to train the model
    n_epochs = 20
    e = 10
    
    criterion = nn.MSELoss()
    # criterion = nn.KLDivLoss()
    for epoch in range(1, n_epochs+1):
        # monitor training loss
        train_loss = 0.0
        
        for data in train_loader:
            prompts, (correct_idxs, incorrect_idxs) = data
            if len(prompts) == 1:
                continue
            
            optimizer.zero_grad()

            output= model(causal_mask)
            # print(f"Output: {output}")
            
            correct_idx, incorrect_idx, _ = apply_mask(llm, output.clone(), prompts, correct_idxs, incorrect_idxs)
            correct_idx = util.apply(correct_idx, lambda x: x.value, Proxy)
            incorrect_idx = util.apply(incorrect_idx, lambda x: x.value, Proxy)
            correct_idx = correct_idx.clone().detach().to(model.device).requires_grad_(True)
            incorrect_idx = incorrect_idx.clone().detach().to(model.device).requires_grad_(True)
            # print(f"Correct idx: {correct_idx}, Incorrect idx: {incorrect_idx}")
            # create a clone of causal loss that is attatched to the graph of model and trainable
            # loss = causal_loss.detach().clone().to(model.device).requires_grad_(True)
            # loss = kl_divergence_loss(incorrect_idx, correct_idx).requires_grad_(True)
            
            with torch.enable_grad():
                # causal_loss = kl_divergence_loss(incorrect_idx,correct_idx)
                causal_loss = relative_distance_loss(incorrect_idx,correct_idx)
                # loss = criterion(output, causal_mask)/1000 + incorrect_idx.mean()
                loss = criterion(output, causal_mask) + causal_loss

                # print(f"diff: {criterion(output, causal_mask)}, {causal_loss*e}")
                loss.backward()
            
                # for name, param in model.named_parameters():
                #     print(f"{name} - Gradient: {param.grad}")

            optimizer.step()
            print(f"Batch loss: {loss.item()}")
            if wandb_on:
                wandb.log({'training_loss': loss.item(), "epoch":epoch, "batch_size":batch_size})
            train_loss += loss.item()
            
            
        # print avg training statistics 
        train_loss = train_loss/len(train_loader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch, 
            train_loss
            ))
        
        # for name, param in model.named_parameters():
        #     print(f"{name} - Gradient: {param.grad}")
            
        # save output to pt file
        with torch.no_grad():
            with open(f"_causal_mask_epoch_{epoch}.pt", "wb") as f:
                out = model(causal_mask)
                out = z_score_normalize(out)
                out = (out > 0.5).float()
                torch.save(out, f)
            # save model
            torch.save(model.state_dict(), f"_model_epoch_{epoch}.pt")
    if wandb_on:
        wandb.finish()

def mask_to_matrix(llm : LanguageModel, mask_json : dict):
    """
    mask_json: dict
        {
            0 : {"attn" : [1,2,3],
                 "mlp" : [4,5,6]}, 
        }
    """
    l = llm.config.n_layer
    d = llm.config.n_embd
    a = llm.config.n_head
    if d % a != 0:
        raise ValueError("MLP dim must be divisible by number of attention heads")
    mask = torch.zeros((l, a, (d//a)+1))
    for layer in mask_json:
        attn_mask_lst = mask_json[layer]["attn"]
        mlp_mask_lst = mask_json[layer]["mlp"]
        
        for idx in attn_mask_lst:
            mask[layer][idx][0] = 1
        
        for idx in mlp_mask_lst:
            row = idx // (d // a)
            col = idx + 1 - row*(d // a)
            mask[layer][row][col] = 1
    return mask


if __name__ == "__main__":
    device = sys.argv[1]
    if device == -1:
        device = "cpu"
    else:
        device = f"cuda:{device}"
    starcoderbase_1b = "/home/arjun/models/starcoderbase-1b/"
    llm = LanguageModel(starcoderbase_1b, device_map=device)
    
    # prompts = ["""
    # def my_print(s : <FILL>): 
    #     print("My name is ", s)
    # """,
    # """
    # def my_print(n : <FILL>): 
    #     print("My age is ", n)
    # """]*5
    # solns = ["str", "int"]*5
    # this_correct_idxs = [llm.tokenizer.encode(soln) for soln in solns]
    # this_incorrect_idxs = [llm.tokenizer.encode("int"), llm.tokenizer.encode("str")]*5
    
    # prompts = [placeholder_to_std_fmt(p, STARCODER_FIM) for p in prompts]

    # mask_mat = torch.zeros((llm.config.n_layer, llm.config.n_head))
    # # mask_mat = torch.randint(2, size=(llm.config.n_layer, llm.config.n_head, llm.config.n_head))
    # out = apply_mask(llm, mask_mat.clone().detach().requires_grad_(True), prompts, this_correct_idxs, this_incorrect_idxs, debug=True)
    # out = util.apply(out, lambda x: x.value, Proxy)
    # print(out[0], [llm.tokenizer.decode(o.item()) for o in out[1]])

    train_loop(llm)
