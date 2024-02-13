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

model takes input mask, the output mask is then passed to next train item
loss is calculated over batch logits (each same mask thankfully)
mask needs to be applied in a batch 

Modle inp-out are the mask
Loss is derived from a function that uses mask applied to traiuning items
"""

"""
TODO: problem. Model does not update the mask correctly

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


def kl_divergence_loss(prob_dist1, prob_dist2):
    # Calculate KL divergence between prob_dist1 and prob_dist2
    kl_divergence = F.kl_div(torch.log(prob_dist2), prob_dist1, reduction='batchmean')
    return kl_divergence

def relative_loss2(prob_dist1, prob_dist2, max_probs):
    prob_dist1 = prob_dist1.mean()
    prob_dist2 = prob_dist2.mean()
    max_probs = max_probs.mean()
    return (prob_dist1 / prob_dist2) + (max_probs / prob_dist2)

def relative_loss3(prob_dist1, prob_dist2, max_probs):
    prob_dist1 = prob_dist1.mean()
    prob_dist2 = prob_dist2.mean()
    max_probs = max_probs.mean()
    return (max_probs / prob_dist2) + (1 / prob_dist1)

def relative_loss(prob_dist1, prob_dist2):
    prob_dist1 = prob_dist1.mean()
    prob_dist2 = prob_dist2.mean()
    return (prob_dist1 / prob_dist2) 



def apply_mask(llm: LanguageModel,
               mask: torch.Tensor,
               prompts: List[str],
               correct_idxs: List[int],
               incorrect_idxs: List[int],
               debug=False) -> List[int]:
    
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
            max_prob_idxs = final_logits[:,-1,:].max(dim=-1).save()
            # max_prob_idxs = final_logits[:,-1,:].argmax().save()
            # print(final_logits.shape)
        else:
            max_prob_idxs = None

        correct_idx_prob = final_logits[:,-1,correct_idxs].save()
        incorrect_idx_prob = final_logits[:,-1,incorrect_idxs].save()

    return correct_idx_prob, incorrect_idx_prob, max_prob_idxs
        
        
class CNNEncoder(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        self.conv1 = nn.Conv1d(24, 16, 3, padding=1)  
        self.conv2 = nn.Conv1d(16, 4, 3, padding=1)
        # self.conv3 = nn.Conv1d(4, 1, 3, padding=1)
        # self.conv4 = nn.Conv1d(1, 1, 3, padding=1)
        self.pool = nn.MaxPool1d(2, 2)
        # self.t_conv0 = nn.ConvTranspose1d(1, 1, 2, stride=2)
        # self.t_conv1 = nn.ConvTranspose1d(1, 4, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose1d(4, 16, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose1d(16, 24, 2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        # x = self.pool(x)
        # x = F.relu(self.conv3(x))
        # x = self.pool(x)
        # x = F.relu(self.conv4(x))
        x = self.pool(x)
        # x = F.relu(self.t_conv0(x))
        # x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = F.sigmoid(self.t_conv3(x))
        return x

class FFNEncoder(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        self.w = nn.Linear(16,24)
        self.b = nn.Linear(24,16)
    
    def forward(self, x):
        x = F.relu(self.w(x))
        x = F.sigmoid(self.b(x))
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
    wandb_on = False
    seed = 42
    torch.random.manual_seed(seed)
    random.seed(seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    batch_size = 5# for OOM
    n_epochs = 10
    lr = 2e-5
    weight_decay = 0.01
    loss_fn = relative_loss

    if wandb_on:
        wandb.init(project='crazy_interp_mask2', name='ffn_v0')
    
    model = FFNEncoder()
    model.train()
    
    # causal_mask = torch.randint(2, size=(llm.config.n_layer, llm.config.n_head)).float()
    causal_mask = torch.zeros((llm.config.n_layer, llm.config.n_head)).float()
    # causal_mask = torch.load("masks/success_maybe_attn_only/causal_mask_epoch_1.pt")
    # print(causal_mask)
    dataset = datasets.load_dataset("franlucc/ts_bench_starcoder1b_funcfim_incorrect_uniq_v1", split="train")
    dataset = dataset.filter(lambda x: len(x["prompt"]) < 8000) # for OOM
    prompts = [d["prompt"] for d in dataset]
    correct_idxs = [llm.tokenizer.encode(d["fim_sol"])[0] for d in dataset]
    incorrect_idxs = [llm.tokenizer.encode(d["generated_text"])[0] for d in dataset]
    idxs = list(zip(correct_idxs, incorrect_idxs))
    train_data = list(zip(prompts, idxs))

    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=1, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(),  lr=lr, weight_decay=weight_decay)
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Apply gradient clipping

    print("Training...")
    # number of epochs to train the model
    
    criterion = nn.MSELoss()
    # criterion = nn.KLDivLoss()
    for epoch in range(1, n_epochs+1):
        # monitor training loss
        train_loss = 0.0
        
        for data in train_loader:
            prompts, (correct_idxs, incorrect_idxs) = data
            if len(prompts) < batch_size:
                continue
            
            optimizer.zero_grad()

            output= model(causal_mask)
            # print(f"Output: {output}")
            # set all idxs that incremented from output to causal_mask to 1
            output = (output < 0.5).float()
            
            correct_idx, incorrect_idx, max_probs = apply_mask(llm, output.clone(), 
                                                               prompts, 
                                                               correct_idxs, 
                                                               incorrect_idxs, 
                                                               debug=True)
            correct_idx = util.apply(correct_idx, lambda x: x.value, Proxy)
            incorrect_idx = util.apply(incorrect_idx, lambda x: x.value, Proxy)
            max_probs = util.apply(max_probs, lambda x: x.value, Proxy)
            correct_idx = correct_idx.clone().detach().to(model.device).requires_grad_(True)
            incorrect_idx = incorrect_idx.clone().detach().to(model.device).requires_grad_(True)
            max_probs = max_probs.values.clone().detach().to(model.device).requires_grad_(True)
            
            with torch.enable_grad():
                causal_loss = loss_fn(max_probs,correct_idx)
                crit = criterion(output, causal_mask) # hack
                loss = crit + causal_loss
                loss.backward()
                
            optimizer.step()
            # format all string to 5 digits
            bloss = "{:.5f}".format(loss.item())
            causal_loss = "{:.5f}".format(causal_loss.item())
            correct = "{:.5f}".format(correct_idx.mean().item())
            incorrect = "{:.5f}".format(incorrect_idx.mean().item())
            crit = "{:.5f}".format(crit.item())
            max_probs = "{:.5f}".format(max_probs.mean().item())
            
            print(f"Batch loss: {bloss}\t\tincorrect {incorrect}\tcorrect {correct}\tmax_probs {max_probs}. {correct > incorrect}. {correct > max_probs}")
            if wandb_on:
                wandb.log({'training_loss': loss.item(), "epoch":epoch, "batch_size":batch_size})
            train_loss += loss.item()

            
        # print avg training statistics 
        train_loss = train_loss/len(train_loader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch, 
            train_loss
            ))
            
        # save output to pt file
        with torch.no_grad():
            with open(f"_causal_mask_epoch_{epoch}.pt", "wb") as f:
                out = model(causal_mask)
                out = (out > output.mean().item()).float()
                torch.save(out, f)
            # save model
            torch.save(model.state_dict(), f"_model_epoch_{epoch}.pt")
    if wandb_on:
        wandb.finish()



if __name__ == "__main__":
    device = sys.argv[1]
    if device == -1:
        device = "cpu"
    else:
        device = f"cuda:{device}"
    starcoderbase_1b = "/home/arjun/models/starcoderbase-1b/"
    llm = LanguageModel(starcoderbase_1b, device_map=device)

    train_loop(llm)
