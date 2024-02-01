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

TODO: change both to 2048
n_dim is 2048, h_dim is 128, a is 16 -> 16*128 = 2048

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


def apply_mask(llm: LanguageModel,
               mask: torch.Tensor,
               prompt: str,
               correct_idx: int,
               incorrect_idx:int) -> int:

    with llm.generate(max_new_tokens=1, 
                    return_dict_in_generate=True,
                    output_attentions=True,) as generator:
        with generator.invoke(prompt) as invoker:
            # apply mask
            for layer in range(llm.config.n_layer):
                
                target_shape = llm.transformer.h[layer].attn.c_proj.output.shape
                print(target_shape)
                # get a tensor of 0s
                attn_mask = torch.zeros(target_shape, device=device)
                # apply mask to any attn heads
                llm.transformer.h[layer].attn.c_proj.output= llm.transformer.h[layer].attn.c_proj.output.mul(attn_mask)
                #^ this works
                
                # target_shape = llm.transformer.h[layer].attn.c_proj.output.shape
                # # apply mask to any attn heads
                # attn_mask = mask[layer][:,0]
                # attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
                # print(target_shape, attn_mask.shape)
                # # repeat for head_dim
                # attn_mask = attn_mask.repeat(1,target_shape[2],1)
                # print(attn_mask.shape)
                # attn_mask = einops.rearrange(attn_mask, 'b h t -> b t h')
                # print(attn_mask.shape)
                
                # # expand per prompts n
                # attn_mask = attn_mask.expand(target_shape[0], attn_mask.shape[1], attn_mask.shape[2])
                # print(attn_mask.shape)
                # # expand per tokens t
                # attn_mask = attn_mask.expand(attn_mask.shape[0], target_shape[1], attn_mask.shape[2])
                # print(attn_mask.shape)
                
                # print(attn_mask.shape)
                # print(llm.transformer.h[layer].attn.c_attn.output.shape)
                # # print(llm.transformer.h[layer].attn.output[1].shape)
                # # print(llm.transformer.h[layer].attn.output[-1].shape)
                # llm.transformer.h[layer].attn.c_attn.output= llm.transformer.h[layer].attn.c_attn.output.mul(attn_mask)
                
                # # apply mask to any mlp dim
                # mlp_mask = mask[layer][1]
                # llm.transformer.h[layer].mlp.output = llm.transformer.h[layer].mlp.output.mul(mlp_mask)
            
            final_hs = llm.transformer.h[-1].output[0]
            final_hs = llm.lm_head(llm.transformer.ln_f(final_hs)).save()
    
        final_logits = final_hs.softmax(dim=-1)
        max_prob_idx = final_logits[:,-1].argmax().item().save()
        correct_idx_prob = final_logits[:,-1,correct_idx].item()
        incorrect_idx_prob = final_logits[:,-1,incorrect_idx].item()
        
        diff = correct_idx_prob - incorrect_idx_prob
        diff = diff.save()
        
    return diff, max_prob_idx
        
        
class MatrixAutoEncoder(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        ## encoder layers ##
        # conv layer (depth from 3 --> 16), 3x3 kernels
        self.conv1 = nn.Conv2d(24, 16, 3, padding=1)  
        # conv layer (depth from 16 --> 4), 3x3 kernels
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)
        
        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 24, 2, stride=2)

    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # compressed representation
        
        ## decode ##
        # add transpose conv layers, with relu activation function
        x = F.relu(self.t_conv1(x))
        # output layer (with sigmoid for scaling from 0 to 1)
        x = F.sigmoid(self.t_conv2(x))
        # print(x.shape)
        return x

def causal_loss(correct_tok_prob, incorrect_tok_prob):
    return incorreect_tok_prob - correct_tok_prob


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
    device = int(sys.argv[1])
    if device == -1:
        device = "cpu"
    else:
        device = f"cuda:{device}"
    starcoderbase_1b = "/home/arjun/models/starcoderbase-1b/"
    llm = LanguageModel(starcoderbase_1b, device_map=device)
    
    prompt = """
    def my_print(s : <FILL>): 
        print("My name is ", s)
    """
    prompt = placeholder_to_std_fmt(prompt, STARCODER_FIM)
    correct = llm.tokenizer.encode("str")
    incorrect = llm.tokenizer.encode("int")
    
    mask = {
        0 : {"attn" : [1,2,3],
             "mlp" : [4,5,130*3]}, 
    }
    mask_mat = mask_to_matrix(llm, mask)
    # print(mask_mat[0][3])
    
    out = apply_mask(llm, torch.tensor(mask_mat), prompt, correct, incorrect)
    out = util.apply(out, lambda x: x.value, Proxy)
    print(out[0], llm.tokenizer.decode(out[1]))

# model = MatrixAutoEncoder()

# import random
# import numpy as np

# values = np.array([random.randint(0,1)/1 for i in range(24*16*2048)])
# inpt = torch.from_numpy(values).view(24,16,2048).to(device=model.device, dtype=model.conv1.weight.dtype)
# print(inpt.shape, inpt[0][0], inpt.dtype)
# model(inpt)

# train_data = []
# for ex in tqdm(range(100)):
#     random.seed(ex)
#     random.shuffle(values)
#     inpt = torch.from_numpy(values).view(24,16,2048)
#     train_data.append(inpt)

# train_loader = torch.utils.data.DataLoader(train_data, batch_size=10, num_workers=4)

# # specify loss function
# criterion = nn.BCELoss()
# ceriterion=my_loss
# # specify loss function
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# print("Training...")
# # number of epochs to train the model
# n_epochs = 10

# for epoch in range(1, n_epochs+1):
#     # monitor training loss
#     train_loss = 0.0
    
#     ###################
#     # train the model #
#     ###################
#     for data in train_loader:
#         # _ stands in for labels, here
#         # no need to flatten images
#         prompt, correct_incorrect_idx_tup = data.to(device=model.device, dtype=model.conv1.weight.dtype)
#         # clear the gradients of all optimized variables
#         optimizer.zero_grad()
#         # forward pass: compute predicted outputs by passing inputs to the model
#         outputs = model(images)
#         # calculate the loss
#         loss = criterion(outputs, correct_incorrect_tup, model.device)
#         # backward pass: compute gradient of the loss with respect to model parameters
#         loss.backward()
#         # perform a single optimization step (parameter update)
#         optimizer.step()
#         # update running training loss
#         train_loss += loss.item()*images.size(0)
            
#     # print avg training statistics 
#     train_loss = train_loss/len(train_loader)
#     print('Epoch: {} \tTraining Loss: {:.6f}'.format(
#         epoch, 
#         train_loss
#         ))