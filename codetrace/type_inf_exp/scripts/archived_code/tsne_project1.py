from multiprocessing import cpu_count
from codetrace.interp_utils import collect_hidden_states_at_tokens, insert_patch
from codetrace.utils import placeholder_to_std_fmt, STARCODER_FIM
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
import argparse
import datasets
import torch
from nnsight import LanguageModel
import numpy as np
from tqdm import tqdm
from einops import rearrange
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from collections import Counter

def batched_activations(model, prompts, patch, layers_to_patch, tokens_to_patch, batch_size, patch_mode="add"):
    prompt_batches = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]
    results = []
    for i,batch in tqdm(enumerate(prompt_batches), desc="Insert Patch Batch", total=len(prompt_batches)):
        res : TraceResult = insert_patch(model, batch, patch, layers_to_patch, tokens_to_patch, patch_mode)
        results.append(res)
           
    return results

def batched_hs(model, prompts, tokens_to_patch, batch_size, patch_mode="add"):
    prompt_batches = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]
    results = []
    for i,batch in tqdm(enumerate(prompt_batches), desc="Collect activations", total=len(prompt_batches)):
        res = collect_hidden_states_at_tokens(model, batch, tokens_to_patch)
        results.append(res.detach().cpu())
           
    return results

def main(args):
    data = datasets.load_from_disk(args.data_dir + "/ood_steering_results_ds")
    if args.max_size == -1:
        args.max_size = len(data)
    
    correct = data.filter(lambda x: x["correct_steer"])
    incorrect = data.filter(lambda x: not x["correct_steer"])
    print(correct, incorrect)
    
    steering_vector = torch.load(args.steering_vector)
    
    model = LanguageModel("bigcode/starcoderbase-1b", device_map="cuda")
    layers = [10,11,12,13,14]
    
    correct = correct.shuffle(seed=42).select(range(min(args.max_size, len(correct))))
    incorrect = incorrect.shuffle(seed=42).select(range(min(args.max_size, len(incorrect))))
    
    correct_prompts = [placeholder_to_std_fmt(p, STARCODER_FIM) for p in correct["fim_program"]]
    incorrect_prompts = [placeholder_to_std_fmt(p, STARCODER_FIM) for p in incorrect["fim_program"]]
    
    if os.path.exists(Path(args.save_activations+"_correct")):
        correct_activations = torch.load(args.save_activations+"_correct")
    if os.path.exists(Path(args.save_activations+"_incorrect")):
        incorrect_activations = torch.load(args.save_activations+"_incorrect")
    else:
        incorrect_res = batched_activations(model, incorrect_prompts, steering_vector, layers, "<fim_middle>", batch_size=args.batch_size)
        incorrect_activations = torch.stack([i._hidden_states[:,:,-1,:] for i in incorrect_res], dim=0)
        incorrect_activations = rearrange(incorrect_activations, "t l i d -> l t i d")
        del(incorrect_res)
        print(incorrect_activations.shape, steering_vector.shape)
        
        correct_res = batched_activations(model, correct_prompts, steering_vector, layers, "<fim_middle>",  batch_size=args.batch_size)
        correct_activations = torch.stack([i._hidden_states[:,:,-1,:] for i in correct_res], dim=0)
        correct_activations = rearrange(correct_activations, "t l i d -> l t i d")
        del(correct_res)
        print(correct_activations.shape, steering_vector.shape)
    
        # save
        torch.save(correct_activations, args.save_activations+"_correct")
        torch.save(incorrect_activations, args.save_activations+"_incorrect")
    
    if os.path.exists(Path(args.save_activations+"_correct_hs")):
        correct_hs = torch.load(args.save_activations+"_correct_hs")
    if os.path.exists(Path(args.save_activations+"_incorrect_hs")):
        incorrect_hs = torch.load(args.save_activations+"_incorrect_hs")
    else:
        incorrect_hs = batched_hs(model, incorrect_prompts,"<fim_middle>", batch_size=args.batch_size)
        incorrect_hs = torch.stack(incorrect_hs, dim=1)
        correct_hs = batched_hs(model, correct_prompts,"<fim_middle>",  batch_size=args.batch_size)
        correct_hs = torch.stack(correct_hs, dim=1)
        torch.save(correct_hs, args.save_activations+"_correct_hs")
        torch.save(incorrect_hs, args.save_activations+"_incorrect_hs")
        
    print(correct_hs.shape, incorrect_hs.shape)
        
    pca = PCA(n_components=2)
    scaler = StandardScaler()
    
    def scale(x):
        scaler.fit(x)
        return scaler.transform(x)
    
    l = 20
    embedded_in_correct = correct_activations[l,:,:,:].squeeze().detach().numpy()
    embedded_in_incorrect = incorrect_activations[l,:,:,:].squeeze().detach().numpy()
    embedded_in_incorrect2 = incorrect_hs[l,:,:,:].squeeze().detach().numpy()
    embedded_in_correct2 = correct_hs[l,:,:,:].squeeze().detach().numpy()
    
    print(embedded_in_correct.shape, embedded_in_incorrect.shape)
    e_in = np.concatenate((embedded_in_correct,embedded_in_correct2,embedded_in_incorrect, embedded_in_incorrect2))
    e_in = scale(e_in)
    
    embedded_out = pca.fit_transform(e_in)
    print(embedded_out.shape)
    a = len(embedded_in_correct)
    b = len(embedded_in_correct2)
    c = len(embedded_in_incorrect)
    d = len(embedded_in_incorrect2)
    print(a,b,c,d)
    plt.scatter(embedded_out[:a, 0], embedded_out[:a, 1], c="green", s=20)
    # plt.scatter(embedded_out[a:a+b, 0], embedded_out[a:a+b, 1], c="blue", s=20)
    plt.scatter(embedded_out[a+b:a+b+c, 0], embedded_out[a+b:a+b+c, 1], c="red", s=20)
    # plt.scatter(embedded_out[a+b+c:, 0], embedded_out[a+b+c:, 1], c="orange", s=20)
    plt.tight_layout()
    plt.savefig(args.out_plot)
    
    # e_in = np.concatenate((embedded_in_correct,embedded_in_incorrect))
    # e_in = scale(e_in)
    # embedded_out = pca.fit_transform(e_in)
    # print(embedded_out.shape)
    # a = len(embedded_in_correct)
    # plt.scatter(embedded_out[:a, 0], embedded_out[:a, 1], c="green")
    # plt.scatter(embedded_out[a:, 0], embedded_out[a:, 1], c="red")
    
    plt.tight_layout()
    plt.savefig(args.out_plot)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--steering_vector", required=True)
    parser.add_argument("--max_size", required=True, type=int)
    parser.add_argument("--out-plot", required=True)
    parser.add_argument("--save_activations", required=True)
    parser.add_argument("--batch_size", default=1)
    args = parser.parse_args()
    main(args)
