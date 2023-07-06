from collections import defaultdict
from typing import Any, Dict, Optional, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trace_utils import *
from baukit.nethook import get_module
from model_utils import layername
"""
Modified and simplified from baukit
TODO
- possibly replace with tuned lens
"""


class LogitLens:
    """
    Applies the LM head at the output of each hidden layer, then analyzes the
    resultant token probability distribution.

    Only works when hooking outputs of *one* individual generation.

    Inspiration: https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens

    Warning: when running multiple times (e.g. generation), will return
    outputs _only_ for the last processing step.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        # layer_module_tmp: str,
        # ln_f_module: str,
        # lm_head_module: str,
        activations,
        top_k: int = 5,
    ):
        self.model, self.tok = model, tok
        self.k = top_k
        
        self.lm_head, self.ln_f = (
            get_module(model, "lm_head"),
            get_module(model, "transformer.ln_f"),
        )

        self.output= None
        
        self.output = {}
        
        for (layer, np_tensor) in activations.items():
            tensor = torch.from_numpy(np_tensor).to(model.device)
            assert (
                tensor.size(0) == 1
            ), "Make sure you're only running LogitLens on single generations only."
            print(tensor.shape)
            self.output[layer] = torch.softmax(
                self.lm_head(self.ln_f(tensor[:, -1, :])), dim=1
            )
            print(self.output[layer].shape)

        
    
    def __call__(self):
        to_print = defaultdict(list)
        print(self.output.keys(), self.output)
        for layer, pred in self.output.items():
            print("LP",layer, pred)
            rets = torch.topk(pred[-1], self.k)

            for i in range(self.k):
                to_print[layer].append(
                    (
                        self.tok.decode(rets[1][i]),
                        round(rets[0][i].item() * 1e2) / 1e2,
                    )
                )

        return to_print

    # def pprint(self, k=5):
    #     to_print = defaultdict(list)

    #     for layer, pred in self.output.items():
    #         rets = torch.topk(pred[0], k)

    #         for i in range(k):
    #             to_print[layer].append(
    #                 (
    #                     self.tok.decode(rets[1][i]),
    #                     round(rets[0][i].item() * 1e2) / 1e2,
    #                 )
    #             )

    #     print(
    #         "\n".join(
    #             [
    #                 f"{layer}: {[(el[0], round(el[1] * 1e2)) for el in to_print[layer]]}"
    #                 for layer in range(self.n_layers)
    #             ]
    #         )
    #     )
