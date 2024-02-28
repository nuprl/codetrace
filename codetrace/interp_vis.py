"""
Interp visualizations
"""
import sys
import os
from codetrace.utils import *
import glob
import datasets
import pandas as pd
import torch
from tqdm import tqdm
from nnsight import LanguageModel,util
from nnsight.tracing.Proxy import Proxy
from codetrace.interp_utils import *
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import matplotlib as mpl
import os
    
def patched_heatmap_prediction(model : LanguageModel,
                        clean_prompts : List[str] | str, 
                        corrupted_prompts : List[str] | str,
                        patched_final_logits : List[TraceResult],
                        layers_patched : List[int],
                        clean_token_idx : int = -1,
                        corrupted_token_idx : int = -1, 
                        annotations : List[Tuple[str,str]] | None = None,
                        figsize : Tuple[int,int] = (10,10),
                        bbox_to_anchor : Tuple[int,int] = (1.6, 1.0),
                        figtitle : str = "Patch Logit-Prediction Heatmap",
                        outfile : str = "temp.png"):
    """
    Cells on heatmap should be the final layer top logit that changes
    after patch.
    
    All prompt pairs are patched in the same layers. Patched logits is
    list of trace results of the final layer, one for each prompt pair.
    
    NOTE: 
    - assumes top_k in logits is 1
    - patched_final_logits is 2D when there are multiple prompt pairs, else 1D
        - 1D is one TraceResult for each layer
        - 2D is a list of TraceResults for each layer.
            ex: patched_final_logits[0] is results for each prompt at layer layers_patched[0]
    """
    clean_prompts = arg_to_list(clean_prompts)
    corrupted_prompts = arg_to_list(corrupted_prompts)
    if not isinstance(patched_final_logits, list):
        raise ValueError("Given TraceResult, expected List.\nDid you forget to do an individual patch for each layer?")
    
    plt.figure(figsize=figsize)
    x_len = len(layers_patched)
    x_axis_labels = [f"Layer {i}" for i in layers_patched]
    y_len = len(clean_prompts)
    y_axis_labels = [f"Prompt {i}" for i in range(y_len)]
    
    plt.yticks(list(range(len(y_axis_labels))), y_axis_labels)
    plt.xticks(list(range(len(x_axis_labels))), x_axis_labels)
    plt.xlabel("Patched Layer")
    plt.title(figtitle)

    # need token and probabilities
    tokens, probs = [], []
    for layer_patch in patched_final_logits:
        patched_logits : LogitResult = layer_patch.decode_logits(prompt_idx=list(range(y_len)))
        pt, pp = [], []
        for i in range(y_len):
            pt.append(np.array(patched_logits[0][i].tokens(model.tokenizer)).item())
            pp.append(patched_logits[0][i].probs().item())
        tokens.append(pt)
        probs.append(pp)
        
    tokens = np.array(tokens)
    probs = np.array(probs)
    
    cmap = mpl.cm.cool
    im = plt.imshow(probs.transpose(), cmap=cmap, aspect="auto")
    
    # color is the probability of patched token
    im.cmap(probs)
    # add colorbar
    plt.colorbar(im)

    # legend is clean prediction -> corrupted prediction
    patches = []
    if not annotations:
        def original_prompt_pred(model, prompts):
            logits = logit_lens(model, prompts).decode_logits(prompt_idx=list(range(y_len)))
            tokens, probs = [], []
            for i in range(len(prompts)):
                t = logits[0][i].tokens(model.tokenizer)
                p = logits[0][i].probs()
                tokens.append(t)
                probs.append(p)
            return np.array(tokens), probs
        
        original_clean = original_prompt_pred(model, clean_prompts)
        original_corrupted = original_prompt_pred(model, corrupted_prompts)

        # for each probability square, legend shows relative clean_tok->corr_tok
        patches = []
        for i in range(y_len):
            a_pred = original_clean[0][i].item()
            a_prob = float(f"{original_clean[1][i].item():.2f}")
            b_pred = original_corrupted[0][i].item()
            b_prob = float(f"{original_corrupted[1][i].item():.2f}")
            a = (a_pred, a_prob)
            b = (b_pred, b_prob)
            patches.append(mpatches.Patch(color='grey', label=f"{i}: {a} (->) {b}"))
    else:
        # NOTE: probs not supported for annotations
        for i,(a,b) in enumerate(annotations):
            patches.append(mpatches.Patch(color='grey', label=f"{i}: {repr(a)} (->) {repr(b)}"))
        
    plt.legend(handles=patches, loc="center right", bbox_to_anchor=bbox_to_anchor, title="original predictions\nprompt: clean (->) corrupt")

    # build an annotations dict for each square in grid with values from probs_patched_results
    annotations = {}
    for i in range(x_len):
        for j in range(y_len):
            p = probs[i][j].item()
            annotations[(i,j)] = {"toptok": f"{tokens[i][j].item()}","prob": f"{p:.2f}"}
    
    # add annotations
    for pos, text in annotations.items():
        plt.annotate(text["toptok"], xy=pos,color="black", fontsize=10, ha="center", va="center")

    plt.tight_layout()
    plt.savefig(outfile)
    
    
"""
Tuned lens visualizations

- for each token, show at all layers what tuned lens (or any decoder) predicts
"""

prefix = '''<!DOCTYPE html>
<html>
<head>
    <title>Hover Text</title>
    <style>
        .barcode {
            border-bottom: 1px dashed blue;
            cursor: pointer;
        }

        .hover-text {
            position: absolute;
            background-color: #ffffff;
            border: 1px solid black;
            padding: 5px;
            display: none;
        }

        .barcode:hover .hover-text {
            display: block;
        }
    </style>
</head>
<body>

<p id="text">'''

def make_suffix(hover_data: str):
    """
    hover data is a json string
    """
    suffix = """</p>

<script>
    var spans = document.querySelectorAll('.barcode');
    
    var barcodeinfo = """+ hover_data+""";

    spans.forEach(function(span, index) {
        var backgroundColor = span.style.backgroundColor;
        var text = span.textContent.trim();
        var hoverText = document.createElement('span');
        hoverText.className = 'hover-text';
        hoverText.textContent = backgroundColor + ' ' + text;

        var jsonDataDiv = document.createElement('div');
        jsonDataDiv.textContent = JSON.stringify(barcodeinfo[index],null,4);
        hoverText.appendChild(jsonDataDiv);

        span.appendChild(hoverText);
    });
</script>

</body>
</html>
"""
    return suffix

# from https://gist.github.com/ihsgnef/f13c35cd46624c8f458a4d23589ac768
def colorize(words, color_array):
    # words is a list of words
    # color_array is an array of numbers between 0 and 1 of length equal to words
    # returns a HTML string
    cmap = mpl.colormaps['cool']
    
    template = '<span class="barcode"; style="color: black; background-color: {}">{}</span>\n'
    colored_string = ''
    for word, color in zip(words, color_array):
        color = color * 100
        color = mpl.colors.rgb2hex(cmap(color)[:3])
        word = word.replace(" ", "&nbsp;").replace("\n", "<br>").replace("\t", "&nbsp;&nbsp;&nbsp;&nbsp;")
        word = word.replace("<fim_suffix>", "*fim_suffix*").replace("<fim_prefix>", "*fim_prefix*").replace("<fim_middle>", "*fim_middle*")
        colored_string += template.format(color, word)
    return colored_string + "<br>"*10

def tuned_lens_viz(prompt: str,
                   model: LanguageModel,
                   decoder,
                   layer: int,
                   activations: List[torch.Tensor] = None,
                   top_k: int = 1,) -> str:
    """
    Given a prompt and a custom decoder, first get
    activations of model on prompt at layer, then
    decode using the given decoder.
    Display the top_k predictions for each token in html.
    """
    tokenizer = model.tokenizer
    if activations is None:
        topk,activations,logits = custom_lens(model, decoder, prompt, layer, k=top_k)
    else:
        topk,activations,logits = custom_lens(model, decoder, prompt, layer, k=top_k, activations=activations)
    
    # tokenize but render special tokens as is
    tokens = tokenizer.tokenize(prompt, add_special_tokens=True)
    tokens = [tokenizer.convert_tokens_to_string([t]) for t in tokens]

    # tokens = tokenizer.tokenize(prompt)
    indices = topk.indices # [nlayer, nprompt, ntoken, topk]
    values = topk.values # [nlayer, nprompt, ntoken, topk]
    values = values.squeeze(0).squeeze(0).detach() # nlayer and nprompt are 1
    indices = indices.squeeze(0).squeeze(0).detach() # nlayer and nprompt are 1
    
    # make a list where for each token idx, we have the tokenized idx and the value

    hover_data = []
    color_array = [] # this will be max value
    for i in range(values.shape[0]):
        topk_preds = [tokenizer.decode(i) for i in indices[i].tolist()]
        vals = values[i].tolist()
        # round to 6 decimal places
        vals = [round(i, 6) for i in vals]
        color_array.append(max(vals))
        pred_values = list(zip(topk_preds, vals))
        # sort by value
        pred_values = sorted(pred_values, key=lambda x: x[1], reverse=True)
        pred_value_dict = {pred: val for pred,val in pred_values}
        hover_data.append({**pred_value_dict, "token": tokens[i]})
    
    # colorize the tokens
    colored_string = colorize(tokens, color_array)
    hover_data = json.dumps(hover_data, indent=1)
    print(hover_data)
    return prefix + colored_string + make_suffix(hover_data)


def wrap_htmls(htmls : List[str], outdir = str, labels : List[int] = None) -> str:
    """
    Wrap html files into one file with tab for navigation
    """
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    rendering = {}
    for file in glob.glob(f"{parent_dir}/rendering/*"):
        with open(file, "r") as f:
            basename = os.path.basename(file)
            rendering[basename] = f.read()
    print(rendering.keys())
    template = rendering["template.html"]
    tablinks = ['<div class="tabs">']
    tabcontent = []
    for i,s in enumerate(htmls):
        # write to tmp file
        with open(f"{outdir}/layer{i}.html", "w") as f:
            f.write(s)

        if labels is not None:
            label = labels[i]
        else:
            label = i
        link = f'<button class="tablinks" onclick="openTab(event, \'tab{i}\')">Layer {label}</button>\n'
        content = f'<div id="tab{i}" class="tabcontent">\n<iframe src="layer{i}.html" width="100%" height="500px"></iframe>\n</div>\n'
        tablinks.append(link)
        tabcontent.append(content)
        
    tablinks.append("</div>")
    template = template.replace("<!--tablinks-->", "\n".join(tablinks) + "\n"+ "\n".join(tabcontent))
    
    os.makedirs(outdir, exist_ok=True)
    with open(f"{outdir}/template.html", "w") as f:
        f.write(template)
    with open(f"{outdir}/styles.css", "w") as f:
        f.write(rendering["styles.css"])
    with open(f"{outdir}/scripts.js", "w") as f:
        f.write(rendering["scripts.js"])
    return template