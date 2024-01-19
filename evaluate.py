"""
Automatically evaluate LLM on typescript dataset "stenotype-eval-dataset".

Each eval item is one type annotation FIM'd out, for every type annotation in each prog.

Do 1 token completion evals first, then run TS to parse.
Do multi-token completion eval second, then run TS to parse.

Greedy generation only.
"""
import datasets
from vllm import LLM, SamplingParams
import os
import json
from tree_sitter import Language, Parser
from typing import Tuple
from tqdm import tqdm
from build_dataset import *
import glob
import tempfile
import random

vendor = "/home/franlucc/llm_libs"
Language.build_library(
    "build/my-languages.so",
    [f"{vendor}/tree-sitter-typescript/typescript"],
)
TS_LANGUAGE = Language("build/my-languages.so", "typescript")
parser = Parser()
parser.set_language(TS_LANGUAGE)
fim_placeholder = "<FILL>"

def replace_between_points(original_string : str, start_point : Tuple[int], end_point : Tuple[int], replacement):
    '''
    Replace a string A with a string B at interval (start_point, end_point) where each point 
    is a tuple of (column,row) of a char in the multiline string A [tree-sitter convention]
    '''
    with tempfile.NamedTemporaryFile(mode='w+t', delete=False) as temp:
        temp.write(original_string)
        temp.seek(0)
        original_string = temp.read()
    replacement = ": "+replacement
    start_index = len("\n".join(original_string.splitlines()[:start_point[0]])) + start_point[1]+1
    end_index = len("\n".join(original_string.splitlines()[:end_point[0]])) + end_point[1]+1
    modified_string = (
        original_string[:start_index] + replacement + original_string[end_index:]
    )
    return modified_string
    
    
def test_tree_sitter():
    prog = """
interface Person {
  name: string;
  next: Line;
}

type Line = "end of line" | Person;

// Examples of Line
const LINE_EX_1: Line = "end of line";
const LINE_EX_2: Line = { name: "Alice", next: "end of line" };
const LINE_EX_3: Line = { name: "Alice", next: { name: "Bob", next: "end of line" } };

// Counts the number of people in the line.
function countPeople(ln: Line): number {
  if (typeof ln === "string" && ln === "end of line") {
    return 0;
  } else {
    return 1 + countPeople(ln.next);
  }
}"""
    tree = parser.parse(bytes( prog, "utf8"))
    
    query = TS_LANGUAGE.query(
        """
((type_annotation) @name)
    """
    )
    
    captures = query.captures(tree.root_node)
    c = captures[0]
    print(c)
    s = replace_between_points(prog, c[0].start_point, c[0].end_point, fim_placeholder)
    with open("out.ts","w") as f:
        f.write(s)


def fim_example(ex):
    prog = ex["content"]
    tree = parser.parse(bytes( prog, "utf8"))
    
    query = TS_LANGUAGE.query(
        """
((type_annotation) @name)
    """
    )
    fim_variations = []
    captures = query.captures(tree.root_node)
    for c in captures:
        s = replace_between_points(prog, c[0].start_point, c[0].end_point, fim_placeholder)
        fim_variations.append(s)
    return fim_variations

def fim_dataset(hf_dataset):
    fim_examples = []
    for ex in tqdm(hf_dataset):
        fim_progs = fim_example(ex)
        fim_examples.append(fim_progs)
    return fim_examples


if __name__ == "__main__":
    # test_tree_sitter()
    # ds = datasets.load_dataset("franlucc/stenotype-eval-dataset", split="train")
    # fim_examples = fim_dataset(ds)
    # with open("fim_eval_prompts.json", "w") as f:
    #     json.dump(fim_examples, f, indent=4)
    
    with open("fim_eval_prompts.json", "r") as f:
        fim_examples = json.load(f)
    
    ## model
    root = "/home"
    starcoder = f"{root}/arjun/models/starcoderbase-7b"
    starcoder_fim = FimObj("<fim_prefix>", "<fim_suffix>","<fim_middle>", fim_placeholder)
    llm = LLM(model=starcoder)
    
    def test_single():
        params = SamplingParams(temperature=0, max_tokens=1)
        random_ex = random.randint(0, len(fim_examples))
        prompt = placeholder_to_std_fmt(fim_examples[random_ex][0], starcoder_fim)
        out = llm.generate([prompt], params)
        generated_text = out[0].outputs[0].text
        with open("out.ts","w") as f:
            f.write(unfim(prompt+generated_text, starcoder_fim))

    def full_generate():
        completions_dir = "completions/starcoder/singletok"
        os.makedirs(completions_dir, exist_ok=True)
        params = SamplingParams(temperature=0, max_tokens=1)
        for k,ex in enumerate(fim_examples):
            prompts = [placeholder_to_std_fmt(prompt_fim_var, starcoder_fim) for prompt_fim_var in ex]
            out = llm.generate(prompts, params)
            for i,output in enumerate(out):
                prompt = prompts[i]
                generated_text = output.outputs[0].text # n = 1, one greedy completion
                os.makedirs(f"{completions_dir}/prog_{k}", exist_ok=True)
                with open(f"{completions_dir}/prog_{k}/var_{i}.ts","w") as f:
                  f.write(unfim(prompt+generated_text,starcoder_fim))

    # full_generate()

    def run_tsc(ts_dir):
        """
        todo: run tsc on all ts files in ts_dir and capture output - parse or no parse
        """
        for prog in glob.glob(f"{ts_dir}/*.ts"):
            cont = open(prog, "r").read()
            pass