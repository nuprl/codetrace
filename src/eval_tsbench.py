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
from utils import *
    
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
    tree = TS_PARSER.parse(bytes( prog, "utf8"))
    
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


if __name__ == "__main__":
    # test_tree_sitter()
    ds = datasets.load_from_disk("ts-benchmark-dataset")
    fim_examples = fim_dataset(ds)
    
    ## model
    starcoder = "/home/arjun/models/starcoderbase-7b"
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
        completions_dir = "completions/starcoderbase-7b/multitok"
        os.makedirs(completions_dir, exist_ok=True)
        params = SamplingParams(temperature=0, max_tokens=50)
        for k,ex in enumerate(fim_examples):
            prompts = [placeholder_to_std_fmt(prompt_fim_var, starcoder_fim) for prompt_fim_var in ex]
            out = llm.generate(prompts, params)
            for i,output in enumerate(out):
                prompt = prompts[i]
                generated_text = output.outputs[0].text # n = 1, one greedy completion
                os.makedirs(f"{completions_dir}/prog_{k}", exist_ok=True)
                with open(f"{completions_dir}/prog_{k}/var_{i}.ts","w") as f:
                  f.write(unfim(prompt+generated_text,starcoder_fim))

    full_generate()
    test_single()
