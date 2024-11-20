from codetrace.steering import (
    subtract_avg,
    get_model_fim,
    prepare_prompt_pairs,
    _steer_test_split
)
import datasets
import torch

def test_subtract_avg():
    x = torch.Tensor(
        [
            # layer0
            [
                # prompt 0
                [[1,2],[3,4],[5,6]],
                # prompt 1
                [[1,2],[3,4],[5,6]],
            ],

            # layer1
            [
                # prompt 0
                [[1,2],[3,4],[5,6]],
                # prompt 1
                [[-1,-2],[-3,-4],[-5,-6]],
            ],
            # layer2
            [
                # prompt 0
                [[1,2],[3,4],[5,6]],
                # prompt 1
                [[1,-2],[3,-4],[-5,6]],
            ],
        ]
    )
    output = subtract_avg(x)
    expected = torch.Tensor(
        [
            # layer0
            [
                [[0,0],[0,0],[0,0]],
            ],

            # layer1
            [
                # prompt 0
                [[2,4],[6,8],[10,12]],
            ],
            # layer2
            [
                [[0,4],[0,8],[10,0]],
            ],
        ]).to(dtype=torch.float)
    print(x.shape, expected.shape, output.shape)
    assert torch.equal(output, expected), f"{output} != {expected}"

def test_prepare_prompts():
    prompts = [
        {"fim_program":"my name is <FILL> !","mutated_program":"my name is NOT <FILL> !"},
        {"fim_program":"my job is <FILL> !","mutated_program":"my job is NOT <FILL> !"},
        {"fim_program":"my house is <FILL> !","mutated_program":"my house is NOT <FILL> !"},
        {"fim_program":"my car is <FILL> !","mutated_program":"my car is NOT <FILL> !"},
    ]
    fim_obj = get_model_fim("starcoder")
    output = prepare_prompt_pairs(prompts, (lambda x: fim_obj.placeholder_to_fim(x)))
    expected = []
    for i in prompts:
        expected.append(fim_obj.placeholder_to_fim(i["fim_program"]))
        expected.append(fim_obj.placeholder_to_fim(i["mutated_program"]))
    assert output == expected, f"{output}!={expected}"

def test_stratify():
    data = [{"hexsha":0},
            {"hexsha":1},
            {"hexsha":2},
            {"hexsha":3},
            {"hexsha":4},
            {"hexsha":5},
            {"hexsha":5},
            {"hexsha":5},
            {"hexsha":6},
            {"hexsha":6},
            {"hexsha":7}]
    ds = datasets.Dataset.from_list(data)
    steer_split,test_split = _steer_test_split(
        ds,
        4,
        True,
        None,
        "hexsha"
    )
    steer_split = [x["hexsha"] for x in steer_split]
    test_split = [x["hexsha"] for x in test_split]
    print(steer_split, test_split)
    assert set(steer_split).intersection(set(test_split)) == set(), f"{steer_split} - {test_split}"

def test_prepare_chat_prompt():
    from transformers import AutoTokenizer
    fim_obj = get_model_fim("codellama_instruct")

    program = f"""
def is_palindrome(s: {fim_obj.placeholder}):
    return s[::-1]==s
""".strip()
    
    tokenizer = AutoTokenizer.from_pretrained("/mnt/ssd/arjun/models/codellama_7b_instruct")
    output = tokenizer.apply_chat_template(fim_obj.placeholder_to_fim(program), tokenize=False, 
                                           add_generation_prompt=False, continue_final_message=True)
    expected = '''<s>[INST] Continue this program with the correct substitution for <FILL>:

def is_palindrome(s: <FILL>):
    return s[::-1]==s [/INST] def is_palindrome(s:'''
    print(expected)
    print(output)
    assert output == expected, f"{output}!={expected}"