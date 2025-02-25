# SantaCoder FIM Task

As a sanity-check, we run the FIM task from the SantaCoder paper, which has
been used to benchmark several Code LLMs. If we cannot reproduce other
models' results on this task, we are likely using them incorrectly.

See the following repository for the original implementation:

https://github.com/arjunguha/santacoder_fim_benchmark

## Usage

There are three steps:

1. Prepare a dataset of prompts with the right FIM tokens for your model
   using the `prepare_prompts.py` script.

2. Generate completions for your model using the `generate.sh` script.

3. Evaluate the results with `evaluate.py`.

## Usage on Boa

For StarCoderBase:

```bash
python3 prepare_prompts.py --fim-style starcoder1
./generate.sh /home/arjun/models/starcoderbase/ "jsonl:starcoder1_fim_task.jsonl" starcoderbase_15b
python3 evaluate.py starcoderbase_15b
```

For Code Llama 7B Base:

```bash
python3 prepare_prompts.py --fim-style codellama
./generate.sh /home/arjun/models/CodeLlama-7b-hf/ "jsonl:codellama_fim_task.jsonl" codellama_7b_base
python3 evaluate.py codellama_7b_base
```

For Code Llama 7B Instruct:

```bash
python3 prepare_prompts.py --fim-style codellama
./generate.sh /mnt/ssd/arjun/models/codellama_7b_instruct "jsonl:codellama_fim_task.jsonl" codellama_7b_instruct
python3 evaluate.py codellama_7b_instruct
```

For Qwen2.5 Coder 7B:

```bash
python3 prepare_prompts.py --fim-style qwencoder
./generate.sh /mnt/ssd/arjun/models/qwen2p5_coder_7b_base/ "jsonl:qwencoder_fim_task.jsonl" qwencoder_7b_base
python3 evaluate.py qwencoder_7b_base
```
### Results

It takes less than 5 mins to run each model on Boa.

```
                                accuracy
model                 language          
codellama_7b_base     java         0.243
                      js           0.365
                      py           0.117
codellama_7b_instruct java         0.137
                      js           0.158
                      py           0.058
qwencoder_7b_base     java         0.803
                      js           0.731
                      py           0.698
starcoderbase_15b     java         0.745
                      js           0.750
                      py           0.629
```

I have to assume we are doing something wrong with Code Llama. But, I have
reproduced the results for StarCoderBase. Qwen2.5 Coder 7B does well too.