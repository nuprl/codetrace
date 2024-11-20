from typing import List,Union,Dict,Any, AsyncGenerator, Generator, Optional
from vllm import AsyncLLMEngine, SamplingParams, AsyncEngineArgs, LLM
from vllm.outputs import RequestOutput
import os 

def load_vllm(
    model:str,
    dtype:str,
    tps:int,
    tokenizer: Optional[str] = None,
    async_inference: bool = False,
    **kwargs
)-> Union[AsyncLLMEngine, LLM]:
    tokenizer=model if not tokenizer else tokenizer
    if async_inference:
        engine_args = AsyncEngineArgs(
            model=model, 
            tensor_parallel_size=tps,
            dtype=dtype,
            tokenizer=tokenizer
        )
        llm = AsyncLLMEngine.from_engine_args(engine_args)
        llm.log_requests = False
    else:
        llm = LLM(model,dtype=dtype, tensor_parallel_size=tps,tokenizer=tokenizer)
    return llm

def request_vllm_generations(
    llm: Union[LLM,AsyncLLMEngine],
    prompts: List[str],
    sampling_params: SamplingParams,
    **kwargs
) -> Union[List[RequestOutput],AsyncGenerator[None,RequestOutput]]:
    return llm.generate(prompts, sampling_params, **kwargs)
    
def request_vllm_chat(
    llm: LLM,
    prompts: List[List[Dict[str,str]]],
    sampling_params: SamplingParams,
    **kwargs
) -> List[RequestOutput]:  
    # chat template
    chat_template = kwargs.pop("chat_template",None)
    if not chat_template:
        chat_template = llm.get_tokenizer().chat_template
    if not chat_template:
        raise ValueError("Model does not have chat template! Make sure you have the instruct version of the model.")
    
    return llm.chat(prompts, sampling_params, chat_template=chat_template,
                    continue_final_message=True, add_generation_prompt=False,**kwargs)

def request_vllm_completions(
    llm: Union[LLM, AsyncLLMEngine],
    prompts: Union[List[str],List[List[Dict[str,str]]]],
    sampling_params: SamplingParams,
    **kwargs
)-> Union[List[RequestOutput], AsyncGenerator[None,RequestOutput]]:
    if isinstance(prompts[0], str):
        generated = request_vllm_generations(llm, prompts, sampling_params, **kwargs)
    elif isinstance(llm,LLM):
        generated = request_vllm_chat(llm, prompts, sampling_params, **kwargs)
    else:
        raise NotImplementedError("Chat not implemented for AsyncLLMEngine")
    return generated

def get_vllm_config(llm: Union[LLM,AsyncLLMEngine]) -> Dict[str,Any]:
    if hasattr(llm, "llm_engine"):
        return llm.llm_engine.get_model_config().hf_config
    else:
        return llm.get_model_config().hf_config