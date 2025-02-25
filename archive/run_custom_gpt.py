from transformers import GPTBigCodeForCausalLM, AutoTokenizer, GPTBigCodePreTrainedModel
from codetrace.training.gpt_bigcode_modeling.gpt_bigcode_modeling_custom_mod import GPTBigCodeModel as Custom_GPTBigCodeModel

starcoderbase_1b = "/home/arjun/models/starcoderbase-1b"

tokenizer = AutoTokenizer.from_pretrained(starcoderbase_1b)
model = GPTBigCodeForCausalLM.from_pretrained(starcoderbase_1b)
model.transformer = Custom_GPTBigCodeModel(model.config)
assert hasattr(model.transformer, 'CUSTOM_LINEAR')
model = model.to('cuda')

print(model, model.device)

# test generation
prompt = "print('hello_"

def prepare_inp(prompt, tokenizer, device="cuda"):
    input_ids = tokenizer(prompt, return_tensors="pt")
    attn_mask = input_ids.attention_mask.to(device)
    input_ids = input_ids.input_ids.to(device)
    return input_ids, attn_mask

input_ids, attn_mask = prepare_inp(prompt, tokenizer, model.device)

# generate 1 output
output = model.generate(input_ids, max_new_tokens=1, 
                        do_sample=False, 
                        pad_token_id=tokenizer.eos_token_id,
                        attention_mask=attn_mask)
print(tokenizer.decode(output[0], skip_special_tokens=True))

prompt = "print('hello_world')"
input_ids, attn_mask = prepare_inp(prompt, tokenizer, model.device)
# backward pass
loss = model(input_ids, labels=input_ids).loss
loss.backward()
print("Backward pass successful")
print("Loss:", loss.item())

# print trainable parameters
for name, param in model.named_parameters():
    if not param.requires_grad:
        print("NOGRAD:",name, param.shape)

# redo backward pass
loss = model(input_ids, labels=input_ids).loss
loss.backward()
print("Normal Backward pass successful w/ custom")
print("Normal Loss:", loss.item())

print("Freezing all parameters except custom linear layer")
# freeze all trainable parameters except the custom linear layer
for name, param in model.named_parameters():
    if 'custom' not in name.lower():
        param.requires_grad = False
    else:
        print(name, param.shape)

# redo backward pass
loss = model(input_ids, labels=input_ids).loss
loss.backward()
print("Backward pass successful")
print("Loss:", loss.item())

# TODO: make sure you can freeze parts of custom layers - activation func? or keep linear