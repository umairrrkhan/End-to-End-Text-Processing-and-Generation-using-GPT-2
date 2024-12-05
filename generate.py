from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_directory = './nanogpt'

tokenizer = AutoTokenizer.from_pretrained(model_directory)
model = AutoModelForCausalLM.from_pretrained(model_directory, torch_dtype=torch.float32)


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  

if model.config.pad_token_id is None:
    model.config.pad_token_id = model.config.eos_token_id  

input_text = "Once upon a time"
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

outputs = model.generate(
    inputs['input_ids'], 
    max_length=100, 
    attention_mask=inputs['attention_mask'], 
    do_sample = True,
    temperature=0.7,  
    top_k=50  
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
