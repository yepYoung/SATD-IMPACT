import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import recall_score, precision_score

def query_with_nl(tokenizer, model, prompt):
    # device = 'cuda:1'
    gen_kwargs = {"max_length": 18000, "do_sample": True, "top_k": 1}
    query = prompt
    inputs = tokenizer.apply_chat_template([{"role": "user", "content": query}],
                                        add_generation_prompt=True,
                                        tokenize=True,
                                        return_tensors="pt",
                                        return_dict=True
                                        )
    inputs = inputs.to(model.device)
    response = ''
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace('\n', ' ')
        
    return response
    
