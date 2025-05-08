from get_isSATD import get_isSATD_with_0_or_1
from query_with_nl import query_with_nl
import pandas as pd
from tqdm import tqdm
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from tqdm import tqdm
from K_sample import get_Ksample
from sentence_transformers import SentenceTransformer

# get prompt
def get_prompt(text, task):
    """ prompt example:
    part1: Self-admitted debts have eight common types : Architecture, Build, Code, Defect, Design, Documentation, Requirements, Test.
           Self-admitted debts have four common sources : code-comments, issues, pull-requests, commit-messages.
    part2: Tell me which of the eight types the following technical debt belongs to?
    part3: ### Technical debt text: (From: 1code-comments) As we don't use the CxfSoap component anymore, it's time to clean it up.
    prompt = part1 + part2 + part3
    """
    context = ""
    with open('Pipeline/prompt_0shot.txt', 'r') as f:
        for line in f:
            context += line
    context = context.strip()
    context_sample  = "\nTell me which of the eight types the following technical debt belongs to?\n"
    instruction = context + context_sample
    filenames = ['1code-comments', '2issues', '3pull-requests', '4commit-messages']
    file_name = filenames[task - 1]
    context2 = f"### Technical debt text: (From: {file_name}){text}\n"
    query = instruction + context2
    
    return query

######################################################################## mine ########################################################################
    
    
def process_for_bert(text):
    
    def process_tokenization(text):
        
        pat_letter = re.compile(r'[^a-zA-Z \! \? \']+')
        
        new_text = text
        new_text = pat_letter.sub(' ', new_text).strip().lower()
        new_text = re.sub(r"\'", "", new_text)
        new_text = re.sub(r"\!", " ! ", new_text)
        new_text = re.sub(r"\?", " ? ", new_text)
        new_text = ' '.join(new_text.split())

        return new_text
    
    def process_remove_stopwords(text):
        stop_words = ['the', 'for']
        new_text = text
        text_list = new_text.split()

        text_list = [w for w in text_list if not w in stop_words]
        words = text_list.copy()
        for i in words:
            if i != '!' and i != '?':
                if len(i) <= 2 or len(i) >= 20:
                    text_list.remove(i)
        text_list = " ".join(text_list)
        
        return text_list
        
        
    text = process_tokenization(text)
    text = process_remove_stopwords(text)
    text = text.rstrip()
    
    return text


def lqy_pipline(text, task, glm_tokenizer, glm_model):
    
    processed_text = process_for_bert(text)
    
    lable1 = get_isSATD_with_0_or_1(processed_text, task)
    
    if lable1 == 1:
        prompt = get_prompt(text, task)
        response = query_with_nl(glm_tokenizer, glm_model, prompt)
        
        return response
    elif lable1 == 0:
        
        return 'NON-SATD'
    else:
        
        raise KeyError("label is error")


if __name__ == '__main__':
    
    #=============================================
    model_name = "/root/yyp/LLaMA-Factory/saves/glm4-9b-chat-satd"
    bench_path = '/root/yyp/dev_all_01satd.csv'
    output_dir = 'Pipeline/@Test_results/testset_res_0shot.csv'

    device = "cuda:1"
    k_sample = 4
    # task_id = 1
    #=============================================


    #================ prepare for models and data ================
    glm_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    glm_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map='auto'
    ).eval()

    #================ start pipeline to get results ====================
    res_list = []
    
    benchmark_data = pd.read_csv(bench_path)
    
    
    for i, row in tqdm(benchmark_data.iterrows(), total=benchmark_data.shape[0]):
        index = row['index']
        text = row['text']
        cls = row['class']
        task = row['source']
        aug = row['aug']
        isSATD = row['isSATD']
        
        task_id = int(task[0])

        # pipeline's each response
        response = lqy_pipline(text, task_id, glm_tokenizer, glm_model)
        
        lst_tmp = [task, index, aug, text, isSATD, cls, response]
        res_list.append(lst_tmp)
        
        print(lst_tmp)
        
         
    df_res = pd.DataFrame(res_list, columns=['source', 'index', 'aug', 'text', 'isSATD', 'class', 'predict'])
    df_res.to_csv(output_dir)   

