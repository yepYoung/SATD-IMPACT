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
def get_prompt_fewshot(train_data_embeds, text, task, k, st_model, text_lists, class_lists):
    """prompt example:
    part1: Self-admitted debts have eight common types : Architecture, Build, Code, Defect, Design, Documentation, Requirements, Test.
           Self-admitted debts have four common sources : code-comments, issues, pull-requests, commit-messages.
           Here are some examples:
            { ### Label:Design    ### Technical debt text: (From:2-issues) text_sample}
            { ### Label:Design    ### Technical debt text: (From:2-issues) text_sample}
            { ### Label:Design    ### Technical debt text: (From:2-issues) text_sample}
    part2: Tell me which of the eight types the following technical debt belongs to?
    part3: ### Technical debt text: (From: 1code-comments) As we don't use the CxfSoap component anymore, it's time to clean it up.
    prompt = part1 + part2 + part3
    """
    text_list = text_lists[task - 1]
    class_list = class_lists[task - 1]
    train_data_embeds = train_data_embeds[task - 1]
    prompt1 = ''
    with open('Pipeline/prompt_fewshot.txt', 'r') as f:
        for line in f:
            prompt1 += line
    prompt1 = prompt1.strip()
    prompt1 += '\n'
    
    filenames = ['1code-comments', '2issues', '3pull-requests', '4commit-messages']
    file_name = filenames[task - 1]
    sample_list = get_Ksample(train_data_embeds, text, k, st_model, text_list, class_list)
    prompt2 = ''
    for text_sample, label in sample_list:
        prompt2_part1 = '{' + f'### Label:{label}'
        prompt2_part2 = f'   ### Technical debt text: (From:{file_name}){text_sample}' + '}\n'
        prompt2 += prompt2_part1 + prompt2_part2
    
    query = "\nTell me which of the eight types the following technical debt belongs to?\n"
   
    context2 = f"### Technical debt text: (From: {file_name}){text}\n"
    
    prompt = prompt1 + prompt2 + query + context2
    
    return prompt


def pipline(text, text_pro, task, glm_tokenizer, glm_model, k_sample):
    lable1 = get_isSATD_with_0_or_1(text_pro, task)
    if lable1 == 1:
        prompt = get_prompt_fewshot(train_data_embeds_set, text, task, k_sample, st_model, text_lists, class_lists)
        response = query_with_nl(glm_tokenizer, glm_model, prompt)
        return response
    elif lable1 == 0:
        return 'NON-SATD'
    else:
        raise KeyError("label is error")


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


def lqy_pipline(text, task, glm_tokenizer, glm_model, k_sample, train_data_embeds_set):
    
    processed_text = process_for_bert(text)
    
    lable1 = get_isSATD_with_0_or_1(processed_text, task)
    
    if lable1 == 1:
        prompt = get_prompt_fewshot(train_data_embeds_set, text, task, k_sample, st_model, text_lists, class_lists)
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
    output_dir = 'Pipeline/@Test_results/testset_res_fewshot.csv'

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

    # retrieval model
    st_model = SentenceTransformer('bert-base-nli-mean-tokens')
    
    
    # KNN database
    knn_tarin_data_path_1 = 'Pipeline/train_data/train_1code-comments.csv'
    knn_tarin_data_path_2 = 'Pipeline/train_data/train_2issues.csv'
    knn_tarin_data_path_3 = 'Pipeline/train_data/train_3pull-requests.csv'
    knn_tarin_data_path_4 = 'Pipeline/train_data/train_4commit-messages.csv'
    knn_tarin_data_pathes = [knn_tarin_data_path_1, knn_tarin_data_path_2, knn_tarin_data_path_3, knn_tarin_data_path_4]
    
    text_lists = []
    class_lists = []
    train_data_embeds_set = []
    
    for i in range(4):
        knn_tarin_data_path = knn_tarin_data_pathes[i]
        df = pd.read_csv(knn_tarin_data_path)
        text_list = []
        class_list = []
        for index, row in df.iterrows():
            text_list.append(row['text'])
            class_list.append(row['class'])  
        train_data_embeds = st_model.encode(text_list, show_progress_bar=True)
        text_lists.append(text_list)
        class_lists.append(class_list)
        train_data_embeds_set.append(train_data_embeds)
    #============================================


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
        response = lqy_pipline(text, task_id, glm_tokenizer, glm_model, k_sample, train_data_embeds_set)
        
        lst_tmp = [task, index, aug, text, isSATD, cls, response]
        res_list.append(lst_tmp)
        
        print(lst_tmp)
        
         
    df_res = pd.DataFrame(res_list, columns=['source', 'index', 'aug', 'text', 'isSATD', 'class', 'predict'])
    df_res.to_csv(output_dir)   

