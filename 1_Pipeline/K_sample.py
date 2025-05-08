import sentence_transformers
from sentence_transformers import SentenceTransformer
import pandas as pd


def get_the_most_relevant_items_for_an_item(query_embed, keys_embed, k):
  cos_sim = sentence_transformers.util.cos_sim(query_embed, keys_embed)
  itemId_similarity = dict(zip(range(len(keys_embed)),cos_sim.tolist()[0]))
  itemId_similarity = dict(sorted(itemId_similarity.items(), key=lambda item: item[1], reverse=True)) # sort
  itemId_similarity = [(idx, itemId_similarity[idx]) for idx in list(itemId_similarity)[:k]] # take top k
  return itemId_similarity 




def get_Ksample(train_data_embeds, text,k, st_model, text_list, class_list):
    text_data_embeds = st_model.encode(text)
    tups = get_the_most_relevant_items_for_an_item(text_data_embeds, train_data_embeds, k)
    tup_list = []
    for tup in tups:
        (i, t) = tup
        tup_list.append((text_list[i], class_list[i]))
    
    return tup_list
