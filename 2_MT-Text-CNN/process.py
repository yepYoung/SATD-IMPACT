import pandas as pd
import re



def getdata(text):
    text = process_tokenization(text)

    text = process_remove_stopwords(text)


    text = text.rstrip()

    return text




class rule:

    pat_letter = re.compile(r'[^a-zA-Z \! \? \']+')  



def process_tokenization(text):
    new_text = text
    new_text = rule.pat_letter.sub(' ', new_text).strip().lower()
    new_text = re.sub(r"\'", "", new_text)
    new_text = re.sub(r"\!", " ! ", new_text)
    new_text = re.sub(r"\?", " ? ", new_text)
    # remove extra space
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




