# IMPACT: Identifying and Classifying Multiple Sourced and Categorized Self-Admitted Technical Debts
---
This  repository contains the models, dataset and experimental code mentioned in the paper. Specifically, experimental code includes the implementation code of our pipeline and the process of training, the dataset includes the preprocessed complete dataset used for training and testing, and models includes our models in pipeline and RQ1-3.
---
#### dataset and models

| Dataset&Models| Description | Link |
| ----------- | ----------- | --------- |
| Dataset-satd_aug | Total augmented dataset including all SATD sentences, their related information, and their division. | [aug_dataset](https://box.nju.edu.cn/library/9b4776c1-af53-4cec-b7c6-60a33f918280/SATD-IMPACT/aug_dataset) |
| Model-glm4-9b-chat-sft-9class | GLM4-9b-Chat model fine-tuned to classify all SATD sentences into 9 categories. | [satd-glm4-9b-chat-sft-9class](https://huggingface.co/chaos1203/satd-glm4-9b-chat-sft-9class) |
| Model-glm4-9b-chat-sft | GLM4-9b-Chat model fine-tuned to apply in IMPACT to classify isSATD sentences into 8 categories. | [satd-glm4-9b-chat-sft](https://huggingface.co/chaos1203/satd-glm4-9b-chat-sft) |
| Model-satd-glm4-9b-chat-sft-noaug | GLM4-9b-Chat model fine-tuned to classify isSATD sentences into 8 categories with the training dataset not augmented. | [satd-glm4-9b-chat-sft-noaug](https://huggingface.co/chaos1203/satd-glm4-9b-chat-sft-noaug) |
| Model-MT-MoE-Bert | MT-BERT Model with MoE trained to apply in IMPACT to identify all SATD sentences into 2 categories (isSATD or nonSATD) | [pytorch_model16.bin](https://box.nju.edu.cn/f/a9455caeac9547159ff2/?dl=1) |
| Model-MT-MoE-Bert-noMerge | Unmerged version of Model-MT-MoE-Bert. One of its roles is to serve as an ablative contrast, and the other is to be used directly to fine-tune the merge version. | [pytorch_model20.bin](https://box.nju.edu.cn/f/a9455caeac9547159ff2/?dl=1) |
| Model-MT-Text-CNN | MT-CNN Model fine-tuned to identify all SATD sentences into 2 categories (isSATD or nonSATD) | [pytorch_model19.bin](https://box.nju.edu.cn/library/9b4776c1-af53-4cec-b7c6-60a33f918280/SATD-IMPACT/MT-Text-CNN) |

#### code
The following is an introduction to the code to make it easier for readers to use.
> The files `main_pipeline_0shot.py` and `main_pipeline_fewshot.py` are the main files to run our pipeline. And The others are tools which used by the two files. The folder `bert_config` contents the related configuration of our bert model. The folder `train_bert` contents the training details of our bert model.
