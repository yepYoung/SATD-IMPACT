# IMPACT: Identifying and Classifying Multiple Sourced and Categorized Self-Admitted Technical Debts
---
This  repository contains the models, dataset, and experimental code mentioned in the paper. Specifically, experimental code includes the implementation code of our pipeline (IMPACT) and the training process, the dataset includes the preprocessed complete dataset used for training and testing, and models include our models in the pipeline (IMPACT) and RQ1-3. 
---
### 
Although the naming of our directories and files already contains the necessary and brief information, we still provide details in the following table.
| Directories | Description|
| ----------- | ----------- |
|1_Pipeline (IMPACT)| This folder contains a two-stage execution process in our pipeline, i.e., MT-MoE-bert is used for binary identifying, and then glm4-9b-chat is used for 8-classifying. Among them, `pipeline_with_0shot.py` and `pipeline_with_fewshot.py` are the main entry files for the execution of the pipeline, which represent the zero-shot and few-shot execution processes, respectively. |
| 2_MT-Text-CNN | This folder contains the fine-tuning and testing code of MT-Text-CNN. |
|3_Flan-T5 | This folder contains the code for prompting Flan-T5-XXL and fine-tuning Flan-T5-XL.|
|4_MT-MoE-BERT | This folder contains the necessary code for training MT-MoE-BERT to form the pipeline and finish the ablation contrast.|
| 5_GLM-9B | This folder contains the configuration when fine-tuning GLM. |
| cross-project-testset.csv | The cross-project test set used for testing in the paper. |


#### dataset and models

| Dataset&Models| Description | Link |
| ----------- | ----------- | --------- |
| Dataset-satd_aug | Total augmented dataset including all SATD sentences, their related information, and their division. | [aug_dataset](https://box.nju.edu.cn/library/9b4776c1-af53-4cec-b7c6-60a33f918280/SATD-IMPACT/aug_dataset) |
| Model-glm4-9b-chat-sft-9class | GLM4-9b-Chat model fine-tuned to classify all SATD sentences into 9 categories. | [satd-glm4-9b-chat-sft-9class](https://huggingface.co/chaos1203/satd-glm4-9b-chat-sft-9class) |
| Model-glm4-9b-chat-sft | GLM4-9b-Chat model fine-tuned to apply in IMPACT to classify isSATD sentences into 8 categories. | [satd-glm4-9b-chat-sft](https://huggingface.co/chaos1203/satd-glm4-9b-chat-sft) |
| Model-satd-glm4-9b-chat-sft-noaug | GLM4-9b-Chat model fine-tuned to classify isSATD sentences into 8 categories with the training dataset not augmented. | [satd-glm4-9b-chat-sft-noaug](https://huggingface.co/chaos1203/satd-glm4-9b-chat-sft-noaug) |
| Model-MT-MoE-Bert | MT-BERT Model with MoE trained to apply in IMPACT to identify all SATD sentences into 2 categories (isSATD or nonSATD) | [pytorch_model16.bin](https://box.nju.edu.cn/library/9b4776c1-af53-4cec-b7c6-60a33f918280/SATD-IMPACT/MT-MoE-BERT) |
| Model-MT-MoE-Bert-noMerge | Unmerged version of Model-MT-MoE-Bert. One of its roles is to serve as an ablative contrast, and the other is to be used directly to fine-tune the merge version. | [pytorch_model20.bin](https://box.nju.edu.cn/library/9b4776c1-af53-4cec-b7c6-60a33f918280/SATD-IMPACT/MT-MoE-BERT) |
| Model-MT-MoE-Bert-noaug | MT-BERT Model with MoE trained to identify all SATD sentences into 2 categories (isSATD or nonSATD) with not augmented dataset. | [pytorch_model_noaug.bin](https://box.nju.edu.cn/library/9b4776c1-af53-4cec-b7c6-60a33f918280/SATD-IMPACT/MT-MoE-BERT) |
| Model-MT-Text-CNN | MT-CNN Model fine-tuned to identify all SATD sentences into 2 categories (isSATD or nonSATD) | [pytorch_model19.bin](https://box.nju.edu.cn/library/9b4776c1-af53-4cec-b7c6-60a33f918280/SATD-IMPACT/MT-Text-CNN) |
