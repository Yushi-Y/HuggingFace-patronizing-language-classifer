# Detect patronizing and condescending language with BERT ensemble models on HuggingFace 
Description:
- Trained three BERT-variant models (DistilBERT, RoBERTa) on the training set and ensembled their predictions to classify patronizing and condescending language, with the ensembled model outperforming SemEval 2022's Task 4 baseline RoBERTa model.

Categories:
- Unbalanced power relations
- Shallow solution
- Presupposition
- Authority voice
- Metaphor
- Compassion
- The poorer, the merrier

Data format:
- Task 1: paragraph_id keyword country_code paragraph label
- Task 2: paragraph_id, paragraph, keyword, country_code, span_start, span_end, span_text, category_label, number_of_annotators_agreeing_in_that_label

Features: Country, Keyword

Useful links:
- **Our report:** [NLP_Report](https://github.com/Yushi-Y/patronizing-language-classifer/blob/main/NLP_Report.pdf)
  
- **CW spec:** https://static.us.edusercontent.com/files/mCjbUPfgDjtNLLx4hdvs2TcF

- **Data paper:** https://aclanthology.org/2020.coling-main.518.pdf

- **Competition:** https://competitions.codalab.org/competitions/34344

- **Competition github:** https://github.com/Perez-AlmendrosC/dontpatronizeme/


Model improvements:
- Cross-validation, precision, recall, f1
- Different pre-trained models (save trained models for ensembling)
- Categorical injection at embedding
- Hyperparameters: lr, lr_schedule, epochs, dropout (final layer)
- Ensemble (average or try linear model)
- Predict the original labels
