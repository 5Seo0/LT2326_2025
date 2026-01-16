# Part 3 – Performance Analysis

This section compares the two models used in Part 1 and Part 2 in terms of performance, architecture, and task suitability.

## Part 1: English NER with distilbert-base-uncased

In Part 1, we fine-tuned the English pre-trained model distilbert-base-uncased on the WNUT 17 dataset. This dataset contains English sentences annotated with named entity recognition (NER) tags, for example, O (non-entity), B-group (beginning of a group entity), and I-group (word inside a group entity).

We used AutoTokenizer and AutoModelForTokenClassification classes to prepare and train the model. The tokenize_and_align_labels function handled the alignment between tokens and labels after tokenization.

Despite high accuracy (0.95), the recall remained relatively low (0.38), meaning that the model tended to miss some true positives. While the model was generally good at distinguishing entities from non-entities, it seemed to make conservative in making positive predictions.

## Part 2: Hindi Chunking with distilbert-base-multilingual-cased-ner-hrl

In Part 2, we applied a similar pipeline but using Hindi data. Instead of named entities, we used chunk tags (e.g., NP, VGF) extracted from the ChunkId field in .conllu files. The data was reformatted into BIO tagging scheme (e.g., B-NP, I-NP) after normalizing chunk types (e.g., NP1 to NP).

We fine-tuned the multilingual model distilbert-base-multilingual-cased, which can handle non-English data including Hindi taxt. After correcting the labeling scheme and training on Hindi data, the model performed significantly better.
Final F1 score(0.990) and accuracy (0.990). In addition, precision and recall were both high and balanced. 



### Conclusion

While both models used the same underlying architecture (DistilBERT), their performance differed significantly due to:
	1.	The pretraining language domain (English vs. multilingual).
	2.	The dataset type and label alignment.
	3.	The labeling quality and preprocessing (Part 2 benefited from cleaner BIO labeling).