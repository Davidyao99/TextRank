from textrank import Textrank
import os
import numpy as np

textrank = Textrank(window=2, directed='None')

# test = """Compatibility of systems of linear constraints over the set of natural numbers.
# Criteria of compatibility of a system of linear Diophantine equations, strict
# inequations, and nonstrict inequations are considered. Upper bounds for
# components of a minimal set of solutions and algorithms of construction of
# minimal generating sets of solutions for all types of systems are given.
# These criteria and the corresponding algorithms for constructing a minimal
# supporting set of solutions can be used in solving all the considered types
# systems and systems of mixed types.
# """
# keyphrases=textrank.get_keyphrases(test.lower(), True)
# print(keyphrases)

keys_dir = "./data/keys"
abstracts_dir = "./data/abstracts"
total_complete_correct = 0
total_partial_correct = 0
total_keywords = 0
total_act_keywords = 0
total_abstracts = 0
total_act_keywords_partial_correct = 0
for filename in os.listdir(abstracts_dir):
    name = filename.split('.')[-2]
    abstract_path = os.path.join(abstracts_dir, name+".txt")
    key_path = os.path.join(keys_dir, name+".key")
    print(key_path)
    with open(key_path, mode='r') as f:
        line=f.readline().lower()
        keywords = []
        while line:
            line = line.replace('\n', '')
            line = line.replace('\t', '')
            keywords.append(line)
            line = f.readline()
    with open(abstract_path, mode='r') as f:
        abstract = f.read().lower()
    results, _ = textrank.get_keyphrases_approach1(abstract) # <---- Change here for different approach
    act_keywords = keywords
    len_act_keywords = len(act_keywords)
    act_keywords_tracker = np.array([0] * len_act_keywords) # track which act keywords contains our keywords to calc recall
    for i in results:
        is_complete_correct = 0
        is_partial_correct = 0
        for j in range(len_act_keywords): # check for partial correct keyword
            if i in act_keywords[j]:
                is_partial_correct = 1
                act_keywords_tracker[j] = 1
        if i in act_keywords: # check complete correct
            is_complete_correct = 1
        total_complete_correct += is_complete_correct
        total_partial_correct += is_partial_correct
    total_keywords += len(results)
    total_act_keywords += len(act_keywords)
    total_abstracts += 1
    total_act_keywords_partial_correct += act_keywords_tracker.sum()
precision_partial = total_partial_correct / total_keywords
recall_partial = total_act_keywords_partial_correct / total_act_keywords
F_measure_partial = (2 * precision_partial * recall_partial) / (precision_partial + recall_partial)
precision_complete = total_complete_correct / total_keywords
recall_complete = total_complete_correct / total_act_keywords
F_measure_complete = (2 * precision_complete * recall_complete) / (precision_complete + recall_complete)
print("Overall Precision (partial): " + str(precision_partial))
print("Overall Recall (partial):" + str(recall_partial))
print("Overall F-Measure (partial): " + str(F_measure_partial))
print("Overall Precision (complete): " + str(total_complete_correct / total_keywords))
print("Overall Recall (complete):" + str(total_complete_correct / total_act_keywords))
print("Overall F-Measure (complete): " + str(F_measure_complete))
print("Total actual keywords: " + str(total_act_keywords))
print("Total keywords: " + str(total_keywords))
print("Total partial correct: " + str(total_partial_correct))
print("Total complete correct: " + str(total_complete_correct))
print("Average correct (partial) per abstract: " + str(total_partial_correct / total_abstracts))
print("Average correct (complete) per abstract: " + str(total_complete_correct / total_abstracts))