import nltk
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import os

class Textrank:

    def __init__(self, window=2, directed="None", d=0.85, er_threshold=0.008):
        self.window = window
        self.d = d
        self.er_threshold = er_threshold
        self.directed = directed

    def _filter(self, tokenized_text):
        unwanted = "[]“”'’"
        tokenized_text = [word for word in tokenized_text if ((word not in unwanted))]
        is_valid = lambda word_pos: True if (word_pos[1][:2]=='NN') | (word_pos[1][:2] == 'JJ') | (word_pos[1] == 'VBG') else False
        filtered_words = [ word_pos[0] for word_pos in nltk.pos_tag(tokenized_text) if is_valid(word_pos)]
        return filtered_words
        

    def _get_matrix(self, filtered_text, tokenized_text): # filtered_text refers to tokenized and filtered text
        word_to_idx = {}
        vertices = []
        n = 0
        for word in filtered_text: # builds a mapping of words to index
            if word in word_to_idx:
                continue
            word_to_idx[word] = n
            vertices.append(word)
            n += 1
        matrix = np.zeros((n,n))
        for i in range(len(tokenized_text)): # finds window in original tokenized_text
            cur_word = tokenized_text[i]
            if cur_word not in word_to_idx:
                continue
            col = word_to_idx[cur_word] # word on col -> word on row for directed graph

            if self.directed == 'None':
                lwr_bound = max(0, i-self.window + 1)
                upr_bound = min(len(tokenized_text), i + self.window)
            elif self.directed == 'Forward':
                lwr_bound = i + 1
                upr_bound = min(len(tokenized_text), i + self.window)
            else:
                lwr_bound = max(0, i-self.window + 1)
                upr_bound = i
            for j in range(lwr_bound, upr_bound): # iterating through window around cur_word
                target_word = tokenized_text[j]
                if (target_word in word_to_idx) and (target_word != cur_word):
                    row = word_to_idx[target_word]
                    matrix[row][col] = 1 # set matrix entry to row representing vertex present between cur_word and target_word
        col_sum = matrix.sum(axis=0)
        col_sum[col_sum == 0] = 1
        normalized_matrix = matrix / col_sum
        return normalized_matrix, np.array(vertices) # change to np array so no indexing type error

    def _visualize_wordcloud(self, freq_dict):
        wordcloud = WordCloud(width=900,height=500, max_words=1628,relative_scaling=1,normalize_plurals=False).generate_from_frequencies(freq_dict)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()

    def _get_topT(self, T, scores, vertices): # list of vertices with corresponding score in scores
        idx_top_T = scores.argsort()[-T:][::-1]
        vertices_top_T = vertices[idx_top_T]
        return vertices_top_T

    def _get_entity_to_score(self, scores, vertices):
        ordered_idx = scores.argsort()[::-1]
        vertices_to_scores = {}
        for i in ordered_idx:
            vertices_to_scores[vertices[i]] = scores[i]
        return vertices_to_scores

    def get_keywords(self, text, visualize=False):
        tokenized_text = word_tokenize(text)
        filtered_text = self._filter(tokenized_text)
        matrix, vertices = self._get_matrix(filtered_text, tokenized_text)
        pagerank_values = np.full((len(vertices), 1), 1.0)
        converged = False
        for _ in range(50):
            new_pagerank_values = (1-self.d) + self.d * np.dot(matrix, pagerank_values)
            diff_pagerank_values = abs(new_pagerank_values - pagerank_values)
            pagerank_values = new_pagerank_values
            if np.sum(diff_pagerank_values) < self.er_threshold:
                converged = True
                break
        if not converged:
            print("50 iterations reached without convergence")
        pagerank_values = pagerank_values.flatten()
        keywords_top_T = self._get_topT(len(vertices) // 2, pagerank_values, vertices)
        keyword_to_rank = self._get_entity_to_score(pagerank_values, vertices)
        if visualize:
            self._visualize_wordcloud(keyword_to_rank)
        return keywords_top_T, keyword_to_rank # return top T keywords in descending order
        
    def get_keyphrases_approach2(self, text, visualize=False): # form multi-word keywords using single word keywords
        tokenized_text = word_tokenize(text)
        keywords, keyword_to_rank = self.get_keywords(text)
        keyphrases = []
        keyphrases_score = []
        left_bnd = 0
        srch_window = 1
        vertices = list(keyword_to_rank.keys())
        keywords_dict = dict.fromkeys(vertices, False) # to track keywords that were used to form multi-word keywords
        n = len(tokenized_text)
        while left_bnd < n:
            cur_word = tokenized_text[left_bnd]
            if cur_word not in vertices:
                left_bnd += srch_window
                continue
            keyphrase_rank = 0 # total rank of the phrase
            while ((left_bnd + srch_window < n) and (tokenized_text[left_bnd + srch_window] in vertices)): 
                srch_window += 1 # increase window size if subsequent word is a keyword
            right_bnd = min(left_bnd + srch_window, n-1)
            keyphrase_list = tokenized_text[left_bnd:right_bnd]
            if len(keyphrase_list) > 1:
                for word in keyphrase_list:
                    keywords_dict[word] = True
                    keyphrase_rank += keyword_to_rank[word]
            else:
                keyphrase_rank = keyword_to_rank[cur_word]
            keyphrase = ' '.join(keyphrase_list)
            if keyphrase not in keyphrases: # add if phrase is a new phrase
                keyphrases.append(keyphrase)
                keyphrases_score.append(keyphrase_rank)
            left_bnd += srch_window
            srch_window = 1
        for key, value in keywords_dict.items(): # remove single keyword that formed multi-word keywords
            if value:
                try:        
                    idx = keyphrases.index(key)
                    keyphrases_score.pop(idx)
                    keyphrases.pop(idx)
                except:
                    continue
        keyphrases_score = np.array(keyphrases_score)
        keyphrases = np.array(keyphrases)
        keyphrases_topT = self._get_topT(len(vertices) // 3, keyphrases_score, keyphrases)
        keyphrase_to_rank = self._get_entity_to_score(keyphrases_score, keyphrases)
        if visualize:
            self._visualize_wordcloud(keyphrase_to_rank)
        return keyphrases_topT, keyphrase_to_rank

    def get_keyphrases_approach1(self, text, visualize=False): # form multi-word keywords using single word keywords
        tokenized_text = word_tokenize(text)
        keywords, keyword_to_rank = self.get_keywords(text)
        keyphrases = set()
        n = len(tokenized_text)
        left_bnd = 0
        keywords_dict = dict.fromkeys(keywords, False) # to track keywords that were used to form multi-word keywords
        while left_bnd < n:
            srch_window = 1
            cur_word = tokenized_text[left_bnd]
            if cur_word not in keywords:
                left_bnd += srch_window
                continue
            while ((left_bnd + srch_window < n) and (tokenized_text[left_bnd + srch_window] in keywords_dict)): 
                srch_window += 1 # increase window size if subsequent word is a keyword
            right_bnd = left_bnd + srch_window
            keyphrase_list = tokenized_text[left_bnd:right_bnd]
            keyphrase = ' '.join(keyphrase_list)
            if keyphrase not in keyphrases: # add if phrase is a new phrase
                keyphrases.add(keyphrase)
                if len(keyphrase_list) > 1:
                    for word in keyphrase_list:
                        keywords_dict[word] = True
            left_bnd = right_bnd + 1
        for key, value in keywords_dict.items(): # remove single keyword that formed multi-word keywords
            if value:
                keyphrases.discard(key)
        return keyphrases, None