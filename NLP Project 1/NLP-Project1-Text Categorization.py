import os
import sys
import string
import nltk
import tqdm
import numpy as np

from math import log
from nltk.stem import PorterStemmer

nltk.download('punkt')

vocab_list = []
category = document = dict()
document_count = 0
p_s = PorterStemmer()

f_1 = input("Please type in the name of the file with the labeled training data: ")

with tqdm.tqdm(total = os.path.getsize(f_1)) as p_bar:
    with open(f_1, "r") as f1:
        for line in f1:
            words = line.split()
            file_line = open(words[0], 'r').read()
            category_line = words[1]
            document_count = document_count + 1

            if category_line not in document.keys():
                document[category_line] = dict()
                category[category_line] ={}
                category[category_line]["wordCount"] = 0
                category[category_line]["documentCount"] = 0

            category[category_line]["documentCount"] = category[category_line]["documentCount"] + 1

            tokens = nltk.word_tokenize(file_line)
            for token in tokens:
                token = p_s.stem(token)
                if token in string.punctuation:
                    continue
                if token in document[category_line].keys():
                    category[category_line]["wordCount"] = category[category_line]["wordCount"] + 1
                    document[category_line][token] = document[category_line][token] + 1
                else:
                    category[category_line]["wordCount"] = category[category_line]["wordCount"] + 1
                    document[category_line][token] = 1

                vocab_list.append(token)
            p_bar.update(len(line))

vocab_list_not_unique = vocab_list
vocab_list = np.unique(vocab_list)

print("Successfully loaded in and tokenized the labeled training data file!")

logprior = dict()
for c in document.keys():
    logprior[c] = log(category[c]["documentCount"]/document_count)

smoothing_method = input("Please type in a smoothing method ('laplace' for Laplace and 'JM' for Jelinek-Mercer): ")

if smoothing_method == "JM":
    alpha = 0.01
elif smoothing_method == "laplace":
    alpha = 0.055

loglikelihood = dict()

for c1 in document.keys():
    loglikelihood[c1]= {}
    for word in vocab_list:
        if word not in loglikelihood[c1].keys():
            loglikelihood[c1][word] = {}
        if word not in document[c1].keys():
            document[c1][word] = 0

        wc_count = document[c1][word]
        wc_p = 0

        for c2 in document.keys():
            if word in document[c2].keys():
                wc_p = wc_p + document[c2][word]
            else:
                wc_p = wc_p + 0

        if smoothing_method == "JM":
            loglikelihood[c1][word] = log((1 - alpha)*(wc_count)/(category[c1]["wordCount"]) + alpha*(wc_p/len(vocab_list)))    
        elif smoothing_method == "laplace":
            loglikelihood[c1][word] = log((alpha + wc_count)/(category[c1]["wordCount"] + alpha*len(vocab_list)))

output = []

f_2 = input("Please type in the name of the file with the unlabeled test list: ")

with tqdm.tqdm(total = os.path.getsize(f_2)) as p_bar:
    with open(f_2, "r") as f2:
        for line in f2:
            locate_file = line.rstrip("\n\r")
            file_line = open(locate_file, 'r').read()
            tokens = nltk.word_tokenize(file_line)
            total = dict()
            for c in document.keys():
                total[c] = logprior[c]
                for token in tokens:
                    token = p_s.stem(token)
                    if token in string.punctuation:
                        continue
                    if token in loglikelihood[c].keys():
                        total[c] = total[c] + loglikelihood[c][token]

            output.append(locate_file + " " + max(total, key = total.get) + "\n")
            p_bar.update(len(line))

print("Successfully loaded in and tokenized the unlabeled test list file!")

f_3 = input("Please type in the name of the output file to write the predictions data to: ")

outfile = open(f_3, "w")
for line_output in output:
    outfile.write(line_output)
outfile.close()

print("Successfully written the predictions data to the file!")