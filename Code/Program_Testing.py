import wikipedia
import csv
import nltk
from nltk import pos_tag
from textblob.classifiers import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
from nltk import RegexpParser
from nltk.corpus import stopwords
import pickle
from nltk.corpus import wordnet
import sys
from nltk.stem import WordNetLemmatizer

path = sys.argv[1]
testing_set = []
all_title = []
reader = csv.reader(open(path+'/DataSet_Testing.csv', 'r'))

def equating(text):
    if "Business" in text:
        text = "Business"
    if "Politics" in text:
        text = "Politics"
    if "Sports" in text:
        text = "Sports"   
    if "Technology" in text:
        text = "Technology"
    if "Travel" in text:
        text = "Travel"
    return text

for row in reader:
    print("Data : " + str(row))
    title, category = row
    wiki_page = wikipedia.page(title)
    all_title.append(title) # Used to print in output in last paragraph
    wiki_content = str.lower(wiki_page.summary)
    word_tokens = word_tokenize(wiki_content)
    print("Tokenizing Words : ")
    print(word_tokens)
    print("\n")
    big_words = [k for k in word_tokens if len(k) >= 3 and not k.startswith('===')]


    #------------------Remove StopWords--------------------
    stop = set(stopwords.words('english'))
    no_stopwords_list = [k for k in big_words if k not in stop]
    print("After Removing stopwords : ")
    print(no_stopwords_list)
    print("\n")


    #------------------ Lemmatization --------------------
    lemmatized_tokens = []
    lemmatizer = WordNetLemmatizer()
    for word in no_stopwords_list:
        lemmatized_tokens.append(lemmatizer.lemmatize(word))
    print("Lemmatization :")
    print(lemmatized_tokens)
    print("\n")


    #------------------ POS Tagging --------------------
    pos_tagged_list = pos_tag(lemmatized_tokens)
    print("POS Tagging :")
    print(pos_tagged_list)
    print("\n")


    #------------------ Chunking (Shallow Parsing) -----------------------
    grammar = """ NP: {<DT>?<JJ>*<NN>}
                      {<NNP>+}
                      {<NN><NN>}
                      {<NNS><VBP>}
                      {<V.*> <TO> <V.*>}
                      {<N.*>(4,)} """
    NPChunker = RegexpParser(grammar)
    chunked_result = NPChunker.parse(pos_tagged_list)
    print("Chunked Tree:")
    print(chunked_result)
    print("\n")

    #------------------ Shallow Parsed List  -----------------------
    shallow_parsed_list = list()
    for sub_tree in chunked_result:
        if type(sub_tree) is nltk.tree.Tree:
            if sub_tree.label() == 'NP':
                for wrd, tg in sub_tree.leaves():
                    if 'NN' in tg:
                        shallow_parsed_list.append(wrd)
    print("Shallow Parsing :")
    print(shallow_parsed_list)
    print("\n")


    #------------------ Hypernym Parsing -----------------------
    hypernym_parsed_list = list()

    for text in shallow_parsed_list:
        for synset in wordnet.synsets(text, pos='n'):
            word = synset.name()
            word_synset = wordnet.synset(word)
            # Accessing hypernyms
            hypernym_list = word_synset.hypernyms()
            for hypernym in hypernym_list:
                if "Business" in text or "Technology" in text or "Politics" in text or "Travel" in text or "Sports" in text:
                    isChanged = False
                    if "Business" in text:
                        text = "Business"
                        isChanged = True
                    if "Politics" in text:
                        text = "Politics"
                        isChanged = True
                    if "Sports" in text:
                        text = "Sports"
                        isChanged = True
                    if "Technology" in text:
                        text = "Technology"
                        isChanged = True                    
                    if "Travel" in text:
                        text = "Travel"
                        isChanged = True
                    if isChanged:
                        hypernym_parsed_list.append(text)
        hypernym_parsed_list.append(text)
    print("Hypernym :")
    print(hypernym_parsed_list)
    print("\n")


    #------------------ Meronym Parsing -----------------------
    meronym_parsed_list = list()

    for text in hypernym_parsed_list:
        for synset in wordnet.synsets(text, pos='n'):
            word = synset.name()
            word_synset = wordnet.synset(word)
            # Accessing hypernyms
            meronym_list = word_synset.part_meronyms()
            for meronym in meronym_list:
                if "Business" in text or "Technology" in text or "Politics" in text or "Travel" in text or "Sports" in text:
                    isChanged = False
                    if "Business" in text:
                        text = "Business"
                        isChanged = True
                    if "Politics" in text:
                        text = "Politics"
                        isChanged = True
                    if "Sports" in text:
                        text = "Sports"
                        isChanged = True
                    if "Technology" in text:
                        text = "Technology"
                        isChanged = True                    
                    if "Travel" in text:
                        text = "Travel"
                        isChanged = True
                    if isChanged:
                        meronym_parsed_list.append(text)
        meronym_parsed_list.append(text)
    print("Meronym matching : ")
    print(meronym_parsed_list)
    print("\n")


#------------------ Join all Sentence And Assign Category -----------------------    
    join_tokens = " ".join(meronym_parsed_list)
    indv_sentence = (join_tokens, category)
    testing_set.append(indv_sentence)


#------------------ Get Naive Baye's Classification from pickle file  -----------------------
open_classifier = open(path+"/naiveBayes.pickle", "rb")
classifier = pickle.load(open_classifier)
open_classifier.close()


#------------------ Classify test against Training Naive Baye's result -----------------------
i = 0
for test in testing_set:
    classified = classifier.classify(test[0])
    print("Topic : " + all_title[i])
    print("Assigned Category : " + str(test[1]).strip())
    print("Obtained Category : " + classified)
    i+=1

print("Accuracy : " + str(classifier.accuracy(testing_set) * 100))
