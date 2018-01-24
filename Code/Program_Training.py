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
training_set = []
reader = csv.reader(open(path+'/DataSet_Training.csv', 'r'))

for row in reader:
    print("Data : " + str(row))
    title, category = row
    #Add a multi-word expression to the lexicon
    wiki_page = wikipedia.page(title)
    #<WikipediaPage 'Abu Dhabi'>
    wiki_content = str.lower(wiki_page.summary)
    #Paragraph separated by space
    word_tokens = word_tokenize(wiki_content)
    #['Abu','Dhabi']
    #Remove Stopwords
    stop = set(stopwords.words('english'))
    big_words = [k for k in word_tokens if len(k) >= 3 and not k.startswith('===') and k not in stop]
    #print(bigger_words)



    #------------------Lemmatization and POS Tagging--------------------
    lemmatized_tokens = []
    #Lematize to a common Word base
    lemmatizer = WordNetLemmatizer()
    for word in big_words:
        lemmatized_tokens.append(lemmatizer.lemmatize(word))
    #Add POS Tagging to each Lematized Words
    pos_tagged_list = pos_tag(lemmatized_tokens)  #[('Abu','NN')]



    #------------------ Chunking (Shallow Parsing) -----------------------
    grammar = """ NP: {<DT>?<JJ>*<NN>}
                      {<NNP>+}
                      {<NN><NN>}
                      {<NNS><VBP>}
                      {<V.*> <TO> <V.*>}
                      {<N.*>(4,)} """
    NPChunker = RegexpParser(grammar) #Chunking Rule
    # Return the best chunk structure for the given tokens and return a tree.
    # http://www.bogotobogo.com/python/NLTK/chunking_NLTK.php
    chunked_result = NPChunker.parse(pos_tagged_list)
    '''
    (S
        (NP accenture/NN)
        (NP plc/NN)
        (NP global/JJ management/NN)
    '''



    #------------------ Shallow Parsed List  -----------------------
    #We only need nouns that contribute to the sentence. Ignore everything else.
    shallow_parsed_list = list()
    for sub_tree in chunked_result:
        if type(sub_tree) is nltk.tree.Tree:
            if sub_tree.label() == 'NP':
                for wrd, tg in sub_tree.leaves():
                    if 'NN' in tg:
                        shallow_parsed_list.append(wrd)
    #['accenture', 'plc', 'management', 'service', 'company', 'strategy', 'technology', 'operation', 'service', 'fortune', 'company', 'dublin', 'ireland', 'september', 'company', 'revenue', 'employee', 'client', 'city', 'country', 'company', 'employee', 'india', 'philippine', 'august', 'apple', 'inc.', 'partnership', 'accenture', 'create', 'io', 'business', 'solution', 'accenture', 'client', 'equity', 'york', 'stock', 'exchange', 'symbol', 'acn', 'index', 'july']



    #hypernym is a word which includes the meanings of other words. For instance, flower is a hypernym of daisy and rose. 
    #------------------ Semantic : Hypernym Parsing -----------------------
    hypernym_parsed_list = list()

    for text in shallow_parsed_list: # management, service
        for synset in wordnet.synsets(text, pos='n'): # Synset('management.n.01'), Synset('service.n.01')
            word = synset.name() # management.n.01, service.n.01
            word_synset = wordnet.synset(word) #Synset('management.n.01'), Synset('service.n.01')
            # Get list of all Hypernymns
            hypernym_list = word_synset.hypernyms() # [Synset('social_control.n.01')] [Synset('administration.n.02')]
            for hypernym in hypernym_list: #hypernym: Synset('social_control.n.01') text: management
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
    #['accenture', 'plc', 'management', 'service', 'company', 'strategy', 'technology', 'operation', 'service', 'fortune', 'company', 'dublin', 'ireland', 'september', 'company', 'revenue', 'employee', 'client', 'city', 'country', 'company', 'employee', 'india', 'philippine', 'august', 'apple', 'inc.', 'partnership', 'accenture', 'create', 'io', 'business', 'solution', 'accenture', 'client', 'equity', 'york', 'stock', 'exchange', 'symbol', 'acn', 'index', 'july']



    #Meronymy is used to describe a part-whole relationship between lexical items. Thus cover and page are meronyms of book.
    #------------------ Semantics : Meronym Parsing -----------------------
    meronym_parsed_list = list()

    for text in hypernym_parsed_list:
        for synset in wordnet.synsets(text, pos='n'):
            word = synset.name()
            word_synset = wordnet.synset(word)
            meronym_list = word_synset.part_meronyms() #[Synset('agra.n.01'), Synset('andhra_pradesh.n.01'), Synset('assam.n.01'), Synset('bangalore.n.01')]
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
    #['accenture', 'plc', 'management', 'service', 'company', 'strategy', 'technology', 'operation', 'service', 'fortune', 'company', 'dublin', 'ireland', 'september', 'company', 'revenue', 'employee', 'client', 'city', 'country', 'company', 'employee', 'india', 'philippine', 'august', 'apple', 'inc.', 'partnership', 'accenture', 'create', 'io', 'business', 'solution', 'accenture', 'client', 'equity', 'york', 'stock', 'exchange', 'symbol', 'acn', 'index', 'july']
    


    #------------------ Join all Sentence And Assign Category -----------------------
    join_tokens = " ".join(meronym_parsed_list)
    indv_sentence = (join_tokens, category) #('accenture plc management service company strategy technology operation service fortune company dublin ireland september company revenue employee client city country company employee india philippine august apple inc. partnership accenture create io business solution accenture client equity york stock exchange symbol acn index july', ' Business')
    training_set.append(indv_sentence) #[('abu dhabi أبو ظبي\u200e ẓabī pronunciation ɐˈbuˈðˤɑbi capital city emirate dubai capital emirate abu abu island coast city population dhabi house government office seat arab government home abu dhabi family president family abu dhabi development urbanisation income population city today city country activity centre position capital abu dhabi account emirate economy', ' Travel'), ('accenture plc management service company strategy technology operation service fortune company dublin ireland september company revenue employee client city country company employee india philippine august apple inc. partnership accenture create io business solution accenture client equity york stock exchange symbol acn index july', ' Business')]


#“Pickling” is the process whereby a Python object hierarchy is converted into a byte stream
#The pickle module keeps track of the objects it has already serialized, so that later references to the same object won’t be serialized again.
#------------------ Naive Bayes Classification -----------------------
print("Training the program : Done")
classifier = NaiveBayesClassifier(training_set)
print(classifier)
save_classifier = open(path+"/naiveBayes.pickle","wb")
pickle.dump(classifier, save_classifier) # Save the classifier in a pickle file to use against texting element
save_classifier.close()