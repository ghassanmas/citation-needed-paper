import sys
import json
import requests
import mwparserfromhell
from bs4 import BeautifulSoup as bs 
import nltk.data
import re
import argparse
import pandas as pd
import pickle
import numpy as np
import types

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix

from keras.utils import to_categorical

from keras import backend as K





search_link='https://en.wikipedia.org/w/api.php?action=query&list=search&utf8=&format=json&srsearch='
article_link='https://en.wikipedia.org/w/api.php?action=parse&format=json&pageid='

def text_to_word_list(text):
    # check first if the statements is longer than a single sentence.
    sentences = re.compile('\.\s+').split(str(text))
    if len(sentences) != 1:
        # text = sentences[random.randint(0, len(sentences) - 1)]
        text = sentences[0]

    text = str(text).lower()

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    text = text.strip().split()

    return text


'''
    Compute P/R/F1 from the confusion matrix.
'''


'''
    Create the instances from our datasets
'''


def construct_instance_reasons(statements, section_dict_path, vocab_w2v_path, max_len=-1):
    # Load the vocabulary
    vocab_w2v = pickle.load(open(vocab_w2v_path, 'rb'))

    # load the section dictionary.
    section_dict = pickle.load(open(section_dict_path, 'rb'))

    # Load the statements, the first column is the statement and the second is the label (True or False)
    #statements = pd.read_csv(statement_path, sep='\t', index_col=None, error_bad_lines=False, warn_bad_lines=False)

    # construct the training data
    X = []
    sections = []
    y = []
    outstring=[]
    statments_last=[]
    for i in range(len(statements[0])):
        try:
                statement_text = text_to_word_list(statements[1][i])

                X_inst = []
                for word in statement_text:
                    if max_len != -1 and len(X_inst) >= max_len:
                        continue
                    if word not in vocab_w2v:
                        X_inst.append(vocab_w2v['UNK'])
                    else:
                        X_inst.append(vocab_w2v[word])

            # extract the section, and in case the section does not exist in the model, then assign UNK
                section = statements[1][i].strip().lower()
                sections.append(np.array([section_dict[section] if section in section_dict else 0]))
 #           label = row['citations']

            # some of the rows are corrupt, thus, we need to check if the labels are actually boolean.
#            if type(label) != types.BooleanType:
#                continue
#                y.append(label)
                X.append(X_inst)
                statments_last.append(statements[1][i])
#            outstring.append(str(row["statement"]))
            #entity_id  revision_id timestamp   entity_title    section_id  section prg_idx sentence_idx    statement   citations

        except Exception as e:
            print 'Error Error Error'
            print statements[0][i],statements[1][i]
            print e.message
    X = pad_sequences(X, maxlen=max_len, value=vocab_w2v['UNK'], padding='pre')
    encoder = LabelBinarizer()
#    y = encoder.fit_transform(y)
#    y = to_categorical(y)

#    return X, np.array(sections), y, encoder, outstring
    return X, np.array(sections),statments_last











query = raw_input('Please type Wikipedia article title\n') 

response=requests.get(search_link+query)
articles=json.loads(response.text)['query']['search']

if(len(articles))==0:
    sys.exit()
else:
    for i in range(len(articles)):
        print i+1, articles[i]['title']
    article_index = int(raw_input('Please type the article number from the table above\n'))-1
    print 'Selected Article is' , articles[article_index]['title']
    article_req=requests.get(article_link+str(articles[article_index]['pageid']))
    article_raw=json.loads(article_req.text)
    html=(mwparserfromhell.parse(article_raw['parse']['text']['*']))
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    parsed=bs(html.encode('utf-8').strip())

    
    
    
    sections_pre=[]
    sentences=[]
    section = 'MAIN_SECTION' #Alaways the first section is the main section
    print '--------------------------'
    for i in((list(parsed.select(".mw-parser-output")[0].children))):
        if i.name=='h2':   #if its a h2 tag then it must be new header
            section=i.text.split('[edit]')[0]
            if(section=='See also'):
                break
        elif i.name=='p':
            paragraph_sentences = (tokenizer.tokenize(i.text))
            paragraph_sentences = [sent for sent in paragraph_sentences if len(sent)>4]
            len_sent = len(paragraph_sentences)
            sections_pre.extend([sections_pre]*len_sent)
            sentences.extend(paragraph_sentences)
    
    
    model = load_model('models/fa_en_model_rnn_attention_section.h5')

    # load the data
    max_seq_length = model.input[0].shape[1].value
    X, sections,statments_last = construct_instance_reasons([sections_pre,sentences],'embeddings/section_dict_en.pck','embeddings/word_dict_en.pck', max_seq_length)

    # classify the data
   
    pred = model.predict([X, sections])
    pred=np.sort(pred)
    print (pred.size)


    # store the predictions: printing out the sentence text, the prediction score, and original citation label.+ sentences[1][idx] 
    outstr = 'Text\tPrediction\n'
    print 'Text \t Predction'  
    for idx, y_pred in enumerate(pred):
        outstr += statments_last[idx].encode('utf-8')+'\t'+str(y_pred[0])+ '\n'
        print statments_last[idx].encode('utf-8')+'\t'+str(y_pred[0])+ '\n'
    fout = open('results' + '/' + 'en' + '_predictions_sections.tsv', 'wt')
    fout.write(outstr)
    fout.flush()
    fout.close()

            
            
         










