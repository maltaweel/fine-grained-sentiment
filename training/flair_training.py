'''
Created on Feb 17, 2021

@author: maltaweel
'''

import pandas as pd
import os

from flair.data_fetcher import NLPTaskDataFetcher
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentRNNEmbeddings, DocumentLSTMEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from pathlib import Path
from flair.datasets import ClassificationCorpus
from flair.models import TextClassifier
from flair.data import Sentence

pn=os.path.abspath(__file__)
pnn=pn.split("training")[0]  
data_file=os.path.join(pnn,'data','sst','sst_dev.csv')
model_path=os.path.join(pnn,'models','flair')
model_file=os.path.join(pnn,'models','flair','best-model.pt')
train_data=os.path.join(pnn,'data','sst','train.csv')
test_data=os.path.join(pnn,'data','sst','test.csv')
dev_data=os.path.join(pnn,'data','sst','dev.csv')
sst_folder=os.path.join(pnn,'data','sst')
    
def processText():
    data = pd.read_csv(data_file, encoding='latin-1').sample(frac=1).drop_duplicates()
    data = data[['Label', 'Text']].rename(columns={"Label":"label", "Text":"text"})
 
    data['label'] = '__label__' + data['label'].astype(str)
    data.iloc[0:int(len(data)*0.8)].to_csv(train_data, sep='\t', index = False, header = False)
    data.iloc[int(len(data)*0.8):int(len(data)*0.9)].to_csv(test_data, sep='\t', index = False, header = False)
    data.iloc[int(len(data)*0.9):].to_csv(dev_data, sep='\t', index = False, header = False);
    
def train():
    corpus: Corpus = ClassificationCorpus(sst_folder,
                                      test_file='test.csv',
                                      dev_file='dev.csv',
                                      train_file='sst_dev.csv') 
    
    label_dict = corpus.make_label_dictionary()
    stacked_embedding = WordEmbeddings('glove')
    
    # Stack Flair string-embeddings with optional embeddings
    word_embeddings = list(filter(None, [
        stacked_embedding,
        FlairEmbeddings('news-forward-fast'),
        FlairEmbeddings('news-backward-fast'),
    ]))
    # Initialize document embedding by passing list of word embeddings
    document_embeddings = DocumentRNNEmbeddings(
        word_embeddings,
        hidden_size=512,
        reproject_words=True,
        reproject_words_dimension=256,
    )
    # Define classifier
    classifier = TextClassifier(
        document_embeddings,
        label_dictionary=label_dict,
        multi_label=False
    )
    
    trainer = ModelTrainer(classifier, corpus)
    trainer.train(model_path, max_epochs=10,train_with_dev=False)

    

def test():
    classifier = TextClassifier.load(model_file)
    sentence=Sentence('Awesome stuff!')
    classifier.predict(sentence)
    print(sentence.labels)

if __name__ == "__main__":
    
    processText()
    train()
    test()
    


