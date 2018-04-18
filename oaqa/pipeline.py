from flask import Flask, request, abort, render_template
from flask import jsonify, render_template
import sys
import json
import copy
from nltk.tokenize import sent_tokenize, word_tokenize
from SpanSelector import SpanSelector
import dynet as dy

from Expander import Expander
from NoExpander import NoExpander
from SnomedctExpander import SnomedctExpander
from UMLSExpander import UMLSExpander

from BiRanker import BiRanker
from CoreMMR import CoreMMR
from SoftMMR import SoftMMR
from HardMMR import HardMMR
from MaLSTMScorer import MaLSTMScorer
from BiLSTMScorer import BiLSTMScorer

from Tiler import Tiler
from Concatenation import Concatenation

from Fusion import Fusion
import EvaluatePrecision

from KMeansSimilarityOrderer import KMeansSimilarityOrderer
from MajorityOrder import MajorityOrder
from MajorityCluster import MajorityCluster

from Evaluator import Evaluator
import pyrouge
from pyrouge import Rouge155

import logging
from logging import config

from pymetamap import MetaMap
from singletonConceptId import *

#import question_classifier

'''
@Author: Khyathi Raghavi Chandu
@Date: October 17 2017

This code has the entire pipeline built from the classes to execute bioasq ideal answer generation.
Running this code results in a json file that can be directly uploaded on the oracle to get the official ROUGE scores.
Running the code:
$> python pipeline.py ./input/phaseB_4b_04.json > submission.json
'''

#logging.config.fileConfig('logging.ini')

'''
# create logger with 'spam_application'
logger = logging.getLogger('bioasq')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('bioAsq.log')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

logging.basicConfig(
    level = logging.DEBUG,
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename = 'bioAsq.log',
    filemode = 'w'
)
'''

#logging.config.fileConfig('logging.ini')
logging.config.fileConfig('logging.ini')
logger = logging.getLogger('bioAsqLogger')


class Pipeline(object):
    def __init__(self, filePath, expanderInstance, biRankerInstance, orderInstance, fusionInstance,
                 tilerInstance, spanSelectorInstance):
        self.filePath = filePath
        self.expanderInstance = expanderInstance
        self.biRankerInstance = biRankerInstance
        self.orderInstance = orderInstance
        self.fusionInstance = fusionInstance
        self.tilerInstance = tilerInstance
        self.spanSelectorInstance =  spanSelectorInstance

    def getSummaries(self):

        metamapInstance = SingletonMetaMap.Instance()
        metamapInstance.startMetaMap()
        #raw_input()

        allAnswerQuestion = []
        infile = open(self.filePath, 'r')
        data = json.load(infile)
        logger.info('Loaded training data')
        #qc = question_classifier.classifier()

        for (i, question) in enumerate(data['questions']): # looping over all questions

            logger.info('Started summarization pipeline for Question '+ str(i))

            ExpansiontoOriginal = {}
            SentencetoSnippet = {}

            #pred_cat = qc.classify(question['body'])
            logger.info('Generated question classification for the question')
            pred_length = 30
            pred_cat = question['type']
            if pred_cat=='summary':
                pred_length = 7
            elif pred_cat=='list':
                pred_length = 5
            elif pred_cat=='factoid':
                pred_length = 3
            elif pred_cat=='yesno':
                pred_length = 4
            else:
                pass
            if pred_cat != 'factoid':
                continue


            modifiedQuestion = copy.copy(question)

            logger.info('Performing expansions...')
            

            #EXECUTIONS OF EXPANSIONS
            #expansion on question body i.e, the text in the question
            expandedQuestion = self.expanderInstance.getExpansions(question['body'])

            #expansion on every sentence in each of the snippets
            expandedSnippets = []
            for snippet in question['snippets']:
                expandedSnippet = snippet
                expandedSentences = ""
                for sentence in sent_tokenize(snippet['text']):
                    expandedSentence = self.expanderInstance.getExpansions(sentence)
                    expandedSentences += expandedSentence + " "
                    ExpansiontoOriginal[expandedSentence.strip()] = sentence.strip()
                    SentencetoSnippet[sentence.strip()] = snippet
                expandedSnippet['text'] = expandedSentences
                expandedSnippets.append(expandedSnippet)

            modifiedQuestion['snippets'] = expandedSnippets
            modifiedQuestion['body'] = expandedQuestion
            logger.info('Updated the question with expander output...')


            #EXECUTION OF ONE OF BIRANKERS
            rankedSentencesList = self.biRankerInstance.getRankedList(question)
            logger.info('Retrieved ranked list of sentences...')


            rankedSentencesListOriginal = []
            rankedSnippets = []
            print question['body']
            for sentence in rankedSentencesList:
                try:
                    rankedSentencesListOriginal.append(ExpansiontoOriginal[sentence.strip()])
                    rankedSnippets.append(SentencetoSnippet[sentence.strip()])
                except:
                    pass

            #EXECUTION OF TILING
            tiler_info = {'max_length': 200, 'max_tokens': 200, 'k': 2, 'max_iter': 20}
            if len(rankedSentencesList) == 0 or len(rankedSentencesListOriginal) == 0:
                continue
            orderedList = self.orderInstance.orderSentences(rankedSentencesListOriginal, rankedSnippets, tiler_info)
            fusedList = rankedSentencesList 
            logger.info('Tiling sentences to get alternative summary...')
            
            #EXECUTION OF EVAULATION (To be done)
            #evaluatorInstance = Evaluator()
            #goldIdealAnswer, r2, rsu = evaluatorInstance.calculateRouge(question['body'], finalSummary)

            #uncomment the following 3 lines for fusion
            #concat_inst = Concatenation()
            #finalSummary = concat_inst.tileSentences(fusedList, 200) #pred_length*5

            #question['ideal_answer'] = finalSummary
            #print (finalSummary)
            
            candidateSentences = rankedSentencesList[:1]
            exact_answer, exact_answer_prob = self.spanSelectorInstance.predict(question['body'], candidateSentences)
            if pred_cat != 'summary':
                if pred_cat in ['list', 'factoid']:
                    question['exact_answer'] = [exact_answer]
                else:
                    question['exact_answer'] = u'yes'
            print ('rankedSentencesList:', candidateSentences)
            print ('answer:', exact_answer)

            AnswerQuestion = question
            allAnswerQuestion.append(AnswerQuestion)
            logger.info('Inserted ideal answer into the json dictionary')
        metamapInstance.stopMetaMap()
        return allAnswerQuestion

if __name__ == '__main__':
    filePath = sys.argv[1]
    expanderInstance = NoExpander()
    biRankerInstance = CoreMMR()
    #m = dy.ParameterCollection()
    #biRankerInstance = MaLSTMScorer(m, "/home/ubuntu/model_dropout_corrected20", "/home/ubuntu/model_vocab.txt")
    orderInstance = MajorityCluster()
    fusionInstance = Fusion()
    tilerInstance = Concatenation()
    spanSelectorInstance = SpanSelector()
    pipelineInstance = Pipeline(filePath, expanderInstance, biRankerInstance, orderInstance, fusionInstance ,tilerInstance,
                                spanSelectorInstance)
    idealAnswerJson = {}
    idealAnswerJson['questions'] = pipelineInstance.getSummaries()
    with open('ordered_MaLSTM_answer_selection.json', 'w') as outfile:
        json.dump(idealAnswerJson, outfile)
    #print json.dumps(idealAnswerJson)
