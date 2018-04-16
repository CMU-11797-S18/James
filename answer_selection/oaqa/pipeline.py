from flask import Flask, request, abort, render_template
from flask import jsonify, render_template
from utils import predict
import pickle
import torch
import sys
import json
import copy
from nltk.tokenize import sent_tokenize, word_tokenize

from Expander import Expander
from NoExpander import NoExpander
from SnomedctExpander import SnomedctExpander
from UMLSExpander import UMLSExpander

from BiRanker import BiRanker
from CoreMMR import CoreMMR
from SoftMMR import SoftMMR
from HardMMR import HardMMR

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


logging.config.fileConfig('logging.ini')
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('bioAsqLogger')


class Pipeline(object):
    def __init__(self, filePath, expanderInstance, biRankerInstance, 
            orderInstance, fusionInstance, tilerInstance, answerExtractor,
            word_dict, char_dict):
        self.filePath = filePath
        self.expanderInstance = expanderInstance
        self.biRankerInstance = biRankerInstance
        self.orderInstance = orderInstance
        self.fusionInstance = fusionInstance
        self.tilerInstance = tilerInstance
        self.answerExtractor = answerExtractor
        self.word_dict = word_dict
        self.char_dict = char_dict

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

            if pred_cat!='factoid':
                continue

            modifiedQuestion = copy.copy(question)

            logger.info('Performing expansions...')
            

            #EXECUTIONS OF EXPANSIONS
            #expansion on question body i.e, the text in the question
            #expandedQuestion = self.expanderInstance.getExpansions(question['body'])
            if len(question['snippets']) == 0:
                continue

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

            logger.info('Updated the question with expander output...')


            #EXECUTION OF ONE OF BIRANKERS
            #rankedSentencesList = self.biRankerInstance.getRankedList(modifiedQuestion)
            rankedSentencesList = self.biRankerInstance.getRankedList(question)
            logger.info('Retrieved ranked list of sentences...')


            #ExpansiontoOriginal = {value: key for key, value in OriginaltoExpansion.iteritems()}
            rankedSentencesListOriginal = []
            rankedSnippets = []
            for sentence in rankedSentencesList:
                try:
                    rankedSentencesListOriginal.append(ExpansiontoOriginal[sentence.strip()])
                    rankedSnippets.append(SentencetoSnippet[sentence.strip()])
                except:
                    pass

            #EXECUTION OF TILING
            tiler_info = {'max_length': 200, 'max_tokens': 200, 'k': 2, 'max_iter': 20}
            orderedList = self.orderInstance.orderSentences(rankedSentencesListOriginal, rankedSnippets, tiler_info)
            best_answer, best_answer_prob = predict(model, question['body'], [orderedList[0]], word_dict, char_dict)

            #fusedList = self.fusionInstance.tileSentences(orderedList, 200)
            #print ('fusedList:')
            #print (fusedList)
            #logger.info('Tiling sentences to get alternative summary...')
            
            #uncomment the following 3 lines for fusion
            #concat_inst = Concatenation()
            #finalSummary = concat_inst.tileSentences(fusedList, 200) #pred_length*5
            #print ('finalSummary:')
            #print (finalSummary)
            #logger.info('Choosing better summary ...')

            question['exact_answer'] = best_answer 

            AnswerQuestion = question
            allAnswerQuestion.append(AnswerQuestion)
            logger.info('Inserted ideal answer into the json dictionary')
            print ('Adding ideal answer to json dictionary')
        metamapInstance.stopMetaMap()
        return allAnswerQuestion

if __name__ == '__main__':
    filePath = sys.argv[1]
    word_dict = pickle.load(sys.argv[2])
    char_dict = pickle.load(sys.argv[3])
    model = torch.load(sys.argv[4])
    #filePath = "../input/BioASQ-trainingDataset5b.json"
    expanderInstance = NoExpander()
    biRankerInstance = CoreMMR()
    orderInstance = MajorityCluster()
    fusionInstance = Fusion()
    tilerInstance = Concatenation()
    #tilerInstance = MajorityOrder()
    #tilerInstance = KMeansSimilarityOrderer()
    pipelineInstance = Pipeline(filePath, expanderInstance, biRankerInstance, orderInstance, fusionInstance ,tilerInstance, model, word_dict, char_dict)
    #pipelineInstance = Pipeline(filePath)
    idealAnswerJson = {}
    idealAnswerJson['questions'] = pipelineInstance.getSummaries()
    with open('ordered_fusion.json', 'w') as outfile:
        json.dump(idealAnswerJson, outfile)
    #print json.dumps(idealAnswerJson)
