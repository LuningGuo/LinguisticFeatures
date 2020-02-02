"""
---------------------------
QuitaText & getQuitaFeature
---------------------------

Calculate 22 linguistic features listed by software "Quantitative Index Text
Analyzer (QUITA)." Supported quantitative linguistic features include:

Frequency Structure indicators
    - Type-Token Ratio (TTR)
    - h-point (h)
    - Vocabulary Richness (R1)
    - Repeat Rate (RR)
    - Relative Repeat Rate of McIntosh (RRmc)
    - Hapax Legomenon Percentage (HL)
    - Lambda (Λ)
    - Gini Coefficient (G)
    - Vocabulary Richness (R4)
    - Curve length (L)
    - Curve length Indicator (R)
    - Entropy (H)
    - Adjusted Modulus (A)

Miscellaneous indicators
    - Verb Distances (VD)
    - Activity (Q)
    - Descriptivity (D)
    - Writer’s View (α)
    - Average Tokens length (ATL)
    - Thematic Concentration (TC)
    - Secondary Thematic Concentration (STC)
"""


# Package import
import re
import pandas as pd
import numpy as np
from nltk import pos_tag
from nltk import regexp_tokenize
from nltk import FreqDist


# QuitaText class
class QuitaText(object):
    """
    a text suitable for quantitative linguistic analysis, with various
    methods to calculate linguistic features listed by Quantitative Index
    Text Analyzer (QUITA) and attributes of type/token data.
    """

    def __init__(self, rawText):
        self.text = self.cleanText(rawText)  # clean text
        self.tokenList = regexp_tokenize(self.text, '\w+')  # tokenize
        self.tokenNum = len(self.tokenList)  # calculate token number
        self.typeData = self.getTypeData(self.tokenList)
        self.typeNum = len(self.typeData)

    @staticmethod
    def cleanText(text):
        text = str(text)  # convert into string
        text = text.lower()  # convert to lower case
        text = re.sub("'m|'s|n't", '', text)  # remove some suffix
        text = re.sub('[0-9]', '', text)  # remove numbers
        text = re.sub(r'\[.+?\]', ' ', text)  # remove bracket
        return text

    @staticmethod
    def isFunctionWord(word):
        functionWordsPOS = ['DT', 'CD', 'CC', 'UH', 'EX', 'MD', 'PP', 'PP$',
                            'WP', 'WP$', 'PDT', 'WDT', 'IN', 'TO', 'WRB']
        return pos_tag([word])[0][1] in functionWordsPOS

    @staticmethod
    def isVerb(word):
        verbPOS = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
        return pos_tag([word])[0][1] in verbPOS

    @staticmethod
    def isAdjective(word):
        adjPOS = ['JJ', 'JJR', 'JJS']
        return pos_tag([word])[0][1] in adjPOS

    @staticmethod
    def getTypeData(tokenList):
        typeData = pd.DataFrame()
        freqDist = FreqDist(tokenList)
        typeNum = len(freqDist)
        typeData['type'] = FreqDist(tokenList).keys()
        typeData['freq'] = [freqDist[i] for i in typeData['type']]
        tokenNum = np.sum(typeData['freq'])
        typeData = typeData.sort_values('freq', ascending=False)
        typeData.index = range(typeNum)
        typeData['prob'] = [freq / tokenNum for freq in typeData['freq']]
        typeData['rank'] = list(range(1, typeNum + 1))
        typeData['cumProb'] = [np.sum(typeData['prob'][:i]) for i in
                               range(1, typeNum + 1)]
        typeData['pos'] = [i[1] for i in pos_tag(typeData['type'])]
        return typeData

    def isExactHPoint(self):
        """whether there is an exact h-point"""
        freq = np.array(self.typeData['freq'])
        rank = np.array(self.typeData['rank'])
        return np.sum(freq == rank) > 0

    def getTTR(self):
        """calculate type-token ratio (TTR)"""
        ttr = self.typeNum / self.tokenNum
        return ttr

    def getHPoint(self):
        """calculate h-point (h)"""
        freq = np.array(self.typeData['freq'])
        rank = np.array(self.typeData['rank'])
        if self.isExactHPoint():
            HPoint = np.where(freq == rank)[0][0] + 1
            return HPoint
        else:
            r1 = np.sum(freq > rank)
            r2 = np.sum(freq > rank) + 1
            f1 = freq[np.where(rank == r1)][0]
            f2 = freq[np.where(rank == r2)][0]
            HPoint = (f1 * r2 - f2 * r1) / (r2 - r1 + f1 - f2)
            return HPoint

    def getEntropy(self):
        """calculate entropy (H)"""
        prob = np.array(self.typeData['prob'])
        entropy = np.sum(prob * np.log2(prob))
        return entropy

    def getATL(self):
        """calculate average token length (ATL)"""
        avgTokenLen = np.average([len(i) for i in self.tokenList])
        return avgTokenLen

    def getVocabRich(self):
        """calculate vocabulary richness (R4)"""
        cumProb = np.array(self.typeData['cumProb'])
        rank = np.array(self.typeData['rank'])
        if self.isExactHPoint():
            h = self.getHPoint()
            hCumProb = cumProb[np.where(rank == h)]
            richness = 1 - (hCumProb - (h ** 2) / (2 * self.tokenNum))
            return richness[0]
        else:
            h = self.getHPoint()
            hLeft = int(h)
            hRight = int(h) + 1
            hCumProbLeft = cumProb[np.where(rank == hLeft)]
            hCumProbRight = cumProb[np.where(rank == hRight)]
            hCumProb = np.average([hCumProbLeft, hCumProbRight])
            richness = 1 - (hCumProb - (h ** 2) / (2 * self.tokenNum))
            return richness

    def getRR(self):
        """calculate repeat rate (RR)"""
        prob = np.array(self.typeData['prob'])
        repeatRate = np.sum(np.square(prob))
        return repeatRate

    def getRRmc(self):
        """calculate Relative Repeat Rate of McIntosh (RRmc)"""
        RR = self.getRR()
        relativeRR = (1 - np.sqrt(RR)) / (1 - 1 / np.sqrt(self.typeNum))
        return relativeRR

    def getTC(self):
        """calculate Thematic Concentration (TC)"""
        h = self.getHPoint()
        TCList = list()
        for i in range(int(h) - 1):
            if not self.isFunctionWord(self.typeData['type'][i]):
                f1 = self.typeData['freq'][0]
                freq = self.typeData['freq'][i]
                rank = self.typeData['rank'][i]
                TC = 2 * ((h - rank) * freq) / (h * (h - 1) * f1)
                TCList.append(TC)
        return np.sum(np.array(TCList))

    def getSTC(self):
        """calculate Secondary Thematic Concentration (STC)"""
        h = self.getHPoint()
        TCList = list()
        for i in range(int(h) - 1, 2 * int(h) - 1):
            if not self.isFunctionWord(self.typeData['type'][i]):
                f1 = self.typeData['freq'][0]
                freq = self.typeData['freq'][i]
                rank = self.typeData['rank'][i]
                TC = 2 * ((h - rank) * freq) / (h * (h - 1) * f1)
                TCList.append(TC)
        return np.sum(np.array(TCList))

    def getActivity(self):
        """calculate Activity (Q)"""
        verbNum = sum([self.isVerb(word) for word in self.tokenList])
        adjNum = sum([self.isAdjective(word) for word in self.tokenList])
        return verbNum / (verbNum + adjNum)

    def getDescriptivity(self):
        """calculate Descriptivity (D)"""
        return 1 - self.getActivity()

    def getCurveLen(self, typeNum=None):
        """calculate Curve Length (CL)"""
        euclidLength = 0
        if typeNum is None:
            typeNum = self.typeNum
        for i in range(typeNum - 1):
            freq1 = self.typeData['freq'][i]
            freq2 = self.typeData['freq'][i + 1]
            euclidLength = euclidLength + ((freq1 - freq2) ** 2 + 1) ** 0.5
        return euclidLength

    def getCLI(self):
        """calculate Curve length Indicator (R)"""
        h = self.getHPoint()
        euclidLengthBeforeH = self.getCurveLen(int(h) - 1)
        euclidLengthAll = self.getCurveLen()
        return 1 - euclidLengthBeforeH / euclidLengthAll

    def getLambda(self):
        """calculate Lambda"""
        euclidLength = self.getCurveLen()
        return (euclidLength * np.log10(self.tokenNum)) / self.tokenNum

    def getAdjMod(self):
        """calculate Adjusted Modulus (A)"""
        f1 = self.typeData['freq'][0]
        h = self.getHPoint()
        V = self.typeNum
        N = self.tokenNum
        M = ((f1 / h) ** 2 + (V / h) ** 2) ** 0.5
        return M / np.log10(N)

    def getGiniCoef(self):
        """calculate Gini Coefficient (G)"""
        freq = np.array(self.typeData['freq'])
        rank = np.array(self.typeData['rank'])
        V = self.typeNum
        N = self.tokenNum
        giniCoef = (V + 1 - 2 * np.sum(freq * rank) / N) / V
        return giniCoef

    def getHL(self):
        """calculate Hapax Legomenon Percentage (HL)"""
        hapaxFreq = sum(self.typeData['freq'] == 1)
        return hapaxFreq / self.tokenNum

    def getAlpha(self):
        """calculate Writer’s View (α)"""
        f1 = self.typeData['freq'][0]
        h = self.getHPoint()
        V = self.typeNum
        up = (1 - h) * (f1 + V - 2 * h)
        down = ((h - 1) ** 2 + (f1 - h) ** 2) ** 0.5 * \
               ((h - 1) ** 2 + (V - h) ** 2) ** 0.5
        return up / down

    def getVerbDist(self):
        """calculate Verb Distances (VD)"""
        verbIndex = np.where([self.isVerb(word) for word in self.tokenList])[0]
        verbDist = list()
        for i in range(len(verbIndex) - 1):
            dist = verbIndex[i + 1] - verbIndex[i]
            verbDist.append(dist)
        return np.average(verbDist)


def getQuitaFeature(text):
    """calculate all linguistic features"""
    features = dict()
    quitaText = QuitaText(rawText=text)
    features['TTR'] = quitaText.getTTR()
    features['R'] = quitaText.getHPoint()
    features['ATL'] = quitaText.getATL()
    features['R4'] = quitaText.getVocabRich()
    features['RR'] = quitaText.getRR()
    features['RRmc'] = quitaText.getRRmc()
    features['TC'] = quitaText.getTC()
    features['STC'] = quitaText.getSTC()
    features['Q'] = quitaText.getActivity()
    features['D'] = quitaText.getDescriptivity()
    features['CL'] = quitaText.getCurveLen()
    features['R'] = quitaText.getCLI()
    features['Lambda'] = quitaText.getLambda()
    features['A'] = quitaText.getAdjMod()
    features['G'] = quitaText.getGiniCoef()
    features['HL'] = quitaText.getHL()
    features['Alpha'] = quitaText.getAlpha()
    features['VD'] = quitaText.getVerbDist()
    return features
