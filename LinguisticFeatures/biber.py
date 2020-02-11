# coding=utf-8
"""
Biber Features
--------------

Calculate 67 linguistic features listed by Douglas Biber in Variation across
Speech and Writing (1995)

feature list
------------

(A) TENSE AND ASPECT MARKERS
    A01: past tense
    A02: perfect aspect
    A03: present tense

(B) PLACE AND TIME ADVERBIALS
    B04: place adverbials
    B05: time adverbials

(C) PRONOUNS AND PRO-VERBS
    (C1) PERSONAL PRONOUNS
        C06: first person pronouns
        C07: second person pronouns
        C08: third person personal pronouns
    (C2) IMPERSONAL PRONOUNS
        C09: pronoun "it"
        C10: demonstrative pronouns
        C11: indefinite pronouns
    (C3) PRO-VERBS
        C12: pro-verb do

(D) QUESTIONS
    D13: direct WH-questions

(E) NOMINAL FORMS
    E14: nominalizations
    E15: gerunds
    E16: total other nouns

(F) PASSIVES
    F17: agentless passives
    F18: by-passives

(G) STATIVE FORMS
    G19: "be" as main verb
    G20: existential there

(H) SUBORDINATION
    (H1) COMPLEMENTATION
        H21: "that" verb complements
        H22: "that" adjective complements
        H23: WH-clauses
        H24: infinitives
    (H2) PARTICIPIAL FORMS
        H25: present participial clauses
        H26: past participial clauses
        H27: past participial WHIZ deletion relatives
        H28: present participial WHIZ deletion relatives
    (H3) RELATIVES
        H29: that relative clauses on subject position
        H30: that relative clauses on object position
        H31: WH relative clauses on subject position
        H32: WH relative clauses on object positions
        H33: pied-piping relative clauses
        H34: sentence relatives
    (H4) ADVERBIAL CLAUSES
        H35: causative adverbial subordinators: because
        H36: concessive adverbial subordinators: although, though
        H37: conditional adverbial subordinators: if, unless
        H38: other adverbial subordinators: (having multiple functions)

(I) PREPOSITIONAL PHRASES
    (I1) PREPOSITIONAL PHRASES
        I39: total prepositional phrases
    (I2) ADJECTIVES AND ADVERBS
        I40: attributive adjectives
        I41: predicative adjectives
        I42: total adverbs

(J) LEXICAL SPECIFICITY
    J43: type/token ratio
    J44: word length

(K) LEXICAL CLASSES
    K45: conjuncts
    K46: downtoners
    K47: hedges
    K48: amplifiers
    K49: emphatics
    K50: discourse particles
    K51: demonstratives

(L) MODALS
    L52: possibility modals
    L53: necessity modals
    L54: predictive modals

(M) SPECIALIZED VERB CLASSES
    M55: public verbs
    M56: private verbs
    M57: suasive verbs
    M58: seem/appear

(N) REDUCED FORMS AND DISPREFERRED STRUCTURES
    N59: contractions
    N60: subordinator-that deletion
    N61: stranded prepositions
    N62: split infinitives
    N63: split auxiliaries

(O) COORDINATION
    O64: phrasal coordination
    O65: independent clause coordination

(P) NEGATION
    P66: synthetic negation
    P67: analytic negation: not

special notes
-------------

As Biber admitted in his book, E15: gerunds and E16: total other nouns are
calculated by hand and does NOT have a generalized algorithm。 Therefore here
we do not provide methods for the two.
"""

# Package import
import re

import numpy as np
from nltk import FreqDist
from nltk import pos_tag
from nltk import word_tokenize, regexp_tokenize


# functions to combine regex together
def OR(patternList):
    """
    get the regex of the "OR" of all patterns in the list
    parameter:
        patternList: list(str, ...), list of patterns
    return:
        str, the regex of the "OR" of all patterns
    """
    pattern = '('
    for i in range(len(patternList) - 1):
        pattern = pattern + patternList[i] + '|'
    return pattern + patternList[-1] + ')'


def REPEAT(pattern, time_range):
    """
    get the regex of the given pattern's repetition
    parameter:
        pattern: str, regular expression pattern to be repeated
        time_range: tuple(int, int), repeat time range
    return:
        str, regex of the given pattern's repetition
    """
    time_range = str(time_range[0]) + ',' + str(time_range[1])
    return pattern + '{' + time_range + '}'


# Regular expression patterns for some grammatical categories
XXX = "( \w+_[A-Z]+)"
DO = "( (do|does|did|doing|done)_[A-Z]+)"
HAVE = "( (have|has|had|having|'ve|'d)_[A-Z]+)"
BE = "( (am|is|are|was|were|being|been|'m|'re)_[A-Z]+)"
MODAL = "( (can|may|shall|will|'ll|could|might|should|would|must)_[A-Z]+)"
AUX = OR([DO, HAVE, BE, MODAL, "( 's_[A-Z]+)"])
SUBJPRO = "( (I|we|he|she|they)_[A-Z]+)"
OBJPRO = "( (me|us|him|them)_[A-Z]+)"
POSSPRO = "( (my|our|your|his|their|its)_[A-Z]+)"
REFLEXPRO = "( (myself|ourselves|himself|themselves|herself|yourself" \
            "|yourselves|itself)_[A-Z]+)"
PRO = OR([SUBJPRO, OBJPRO, POSSPRO, REFLEXPRO, "( (you|her|it)_[A-Z]+)"])
PREP = "( (against|amid|amidst|among|amongst|at|besides|between|by|despite" \
       "|during|except|for|from|in|into|minus|notwithstanding|of|off|on|onto" \
       "|opposite|out|per|plus|pro|re|than|through|throughout|thru|to|toward" \
       "|towards|upon|versus|via|with|within|without)_[A-Z]+)"
ADV = "( \w+_(RB|RBR|RBS))"
ADJ = "( \w+_(JJ|JJR|JJS))"
N = "( \w+_(NN|NNS|NNP|NNPS))"
VBN = "( \w+_VBN)"
VBG = "( \w+_VBG)"
VB = "( \w+_VB)"
VBZ = "( \w+_VBZ)"
PUB = "( (acknowledge|admit|agree|assert|claim|complain|declare|deny|explain" \
      "|hint|insist|mention|proclaim|promise|protest|remark|reply|report|say" \
      "|suggest|swear|write)_[A-Z]+)"
PRV = "( (anticipate|assume|believe|conclude|decide|demonstrate|determine" \
      "|discover|doubt|estimate|fear|feel|find|forget|guess|hear|hope|imagine" \
      "|imply|indicate|infer|know|learn|mean|notice|prove|realize|recognize" \
      "|remember|reveal|see|show|suppose|think|understand)_[A-Z]+) "
SUA = "( (agree|arrange|ask|beg|command|decide|demand|grant|insist|instruct" \
      "|ordain|pledge|pronounce|propose|recommend|request|stipulate|suggest" \
      "|urge)_[A-Z]+)"
V = "( \w+_(VB|VBD|VBG|VBN|VBP|VBZ))"
WHP = "( (who|whom|whose|which)_[A-Z]+)"
WHO = "( (what|where|when|how|whether|why|whoever|whomever|whichever|wherever" \
      "|whenever|whatever|however)_[A-Z]+)"
ART = "( (a|an|the)_[A-Z]+)"
DEM = "( (this|that|these|those)_[A-Z]+)"
QUAN = "( (each|all|every|many|much|few|several|some|any)_[A-Z]+)"
NUM = "( (one|two|three|four|five|six|seven|eight|nine|ten|twelve|thirteen" \
      "|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|hundred" \
      "|thousand)_[A-Z]+)"
DET = OR([ART, DEM, QUAN, NUM])
ORD = "( (first|second|third|fourth|fifth|sixth|seventh|eighth|ninth" \
      "|tenth)_[A-Z]+)"
QUANPRO = "( (everybody|somebody|anybody|everyone|someone|anyone|everything" \
          "|something|anything)_[A-Z]+)"
TITLE = "( (mr|ms|miss|mrs|dr)_[A-Z]+)"
CL_P = "( \._\.|\!_\!|\?_\?|\:_\:|\;_\;|\-_\-)"
ALL_P = OR([CL_P, "( \,_\,)"])
SEEM = "( seem_[A-Z])"
APPEAR = "( appear+_[A-Z])"
DEMOPRO = OR(['( (that|this|these|those)_[A-Z]+)' +
              OR([V, AUX, CL_P, WHP, "( and_[A-Z]+)"]),
              "( that_[A-Z]+ 's_[A-Z]+)"])


def getCONJ():
    pattern_1 = "(alternatively|altogether|consequently|conversely" \
                "|else|furthermore|hence|however|instead|likewise" \
                "|moreover|namely|nevertheless|nonetheless" \
                "|notwithstanding|otherwise|rather|similarly|therefore" \
                "|thus|viz) "
    pattern_2 = "( in_[A-Z]+)" + "( (comparison|contrast|particular " \
                                 "|addition|conclusion|consequence|sum" \
                                 "|summary)_[A-Z]+) "
    pattern_3 = "( in_[A-Z]+)" + "( (any_[A-Z]+ event_[A-Z]+|any_[A-Z]+ " \
                                 "case_[A-Z]+|other_[A-Z]+ words_[A-Z]+)) "
    pattern_4 = "( for_[A-Z]+)" + "( (example|instance)_[A-Z]+)"
    pattern_5 = "( by_[A-Z]+)" + "( (contrast|comparison)_[A-Z]+)"
    pattern_6 = "( as_[A-Z]+ a_[A-Z]+)" + "( (result|consequence)_[A-Z]+)"
    pattern_7 = "( on_[A-Z]+ the_[A-Z]+ contrary_[A-Z]+" \
                "|on_[A-Z]+ the_[A-Z]+ other_[A-Z]+ hand_[A-Z]+)"
    pattern_8 = ALL_P + "( that_[A-Z]+ is_[A-Z]+| else_[A-Z]+" \
                        "| altogether_[A-Z]+)"
    pattern = OR([pattern_1, pattern_2, pattern_3, pattern_4,
                  pattern_5, pattern_6, pattern_7, pattern_8])
    return pattern


# BiberText class
class BiberText(object):
    """
    a text suitable for quantitative linguistic analysis, with various
    methods to calculate linguistic features listed by Douglas Biber
    and attributes of type/token data.
    """

    def __init__(self, rawText):
        self.rawText = rawText
        self.taggedText = self.posTag()
        self.tokenList = word_tokenize(self.rawText)
        self.tagList = [i[1] for i in pos_tag(self.tokenList)]
        self.typeList = FreqDist(self.tokenList).keys()
        self.tokenNum = len(self.tokenList)
        self.typeNum = len(self.typeList)

    def posTag(self):
        tagList = pos_tag(word_tokenize(self.rawText))
        resultList = [i[0].lower() + '_' + i[1] for i in tagList]
        taggedText = ' '.join(resultList)
        return taggedText

    def feature_01(self):
        """A01: past tense"""
        num = sum([pos == 'VBD' for pos in self.tagList])
        return 1000 * num / self.tokenNum

    def feature_02(self):
        """A02: perfect aspect"""
        pattern_a = HAVE + REPEAT(ADV, (0, 2)) + VBN
        pattern_b = HAVE + OR([N, PRO]) + VBN
        pattern = OR([pattern_a, pattern_b])
        num = len(re.findall(pattern, self.taggedText))
        return 1000 * num / self.tokenNum

    def feature_03(self):
        """A03: present tense"""
        num = sum([pos in ['VBP', 'VBZ'] for pos in self.tagList])
        return 1000 * num / self.tokenNum

    def feature_04(self):
        """B04: place adverbials"""
        pattern = '( (aboard|above|abroad|across|ahead|alongside|around' \
                  '|ashore|astern|away|behind|below|beneath|beside|downhill' \
                  '|downstairs|downstream|east|far|hereabouts|indoors|inland' \
                  '|inshore|inside|locally|near|nearby|north|nowhere|outdoors' \
                  '|outside|overboard|overland|overseas|south|underfoot' \
                  '|underground|underneath|uphill|upstairs|upstream|west)' \
                  '_[A-Z]+)'
        num = len(re.findall(pattern, self.taggedText))
        return 1000 * num / self.tokenNum

    def feature_05(self):
        """B05: time adverbials"""
        pattern = '( (afterwards|again|earlier|early|eventually|formerly' \
                  '|immediately|initially|instantly|late|lately|later' \
                  '|momentarily|now|nowadays|once|originally|presently' \
                  '|previously|recently|shortly|simultaneously|soon' \
                  '|subsequently|today|tomorrow|tonight|yesterday)' \
                  '_[A-Z]+)'
        num = len(re.findall(pattern, self.taggedText))
        return 1000 * num / self.tokenNum

    def feature_06(self):
        """C06: first person pronouns"""
        pattern = '( (I|me|we|us|my|our|myself|ourselves)_[A-Z]+)'
        num = len(re.findall(pattern, self.taggedText))
        return 1000 * num / self.tokenNum

    def feature_07(self):
        """C07: second person pronouns"""
        pattern = '( (you|your|yourself|yourselves)_[A-Z]+)'
        num = len(re.findall(pattern, self.taggedText))
        return 1000 * num / self.tokenNum

    def feature_08(self):
        """C08: third person personal pronouns"""
        pattern = '( (she|he|they|her|him|them|his|their|himself|herself' \
                  '|themselves)_[A-Z]+)'
        num = len(re.findall(pattern, self.taggedText))
        return 1000 * num / self.tokenNum

    def feature_09(self):
        """C09: pronoun it"""
        pattern = '( it_[A-Z]+)'
        num = len(re.findall(pattern, self.taggedText))
        return 1000 * num / self.tokenNum

    def feature_10(self):
        """C010: demonstrative pronouns"""
        pattern_a = '( (that|this|these|those)_[A-Z]+)' + \
                    OR([V, AUX, CL_P, WHP, "( and_[A-Z]+)"])
        pattern_b = "( that_[A-Z]+ 's_[A-Z]+)"
        pattern = OR([pattern_a, pattern_b])
        num = len(re.findall(pattern, self.taggedText))
        return 1000 * num / self.tokenNum

    def feature_11(self):
        """C11: indefinite pronouns"""
        pattern = "( (anybody|anyone|anything|everybody|everyone|everything" \
                  "|nobody|none|nothing|nowhere|somebody|someone|something)" \
                  "_[A-Z]+)"
        num = len(re.findall(pattern, self.taggedText))
        return 1000 * num / self.tokenNum

    def feature_12(self):
        """C12: pro-verb do"""
        pattern_a = DO + REPEAT(ADV, (0, 1)) + V
        pattern_b = OR([ALL_P, WHP]) + DO
        num_a = len(re.findall(pattern_a, self.taggedText))
        num_b = len(re.findall(pattern_b, self.taggedText))
        num_do = len(re.findall(DO, self.taggedText))
        return 1000 * (num_do - num_a - num_b) / self.tokenNum

    def feature_13(self):
        """D13: direct WH-questions"""
        pattern = CL_P + WHO + AUX
        num = len(re.findall(pattern, self.taggedText))
        return 1000 * num / self.tokenNum

    def feature_14(self):
        """E14: nominalizations"""
        pattern = '( \w+(tion|tions|ment|ments|ness|nesses|ity|ities)_[A-Z]+)'
        num = len(re.findall(pattern, self.taggedText))
        return 1000 * num / self.tokenNum

    def feature_17(self):
        """F17: agentless passives"""
        pattern_a = BE + REPEAT(ADV, (0, 2)) + VBN + '( by_[A-Z]+)'
        pattern_b = BE + OR([N, PRO]) + VBN + '( by_[A-Z]+)'
        withBy = OR([pattern_a, pattern_b])
        pattern_c = BE + REPEAT(ADV, (0, 2)) + VBN
        pattern_d = BE + OR([N, PRO]) + VBN
        allPassive = OR([pattern_c, pattern_d])
        num1 = len(re.findall(withBy, self.taggedText))
        num2 = len(re.findall(allPassive, self.taggedText))
        return 1000 * (num2 - num1) / self.tokenNum

    def feature_18(self):
        """F18: agentless passives"""
        pattern_a = BE + REPEAT(ADV, (0, 2)) + VBN + '( by_[A-Z]+)'
        pattern_b = BE + OR([N, PRO]) + VBN + '( by_[A-Z]+)'
        pattern = OR([pattern_a, pattern_b])
        num = len(re.findall(pattern, self.taggedText))
        return 1000 * num / self.tokenNum

    def feature_19(self):
        """G19: be as main verb"""
        pattern = BE + OR([DET, POSSPRO, TITLE, PREP, ADJ])
        num = len(re.findall(pattern, self.taggedText))
        return 1000 * num / self.tokenNum

    def feature_20(self):
        """G20: existential there"""
        pattern_a = '( there_[A-Z]+)' + REPEAT(XXX, (0, 1)) + BE
        pattern_b = '( there_[A-Z]+)' + "( 's_[A-Z]+)"
        pattern = OR([pattern_a, pattern_b])
        num = len(re.findall(pattern, self.taggedText))
        return 1000 * num / self.tokenNum

    def feature_21(self):
        """H21: that verb complements"""
        # pattern_a
        pattern_a = OR(['( (and|nor|but|or|also)_[A-Z]+)', ALL_P]) + \
                    '( that_[A-Z]+)' + OR(
            [DET, PRO, '( there_[A-Z]+)', N, TITLE])
        a_num = len(re.findall(pattern_a, self.taggedText))
        # pattern_b
        pattern_b_all = OR(
            [PUB, PRV, SUA, SEEM, APPEAR]) + "( that_[A-Z]+)" + XXX
        pattern_b_except = OR([PUB, PRV, SUA, SEEM, APPEAR]) + \
                           "( that_[A-Z])" + OR([V, AUX, CL_P, "( and_[A-Z]+)"])
        b_all_num = len(re.findall(pattern_b_all, self.taggedText))
        b_except_num = len(re.findall(pattern_b_except, self.taggedText))
        b_num = b_all_num - b_except_num
        # pattern_c
        pattern_c_all = OR(
            [PUB, PRV, SUA]) + PREP + XXX + '+' + N + "( that_[A-Z]+)"
        pattern_c_except = OR([PUB, PRV, SUA]) + PREP + N + N + "( that_[A-Z]+)"
        c_all_num = len(re.findall(pattern_c_all, self.taggedText))
        c_except_num = len(re.findall(pattern_c_except, self.taggedText))
        c_num = c_all_num - c_except_num
        return 1000 * (a_num + b_num + c_num) / self.tokenNum

    def feature_22(self):
        """H22: that adjective complements"""
        pattern = ADJ + "( that_[A-Z]+)"
        num = len(re.findall(pattern, self.taggedText))
        return 1000 * num / self.tokenNum

    def feature_23(self):
        """H23: WH-clauses"""
        pattern_all = OR([PUB, PRV, SUA]) + OR([WHP, WHO]) + XXX
        pattern_except = OR([PUB, PRV, SUA]) + OR([WHP, WHO]) + AUX
        num_all = len(re.findall(pattern_all, self.taggedText))
        num_except = len(re.findall(pattern_except, self.taggedText))
        return 1000 * (num_all - num_except) / self.tokenNum

    def feature_24(self):
        """H24: infinitives"""
        pattern = '( to_[A-Z]+)' + REPEAT(ADV, (0, 1)) + VB
        num = len(re.findall(pattern, self.taggedText))
        return 1000 * num / self.tokenNum

    def feature_25(self):
        """H25: present participial clauses"""
        pattern = ALL_P + VBG + OR([PREP, DET, WHP, WHO, PRO, ADV])
        num = len(re.findall(pattern, self.taggedText))
        return 1000 * num / self.tokenNum

    def feature_26(self):
        """H26: past participial clauses"""
        pattern = ALL_P + VBN + OR([PREP, ADV])
        num = len(re.findall(pattern, self.taggedText))
        return 1000 * num / self.tokenNum

    def feature_27(self):
        """H27: past participial WHIZ deletion relatives"""
        pattern = OR([N, QUANPRO]) + VBN + OR([PREP, BE, ADV])
        num = len(re.findall(pattern, self.taggedText))
        return 1000 * num / self.tokenNum

    def feature_28(self):
        """H28: present participial WHIZ deletion relatives"""
        pattern = N + VBG
        num = len(re.findall(pattern, self.taggedText))
        return 1000 * num / self.tokenNum

    def feature_29(self):
        """H29: that relative clauses on subject position"""
        pattern = N + "( that_[A-Z]+)" + REPEAT(ADV, (0, 1)) + OR([AUX, V])
        num = len(re.findall(pattern, self.taggedText))
        return 1000 * num / self.tokenNum

    def feature_30(self):
        """H30: that relative clauses on object position"""
        pattern = N + "( that_[A-Z]+)" + \
                  OR([DET, SUBJPRO, POSSPRO, "( it_[A-Z]+)", ADJ, N, TITLE])
        num = len(re.findall(pattern, self.taggedText))
        return 1000 * num / self.tokenNum

    def feature_31(self):
        """H31: WH relative clauses on subject position"""
        ASK = '( (ask|asked|asks)_[A-Z]+)'
        TELL = '( (tell|told|tells)_[A-Z]+)'
        pattern_all = XXX + XXX + N + WHP + REPEAT(ADV, (0, 1)) + OR([AUX, V])
        pattern_except = OR([ASK, TELL]) + XXX + N + WHP + REPEAT(ADV,
                                                                  (0, 1)) + OR(
            [AUX, V])
        num_all = len(re.findall(pattern_all, self.taggedText))
        num_except = len(re.findall(pattern_except, self.taggedText))
        return 1000 * (num_all - num_except) / self.tokenNum

    def feature_32(self):
        """H32: WH relative clauses on object positions"""
        ASK = '( (ask|asked|asks)_[A-Z]+)'
        TELL = '( (tell|told|tells)_[A-Z]+)'
        pattern_1 = XXX + XXX + N + WHP + XXX
        pattern_2 = XXX + OR([ASK, TELL]) + N + WHP + OR([ADV, AUX, V])
        pattern_3 = XXX + OR([ASK, TELL]) + N + WHP + XXX
        pattern_4 = XXX + XXX + N + WHP + OR([ADV, AUX, V])
        num_1 = len(re.findall(pattern_1, self.taggedText))
        num_2 = len(re.findall(pattern_2, self.taggedText))
        num_3 = len(re.findall(pattern_3, self.taggedText))
        num_4 = len(re.findall(pattern_4, self.taggedText))
        return 1000 * (num_1 + num_2 - num_4 - num_3) / self.tokenNum

    def feature_33(self):
        """H33: pied-piping relative clauses"""
        pattern = PREP + WHP
        num = len(re.findall(pattern, self.taggedText))
        return 1000 * num / self.tokenNum

    def feature_34(self):
        """H34: sentence relatives"""
        pattern = "( ,_, which_[A-Z]+)"
        num = len(re.findall(pattern, self.taggedText))
        return 1000 * num / self.tokenNum

    def feature_35(self):
        """H35: causative adverbial subordinators: because"""
        pattern = "( because_[A-Z]+)"
        num = len(re.findall(pattern, self.taggedText))
        return 1000 * num / self.tokenNum

    def feature_36(self):
        """H36: concessive adverbial subordinators: although, though"""
        pattern = "( (although|though)_[A-Z]+)"
        num = len(re.findall(pattern, self.taggedText))
        return 1000 * num / self.tokenNum

    def feature_37(self):
        """H37: conditional adverbial subordinators: if, unless"""
        pattern = "( (if|unless)_[A-Z]+)"
        num = len(re.findall(pattern, self.taggedText))
        return 1000 * num / self.tokenNum

    def feature_38(self):
        """H38: other adverbial subordinators: (having multiple functions)"""
        pattern = OR(
            ["( (since|while|whilst|whereupon|whereas|whereby)_[A-Z]+)",
             "( (such|so|such)_[A-Z]+ that_[A-Z]+)",
             "( (inasmuch|forasmuch|insofar|insomuch)_[A-Z]+ as_[A-Z]+)",
             "( as_[A-Z]+ (long|soon)_[A-Z]+ as_[A-Z]+)"])
        num = len(re.findall(pattern, self.taggedText))
        return 1000 * num / self.tokenNum

    def feature_39(self):
        """I39: total prepositional phrases"""
        pattern = PREP
        num = len(re.findall(pattern, self.taggedText))
        return 1000 * num / self.tokenNum

    def feature_40(self):
        """I40: attributive adjectives"""
        pattern = ADJ + OR([ADJ, N])
        num = len(re.findall(pattern, self.taggedText))
        return 1000 * num / self.tokenNum

    def feature_41(self):
        """I41: predicative adjectives"""
        # pattern_a
        pattern_a_all = BE + ADJ + XXX
        pattern_a_except = BE + ADJ + OR([ADJ, ADV, N])
        num_a_all = len(re.findall(pattern_a_all, self.taggedText))
        num_a_except = len(re.findall(pattern_a_except, self.taggedText))
        num_a = num_a_all - num_a_except
        # pattern_b
        pattern_b_all = BE + ADJ + ADV + XXX
        pattern_b_except = BE + ADJ + ADV + OR([ADJ, N])
        num_b_all = len(re.findall(pattern_b_all, self.taggedText))
        num_b_except = len(re.findall(pattern_b_except, self.taggedText))
        num_b = num_b_all - num_b_except
        return 1000 * (num_a + num_b) / self.tokenNum

    def feature_42(self):
        """I42: total adverbs"""
        pattern = ADV
        num = len(re.findall(pattern, self.taggedText))
        return 1000 * num / self.tokenNum

    def feature_43(self):
        """J43: type/token ratio"""
        return self.typeNum / self.tokenNum

    def feature_44(self):
        """J44: word length"""
        cleaned_text = re.sub('[0-9]', '', self.rawText)
        cleaned_text = re.sub("'s|'m|'t", '', cleaned_text)
        wordList = regexp_tokenize(cleaned_text, '\w+')
        return np.average([len(word) for word in wordList])

    def feature_45(self):
        """K45: conjuncts"""
        pattern_1 = "(alternatively|altogether|consequently|conversely" \
                    "|else|furthermore|hence|however|instead|likewise" \
                    "|moreover|namely|nevertheless|nonetheless" \
                    "|notwithstanding|otherwise|rather|similarly|therefore" \
                    "|thus|viz) "
        pattern_2 = "( in_[A-Z]+)" + "( (comparison|contrast|particular " \
                                     "|addition|conclusion|consequence|sum" \
                                     "|summary)_[A-Z]+) "
        pattern_3 = "( in_[A-Z]+)" + "( (any_[A-Z]+ event_[A-Z]+|any_[A-Z]+ " \
                                     "case_[A-Z]+|other_[A-Z]+ words_[A-Z]+)) "
        pattern_4 = "( for_[A-Z]+)" + "( (example|instance)_[A-Z]+)"
        pattern_5 = "( by_[A-Z]+)" + "( (contrast|comparison)_[A-Z]+)"
        pattern_6 = "( as_[A-Z]+ a_[A-Z]+)" + "( (result|consequence)_[A-Z]+)"
        pattern_7 = "( on_[A-Z]+ the_[A-Z]+ contrary_[A-Z]+" \
                    "|on_[A-Z]+ the_[A-Z]+ other_[A-Z]+ hand_[A-Z]+)"
        pattern_8 = ALL_P + "( that_[A-Z]+ is_[A-Z]+| else_[A-Z]+" \
                            "| altogether_[A-Z]+)"
        pattern = OR([pattern_1, pattern_2, pattern_3, pattern_4,
                      pattern_5, pattern_6, pattern_7, pattern_8])
        num = len(re.findall(pattern, self.taggedText))
        return 1000 * num / self.tokenNum

    def feature_46(self):
        """K46: downtoners"""
        pattern = "(almost|barely|hardly|merely|mildly|nearly|only|partially" \
                  "|partly|practically|scarcely|slightly|somewhat)_[A-Z]+ "
        num = len(re.findall(pattern, self.taggedText))
        return 1000 * num / self.tokenNum

    def feature_47(self):
        """K47: hedges"""
        # pattern_a
        pattern_a = "(at_[A-Z]+ about_[A-Z]+|something_[A-Z]+ like_[A-Z]+" \
                    "|more_[A-Z]+ or_[A-Z]+ less_[A-Z]+" \
                    "|almost_[A-Z]+|maybe_[A-Z]+|)"
        num_a = len(re.findall(pattern_a, self.taggedText))
        # pattern_b
        pattern_b_all = XXX + "( (sort|kind)_[A-Z]+ of_[A-Z]+)"
        pattern_b_except = OR([DET, ADJ, POSSPRO, WHO]) + \
                           "( (sort|kind)_[A-Z]+ of_[A-Z]+)"
        num_b_all = len(re.findall(pattern_b_all, self.taggedText))
        num_b_except = len(re.findall(pattern_b_except, self.taggedText))
        num_b = num_b_all - num_b_except
        return 1000 * (num_a + num_b) / self.tokenNum

    def feature_48(self):
        """K48: amplifiers"""
        pattern = "absolutely|altogether|completely|enormously|entirely" \
                  "|extremely|fully|greatly|highly|intensely|perfectly" \
                  "|strongly|thoroughly|totally|utterly|very"
        num = len(re.findall(pattern, self.taggedText))
        return 1000 * num / self.tokenNum

    def feature_49(self):
        """K49: emphatics"""
        pattern = "( for_[A-Z]+ sure_[A-Z]+| a_[A-Z]+ lot_[A-Z]+" \
                  "| such_[A-Z]+ a_[A-Z]+| real_[A-Z]+)" + \
                  OR([ADJ, "( so_[A-Z]+)"]) + OR([ADJ, DO]) + \
                  OR([V, "( (just|really|most|more)_[A-Z]+)"])
        num = len(re.findall(pattern, self.taggedText))
        return 1000 * num / self.tokenNum

    def feature_50(self):
        """K50: discourse particles"""
        pattern = CL_P + "( (well|now|anyway|anyhow|anyways)_[A-Z]+)"
        num = len(re.findall(pattern, self.taggedText))
        return 1000 * num / self.tokenNum

    def feature_51(self):
        """K51: demonstratives"""
        pattern = "( (that|this|these|those)_DT)"
        num = len(re.findall(pattern, self.taggedText))
        return 1000 * num / self.tokenNum

    def feature_52(self):
        """L52: possibility modals"""
        pattern = "( (can|may|might|could)_[A-Z]+)"
        num = len(re.findall(pattern, self.taggedText))
        return 1000 * num / self.tokenNum

    def feature_53(self):
        """L53: necessity modals"""
        pattern = "( (ought|should|must)_[A-Z]+)"
        num = len(re.findall(pattern, self.taggedText))
        return 1000 * num / self.tokenNum

    def feature_54(self):
        """L54: predictive modals"""
        pattern = "( (will|would|shall)_[A-Z]+)"
        num = len(re.findall(pattern, self.taggedText))
        return 1000 * num / self.tokenNum

    def feature_55(self):
        """M55: public verbs"""
        pattern = PUB
        num = len(re.findall(pattern, self.taggedText))
        return 1000 * num / self.tokenNum

    def feature_56(self):
        """M56: private verbs"""
        pattern = PRV
        num = len(re.findall(pattern, self.taggedText))
        return 1000 * num / self.tokenNum

    def feature_57(self):
        """M57: suasive verbs"""
        pattern = SUA
        num = len(re.findall(pattern, self.taggedText))
        return 1000 * num / self.tokenNum

    def feature_58(self):
        """M58: seem/appear"""
        pattern = "( (seem|appear)_[A-Z]+)"
        num = len(re.findall(pattern, self.taggedText))
        return 1000 * num / self.tokenNum

    def feature_59(self):
        """N59: contractions"""
        pattern_all = "( ('d|'ll|'m|'re|'ve|n't|'s)_[A-Z]+)"
        pattern_except = "('s_[A-Z]+)" + OR([V, AUX, ADV]) + OR([V, ADV]) + \
                         OR([AUX, DET, POSSPRO, PREP, ADJ]) + OR([CL_P, ADJ])
        num_all = len(re.findall(pattern_all, self.taggedText))
        num_except = len(re.findall(pattern_except, self.taggedText))
        return 1000 * (num_all - num_except) / self.tokenNum

    def feature_60(self):
        """N60: subordinator-that deletion"""

        pattern_1 = OR([PUB, PRV, SUA]) + OR([DEMOPRO, SUBJPRO])
        pattern_2 = OR([PUB, PRV, SUA]) + OR([PRO, N]) + OR([AUX, V])
        pattern_3 = OR([PUB, PRV, SUA]) + OR([ADJ, ADV, DET, POSSPRO]) + \
                    REPEAT(ADJ, (0, 1)) + N + OR([AUX, V])
        num_1 = len(re.findall(pattern_1, self.taggedText))
        num_2 = len(re.findall(pattern_2, self.taggedText))
        num_3 = len(re.findall(pattern_3, self.taggedText))
        return 1000 * (num_1 + num_2 + num_3) / self.tokenNum

    def feature_61(self):
        """N61: stranded prepositions"""
        pattern = PREP + ALL_P
        num = len(re.findall(pattern, self.taggedText))
        return 1000 * num / self.tokenNum

    def feature_62(self):
        """N62: split infinitives"""
        pattern = "( to_[A-Z]+)" + ADV + REPEAT(ADV, (0, 1)) + VB
        num = len(re.findall(pattern, self.taggedText))
        return 1000 * num / self.tokenNum

    def feature_63(self):
        """N63: split auxiliaries"""
        pattern = AUX + ADV + REPEAT(ADV, (0, 1)) + VB
        num = len(re.findall(pattern, self.taggedText))
        return 1000 * num / self.tokenNum

    def feature_64(self):
        """O64: phrasal coordination"""
        pattern = OR([ADV, ADJ, V, N]) + " (and)_[A-Z]+" + OR([ADV, ADJ, V, N])
        num = len(re.findall(pattern, self.taggedText))
        return 1000 * num / self.tokenNum

    def feature_65(self):
        """O65: independent clause coordination"""
        pattern_1 = "( ,_,)" + "( (and)_[A-Z]+)" + \
                    "( (it|so|then|you|there)_[A-Z]+)" + \
                    OR([BE, DEMOPRO, SUBJPRO])
        pattern_2 = CL_P + "( and_[A-Z]+)"
        pattern_3 = "( and_[A-Z]+)" + OR([WHP, WHO,
                                          "( (because|though|although|if"
                                          "|unless)_[A-Z]+)",
                                          "( (well|now|anyway|anyhow|anyways)"
                                          "_[A-Z]+)", getCONJ()])
        pattern = OR([pattern_1, pattern_2, pattern_3])
        num = len(re.findall(pattern, self.taggedText))
        return 1000 * num / self.tokenNum

    def feature_66(self):
        """P66: synthetic negation"""
        pattern_1 = "( no_[A-Z]+)" + OR([QUAN, ADJ, N])
        pattern_2 = "(neither|nor)_[A-Z]+"
        pattern = OR([pattern_1, pattern_2])
        num = len(re.findall(pattern, self.taggedText))
        return 1000 * num / self.tokenNum

    def feature_67(self):
        """P67: analytic negation"""
        pattern = OR([" not_[A-Z]+", " n't_[A-Z]+"])
        num = len(re.findall(pattern, self.taggedText))
        return 1000 * num / self.tokenNum


def getBiberFeature(text):
    """calculate all linguistic features"""
    features = dict()
    biberText = BiberText(rawText=text)
    features['PASTTENSE'] = [biberText.feature_01()]
    features['PERFECTS'] = [biberText.feature_02()]
    features['PRES'] = [biberText.feature_03()]
    features['PL_ADV'] = [biberText.feature_04()]
    features['TM_ADV'] = [biberText.feature_05()]
    features['PRO1'] = [biberText.feature_06()]
    features['PRO2'] = [biberText.feature_07()]
    features['PRO3'] = [biberText.feature_08()]
    features['IT'] = [biberText.feature_09()]
    features['PDEM'] = [biberText.feature_10()]
    features['PANY'] = [biberText.feature_11()]
    features['PRO_DO'] = [biberText.feature_12()]
    features['WH_QUES'] = [biberText.feature_13()]
    features['N_NOM'] = [biberText.feature_14()]
    features['AGLS_PSV'] = [biberText.feature_17()]
    features['BY_PASV'] = [biberText.feature_18()]
    features['BE_STATE'] = [biberText.feature_19()]
    features['EX_THERE'] = [biberText.feature_20()]
    features['TH_CL'] = [biberText.feature_21()]
    features['ADJ_CL'] = [biberText.feature_22()]
    features['WH_CL'] = [biberText.feature_23()]
    features['INF'] = [biberText.feature_24()]
    features['CL_VBG'] = [biberText.feature_25()]
    features['CL_VBN'] = [biberText.feature_26()]
    features['WHIZ_VBN'] = [biberText.feature_27()]
    features['WHIZ_VBG'] = [biberText.feature_28()]
    features['THTREL_S'] = [biberText.feature_29()]
    features['THTREL_O'] = [biberText.feature_30()]
    features['REL_SUBJ'] = [biberText.feature_31()]
    features['REL_OBJ'] = [biberText.feature_32()]
    features['REL_PIPE'] = [biberText.feature_33()]
    features['SENT_REL'] = [biberText.feature_34()]
    features['SUB_COS'] = [biberText.feature_35()]
    features['SUB_CON'] = [biberText.feature_36()]
    features['SUB_CND'] = [biberText.feature_37()]
    features['SUB_OTHR'] = [biberText.feature_38()]
    features['PREP'] = [biberText.feature_39()]
    features['ADJ_ATTR'] = [biberText.feature_40()]
    features['ADJ_PRED'] = [biberText.feature_41()]
    features['ADVS'] = [biberText.feature_42()]
    features['TYPETOKEN'] = [biberText.feature_43()]
    features['WORDLNGTH'] = [biberText.feature_44()]
    features['CONJNCTS'] = [biberText.feature_45()]
    features['DOWNTONE'] = [biberText.feature_46()]
    features['GENHDG'] = [biberText.feature_47()]
    features['AMPLIFR'] = [biberText.feature_48()]
    features['GEN_EMPH'] = [biberText.feature_49()]
    features['PARTCLE'] = [biberText.feature_50()]
    features['DEM'] = [biberText.feature_51()]
    features['POS_MOD'] = [biberText.feature_52()]
    features['NEC_MOD'] = [biberText.feature_53()]
    features['PRD_MOD'] = [biberText.feature_54()]
    features['PUB_VB'] = [biberText.feature_55()]
    features['PRV_VB'] = [biberText.feature_56()]
    features['SUA_VB'] = [biberText.feature_57()]
    features['SEEM'] = [biberText.feature_58()]
    features['CONTRAC'] = [biberText.feature_59()]
    features['THAT_DEL'] = [biberText.feature_60()]
    features['FINLPREP'] = [biberText.feature_61()]
    features['SPL_INF'] = [biberText.feature_62()]
    features['SPL_AUX'] = [biberText.feature_63()]
    features['P_AND'] = [biberText.feature_64()]
    features['O_AND'] = [biberText.feature_65()]
    features['SYNTHNEG'] = [biberText.feature_66()]
    features['NOT_NEG'] = [biberText.feature_67()]
    return features
