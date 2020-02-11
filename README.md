# LinguisticFeatures

This is a python package for calculation of various linguistic features for quantitative/corpus linguistics study. It includes three modules:

* `quita.py`: Linguistic features listed by software *Quantitative Index Text Analyzer (QUITA)*
* `biber.py`: Linguistic features listed by Douglas Biber in *Variation across Speech and Writing (1995)*
* `MD.py`: Algorithm to extract dimensions in MD study

## QUITA Features

*Quantitative Index Text Analyzer (QUITA)* lists the folowing features:

Frequency Structure indicators:
    * Type-Token Ratio (TTR)
    * h-point (h)
    * Vocabulary Richness (R1)
    * Repeat Rate (RR)
    * Relative Repeat Rate of McIntosh (RRmc)
    * Hapax Legomenon Percentage (HL)
    * Lambda (Λ)
    * Gini Coefficient (G)
    * Vocabulary Richness (R4)
    * Curve length (L)
    * Curve length Indicator (R)
    * Entropy (H)
    * Adjusted Modulus (A)

Miscellaneous indicators
    * Verb Distances (VD)
    * Activity (Q)
    * Descriptivity (D)
    * Writer’s View (α)
    * Average Tokens length (ATL)
    * Thematic Concentration (TC)
    * Secondary Thematic Concentration (STC)

## Biber Features

Biber listed the following 67 linguistic features:


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
