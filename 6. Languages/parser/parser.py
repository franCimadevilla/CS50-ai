import nltk
try:
    nltk.data.find("tokenizers/punkt_tab/english/")
except LookupError:
    print("punkt_tab is now being downloaded... ✅")
    nltk.download("punkt_tab")
    print("Download completed! 👍")

import sys

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> NP VP | S Conj S

NP -> Det N | Det AdjG N | N | NP Conj NP | NP PP

AdjG -> Adj | Adj AdjG

VP -> V | VP NP | VP PP | VP Conj VP | Adv VP 

PP -> P NP

"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)
    
    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print(unicodelines=True, nodedist=4)

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))
            for n in np:
                print("Label= {} Height= {}".format(n.label(), n.height()), end="\n\n")

    print("🔍 Trees found: {}".format(len(trees)))
                    


def _rm_no_alphanumeric(str):
        if str.isalpha():
            return True
        elif len(str) == 1:
            return str.isalpha()
        else:
            for c in str:
                if c.isalpha:
                    return True
        return False

def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    def _rm_no_alphanumeric(str):
        if str.isalpha():
            return True
        elif len(str) == 1:
            return str.isalpha()
        else:
            for c in str:
                if c.isalpha:
                    return True
        return False
    return list(filter(_rm_no_alphanumeric, nltk.tokenize.word_tokenize(sentence.lower())))


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    chunks = []
    for np_tree in tree.subtrees(lambda n: n.label() == "NP"):
        if not any(child.label() == "NP" for child in np_tree.subtrees(lambda t: t != np_tree)):
            chunks.append(np_tree)
    return chunks


if __name__ == "__main__":
    main()