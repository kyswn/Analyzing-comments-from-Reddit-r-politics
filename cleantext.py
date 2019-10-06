#!/usr/bin/env python3

"""Clean comment text for easier parsing."""

from __future__ import print_function

import re
import string
import argparse


__author__ = ""
__email__ = ""

# Depending on your implementation,
# this data may or may not be useful.
# Many students last year found it redundant.
_CONTRACTIONS = {
    "tis": "'tis",
    "aint": "ain't",
    "amnt": "amn't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hell": "he'll",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "id": "i'd",
    "ill": "i'll",
    "im": "i'm",
    "ive": "i've",
    "isnt": "isn't",
    "itd": "it'd",
    "itll": "it'll",
    "its": "it's",
    "mightnt": "mightn't",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "oclock": "o'clock",
    "ol": "'ol",
    "oughtnt": "oughtn't",
    "shant": "shan't",
    "shed": "she'd",
    "shell": "she'll",
    "shes": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "somebodys": "somebody's",
    "someones": "someone's",
    "somethings": "something's",
    "thatll": "that'll",
    "thats": "that's",
    "thatd": "that'd",
    "thered": "there'd",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "wasnt": "wasn't",
    "wed": "we'd",
    "wedve": "wed've",
    "well": "we'll",
    "were": "we're",
    "weve": "we've",
    "werent": "weren't",
    "whatd": "what'd",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whodve": "whod've",
    "wholl": "who'll",
    "whore": "who're",
    "whos": "who's",
    "whove": "who've",
    "whyd": "why'd",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "yall": "y'all",
    "youd": "you'd",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've"
}

# You may need to write regular expressions.

def removeemp(alist):
    for i in alist:
        if i == ' ':
            alist.remove(i)
            return removeemp(alist)
    else:
        return alist

def sanitize(text):
    """Do parse the text in variable "text" according to the spec, and return
    a LIST containing FOUR strings 
    1. The parsed text.
    2. The unigrams
    3. The bigrams
    4. The trigrams
    """

    # YOUR CODE GOES BELOW:
	
    parsed_text = ""
    unigrams = ""
    bigrams = ""
    trigrams = ""
	
    #Convert all text to lowercase.
    #print(text)
    parsed_text=text.lower()
	
    #Replace new lines and tab characters with a single space.
    parsed_text=parsed_text.replace('\n',' ')
    parsed_text=parsed_text.replace('\t',' ')

    #print(parsed_text)
    #Remove URLs. Replace them with what is inside the []. 
    #[] would be deal with later 
    p1 = re.compile(r'https*:\/\/\S*')
    #p1 = re.compile('http:\/\/.*? ')
    parsed_text = p1.sub('', parsed_text)

    p2 = re.compile(r'www\S*')
    parsed_text = p2.sub('', parsed_text)
    #print(parsed_text)

    # Remove all punctuation (including special characters that are not technically punctuation)
    # except punctuation that ends a phrase or sentence and except embedded punctuation
    puntorem = ['.', '!', '?', ',', ';', ':', '-', '\'', '’']
    si = len(parsed_text)
    for i in range(si-1):
        if parsed_text[i].isalpha() or parsed_text[i].isdigit():
            continue
        elif parsed_text[i] not in puntorem:
            parsed_text = parsed_text[:i] + ' ' + parsed_text[i+1:]

    hou = parsed_text[si-1]
    if (not hou.isalpha()) and (not hou.isdigit()) and (hou not in puntorem):
        parsed_text = parsed_text[:si-1] + ' '

    #Separate all external punctuation such as periods, commas, etc. into their own tokens

    p2 = re.compile(r"([.?!,;:])")
    parsed_text = p2.sub(" \\1 ", parsed_text)
	
    #Split text on a single space. If there are multiple contiguous spaces, you will need to remove empty tokens after doing the split.
    thelist=parsed_text.split()




    thelist=removeemp(thelist)
    #print(thelist)

    #get the first string
    parsed_text = ' '.join(thelist)
    #print(parsed_text)


    puntsep = ['.', '!', '?', ',', ';', ':']
    #get the second string
    ulist = [x for x in thelist if x not in puntsep]
    unigrams = ' '.join(ulist)
    #print(unigrams)


    size=len(thelist)
    #get the third string

    blist=[]
    if size >= 2:
        for i in range(0,size-1):
            if thelist[i] not in puntsep and thelist[i+1] not in puntsep:
                temp = thelist[i] + '_' + thelist[i+1]
                blist.append(temp)

    bigrams = ' '.join(blist)
    #print(bigrams)

    #get the fourth string
    tlist = []
    blist=[]
    if size >= 3:
        for i in range(0,size-2):
            if thelist[i] not in puntsep and thelist[i+1] not in puntsep and thelist[i+2] not in puntsep:
                temp = thelist[i] + '_' + thelist[i+1] + '_' + thelist[i+2]
                tlist.append(temp)

    trigrams = ' '.join(tlist)
    #print(trigrams)

    #print([parsed_text, unigrams, bigrams, trigrams])
    return [parsed_text, unigrams, bigrams, trigrams]


if __name__ == "__main__":
    # This is the Python main function.
    # You should be able to run
    # python cleantext.py <filename>
    # and this "main" function will open the file,
    # read it line by line, extract the proper value from the JSON,
    # pass to "sanitize" and print the result as a list.

    # YOUR CODE GOES BELOW.

    # We are "requiring" your write a main function so you can
    # debug your code. It will not be graded.

    a =  "I'm afraid I can't explain myself, sir. Because I am not myself, you see?"
    sanitize(a)