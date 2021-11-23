"""get the info from the json dataset for our model and build tags"""
from json import encoder
from pathlib import Path
import json


def prep_conll_file(inputfile: str, outputfile: str):
    """
    Label Studio has a conll file export.
    Unfortunatly the format is a bit different then what we want to work with
    So we are rewriting it in a new file
    """
    with open(outputfile, "w") as outfile, open(
        inputfile, "r", encoding="utf-8"
    ) as infile:
        for line in infile:
            if not line.startswith("-DOCSTART-"):
                line = line.split(" ")
                outfile.write(line[0] + " " + line[-1])


def load_data(filename: str):
    """
    takes in the file with data in conll format
    return List [[(sentence1_word1, tag1), (word2, tag2), (word3, tag3)], [(sentence2_word1, tag1),(..),(..)]]
    """
    with open(filename, "r") as file:
        lines = [line[:-1].split() for line in file]
    samples, start = [], 0
    for end, parts in enumerate(lines):
        if not parts:
            sample = [(token, tag.split("-")[-1]) for token, tag in lines[start:end]]
            if sample:
                samples.append(sample)
            start = end + 1
    if start < end:
        samples.append(lines[start:end])
    return samples
