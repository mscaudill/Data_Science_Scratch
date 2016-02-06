"""
csv_reader_wrapper.py

provides a wrapper for the csv modules reader object. Given an imput list  
of parsers ex float, int etc., the wrapper applies each parser to each 
column of data in the reader object.

"""

import csv

def try_or_none(f):
    """ the parser f may return an exception when applied to a row element
    for ex int('n/a') so we will return None in this case """
    def f_or_none(x):
        try: 
            return f(x)
        except: 
            return None
    return f_or_none

def parse_row(input_row, parsers):
    """ given a list of parsers (some of which may be None apply the
        appropriate one to each element of the row """
    return [try_or_none(parser)(el) if parser is not None else el for 
            el, parser in zip(input_row, parsers)]

def parse_rows_with(reader, parsers):
    """ wrap a reader to apply the parsers to each of its rows. Note here we
        are using a generator to lazily generate a parse row """
    for row in reader:
       yield  parse_row(row, parsers)

data = []

def main(filename,parsers):
    """ main calls open, generates upto line of interest, appends 
        to data, then deletes row if any parsed el is None """
    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        for line in parse_rows_with(reader, parsers):
            data.append(line)
        print data
        for idx, row in enumerate(data):
            if any(x is None for x in row):
                del data[idx]

    return data

