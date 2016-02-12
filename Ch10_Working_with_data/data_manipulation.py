"""
This module shows how to manipulate data stored as a dictionary. It provides
functions for picking individual keys from a dictionary or across
dictionaries and then demonstrates how the dictionary can be transformed
into another dictionary keyed on a specific key from the original dict.
"""

import datetime

from collections import defaultdict

data = [{'closing_price':102,
         'date': datetime.datetime(2014, 8, 30, 0, 0),
         'symbol':'AAPL'}, 
         {'closing_price':99,
         'date': datetime.datetime(2014, 8, 29, 0, 0),
         'symbol':'AAPL'}, 
         {'closing_price':112,
         'date': datetime.datetime(2014, 9, 2, 0, 0),
         'symbol':'AAPL'}, 
         {'closing_price':89,
         'date': datetime.datetime(2014, 9, 1, 0, 0),
         'symbol':'AAPL'},
         {'closing_price':10,
         'date': datetime.datetime(2014, 7, 1, 0, 0),
         'symbol':'FB'}, 
         {'closing_price':20,
         'date': datetime.datetime(2014, 7, 2, 0, 0),
         'symbol':'FB'}, 
         {'closing_price':5,
         'date': datetime.datetime(2014, 7, 3, 0, 0),
         'symbol':'FB'}, 
         {'closing_price':40,
         'date': datetime.datetime(2014, 7, 4, 0, 0),
         'symbol':'FB'}]

def pluck(key, rows):
    """ turn a list of dicts into a list of values """
    return [row[key] for row in rows]

def group_by(field_name, rows, value_transform=None):
    """ returns a dict where keys are unique values of field_name and value
        is a list of rows """
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[field_name]].append(row)
    
    if value_transform is None:
        return grouped
    else:
        return {key : value_transform(rows)
                for key, rows in grouped.iteritems()}

# Example group by stock symbol and find the max closing price for each
max_price_by_symbol = group_by('symbol', data,
                                lambda rows: 
                                max(pluck('closing_price', rows)))

# what are the largest and smallest one dat percent changes in data?
# 1 order prices by symbol
# 2 order the prices by date
# 3 use zip to get (previous,current)
# 4 turn the pairs into percentage change rows

def percent_price_change(yesterday, today):
    """ returns percentage change in stock price on consecutive days """
    return ((today['closing_price']-yesterday['closing_price'])/
            float(yesterday['closing_price']))
    

def day_over_day_changes(grouped_rows):
    # sort the rows by date
    ordered = sorted(grouped_rows, key=lambda row: row['date'])

    # zip with an offset on consecutive days
    return [{'symbol': today['symbol'], 'date': today['date'],
             'change': percent_price_change(yesterday,today)}
             for yesterday, today in zip(ordered, ordered[1:])]

# now call group by symbol and value transorm the rows with day over day
# changes
changes_by_symbol = group_by('symbol', data, day_over_day_changes)

#collect all the change dicts into a list
all_changes = [change for changes in changes_by_symbol.values() for change
               in changes]
    
# now find largest changes
print max(all_changes, key=lambda row: row['change'])

# similarly we find the min changes
print min(all_changes, key=lambda row: row['change'])
