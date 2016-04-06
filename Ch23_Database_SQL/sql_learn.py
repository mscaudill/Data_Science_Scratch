"""
In Chapter 23, the book presents how to work with databases using SQL. The
problem is that SQL isn't actually used because there is no actual database
to work with. Instead Joel makes a pseudo database and repeats the
functionality of sql in python. This isn't so helpful to me. On codecademy 
there is an excellent tutorial using sql. This module is my notes from that tutorial.
"""

# Definitions #
##################
""" SQL stands for structured query language that operates through simple
declarative statements to manage data stroed in relational databases. Here
are a few basic terms:

A relational database is a database that organizes data into one or more
tables.

A table is a collection of data organized into rows and cols.

A col is a set of data of a particular type (ex.) id, name, age etc...

A row is a record or instance of data in the table

All data stored in a table has a particular data type: integer, date, real
etc... 
"""

# SQL: TABLE MANIPULATION #
###########################
# create a simple table containing celebrity information. The col names
# should be id, name, age

# CREATE A TABLE #
CREATE TABLE celebs(id INTEGER, name TEXT, age INTEGER);

""" Lets break the above line down. CREATE TABLE is called a clause or
command. It is always written in ALL CAPS. Celebs is the name of the table.
(id INTEGER, name TEXT, age INTEGER) is a parameter. Here the parameter is a
list of column names and associated data type """

# INSERT ROW #
INSERT INTO celebs(id, name, age) VALUES (1, 'Justin Bieber', 21);
""" INSERT INTO is a clause that adds the specified row or rows. 
    celebs is the name of the table
    VALUES is a clause that indicates the data being inserted
    (1, 'Justin Bieber', 21) is a parameter identifying the values to be
    inserted. """

# VIEW ALL TABLE ENTRIES #
SELECT * FROM celebs;
"""SELECET is a clause that indicates the statement is a query. The *
indicates the column of the celebs table to return. In this case we return
all columns"""

# UPDATE (edit rows) #
UPDATE celebs
SET age = 22
WHERE id = 1;

""" UPDATE is a clause to edit a row in a table (in this case celebs). SET
is a clause that indicates the column to edit (in this case age). 22 is the
new value to be inserted into the age column. WHERE is a clause that
defines the row(s) to udate with the new column value. In this case id = 1
is the row that have the age column updated to 22"""

# ALTER TABLE #
ALTER TABLE celebs ADD COLUMN twitter_handles TEXT;

""" ALTER TABLE allows you to make specified changes to the celebs table.
ADD COLUMN is a clause indicating a column addition to celebs.
twitter_handle is the name of the column to be added and TEXT is the data
type for the new column"""

# DELETIONS #
DELETE FROM celebs WHERE
twittter_handle IS NULL;

""" To DELETE a row from a table we can use the DELETE clause. The specific
row to be deleted is WHERE the twitter_handle column has a NULL value (i.e.
the value is missing """

# SQL: TABLE QUERYING #
#######################
# We will now look at different SQL commands to query a single table in a
# database named movies







