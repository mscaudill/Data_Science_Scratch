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
# database named movies. It has cols id, name, genre, year and imdb_rating
# and has 220 rows of movies

SELECT name, imdb_rating FROM movies;
""" SELECT selects the columns name and imdb_rating from the movies table
returning a table with cols name and imdb_rating for all 220 movies """

SELECT DISTINCT genre FROM movies;
""" SELECT DISTINCT returns unique values in the result set. So the result
set list contains each genere present in the movies table only once. Here
SELECT DISTINCT is a clause that retrieves from the genre column all unique
values in the movies table. Performing good filtering is an essential part
of SQL. """

SELECT * FROM movies WHERE imdb_rating > 8;
""" This statement SELECTS every row FROM movies table WHERE the imdb_rating
column is greater than 8. It is another example of filtering. Other
mathematical operations like > used in SQL are =, !=, >, <, >=, <= """

SELECT * FROM movies WHERE name LIKE 'se_en';
""" LIKE is an operator that can be used with WHERE to compare similar
values. Here we use like to compare two movies (seven and se7en) with the
same name but are spelled differently. se_en has a wildcard _ character that
means you can substitute any character without breaking the pattern. % is
another wildcard character. """

SELECT * FROM movies WHERE name LIKE '%man';
""" The % is a wildcard that filters the result set to only movies with
names ending in 'man'. Similarly 'A%' will filter the result set to only
inlcude names starting with an 'A', note 'A%' would filter out 'A Beautiful
Mind' because there is a space. % is refering to a alphanumeric character
"""

SELECT * FROM movies WHERE name BETWEEN 'A' AND 'J';
""" BETWEEN filters the result set to only include movies with names BETWEEN
'A' and 'J'. I find it interesting that SQL knows this means 'starts with'
A. This would include 'A beautiful mind' , 'Air Force One' etc...Note it
does not include movies starting with J only upto J's. """

SELECT * FROM movies WHERE year BETWEEN 1990 AND 2000;
""" We can also use BETWEEN for integer data type """

SELECT * FROM movies WHERE year BETWEEN 1990 AND 2000 AND genre = 'Comedy';
""" The AND operator allows us to combine multiple filter conditions. Here
year BETWEEN 1990 AND 2000 is the first clause and genre = 'Comedy' is the
second clause. Both conditions must be true for the row to be included """

SELECT * FROM movies WHERE genre = 'Comedy' OR year > 1980;
""" OR is another operator that can be used with WHERE to filter results. If
either WHERE clause is true the row is included """

SELECT * FROM movies ORDER BY imdb_rating DESC;
""" ORDER BY indicates you want to sort the result setby a particular column
either alphabetically or numerically. DESC is a keyword used with ORDER BY
to sort the results in DESCending order."""

SELECT * FROM movies ORDER BY imdb_rating ASC LIMIT 3;
""" LIMIT is a clause that lets you specify the maximum number of rows to
return. """

# SQL: AGGREGATE FUNCTIONS #
############################
# Aggregate functions compute a single result from a set of input values.
# For instance we can average all the values of a particular column. Here we
# are given a table of fake_apps with cols: id, name, category, downloads
# and price. There are 200 rows in the table

# FYI: Aggregate functions are so named because they aggregate values across
# multiple rows to form a single value.

# Count #
SELECT COUNT(*) FROM fake_apps
""" COUNT is a function that takes the name of a column as an argument and
counts the number of rows where the col is not NULL. Here we count every row
so we pass * as an argument"""

SELECT COUNT(*) FROM fake_apps WHERE price = 0;
""" This wil count the rows where the price column is 0 """

SELECT price, COUNT(*) FROM fake_apps GROUP BY price;
""" GROUP BY is a clause in SQL that is only used with aggregate functions
to arracnge identical data into groups. Here COUNT() is our agg function and
price is the argument passed to GROUP BY. SQL will count the number of
rows(apps) for each price in the table. We SELECT both price and COUNT(*) so
the result set is organized into two columns"""
 
# Sum #
SELECT SUM(downloads) FROM fake_apps;
""" SUM is another aggregate function. Here we sum all values in the
download column in the fake_apps table. """

SELECT category, SUM(downloads) FROM fake_apps GROUP BY category;
""" Here we GROUP BY category and then SUM the downloads for each category
returning the category and sum """

# MAX #
SELECT MAX(downloads) FROM fake_apps;
""" MAX is an agg function that computes the maximum of a column """

SELECT name, category, MAX(downloads) FROM fake_apps GROUP BY category;
""" Groups the apps by category and then locates the name of the app with
the most downloads in each category """

# MIN #
SELECT MIN(downloads) FROM fake_apps;
""" MIN returns the minimum of a column """

SELECT name, category, MIN(downloads) FROM fake_apps GROUP BY category;
""" Groups the apps by category and then locates the name of the app with
the least downloads in each category """

# Average #
SELECT AVG(downloads) FROM fake_apps;
""" AVG returns the average of a column in our database table """

SELECT price, AVG(downloads) FROM fake_apps GROUP BY price;
""" AVG can be combined with GROUP BY to get averages between data sectioned
GROUP BY parameters """

SELECT price, ROUND(AVG(downloads),2) FROM fake_apps GROUP BY price;
""" ROUND can be use to round numbers to arbitrary number of significant
digits """

# MULTIPLE TABLES #
###################
# Tables in a database can be related to each other. For example a table of
# music artists and a table of albums are related because each artists may
# have many albums. SQL allows us to write queries that can combine data
# from multiple tables that are related to each other. Assume we have a
# table of albums already with the columns id, name, artist_id, year

CREATE TABLE artists(id INTEGER PRIMARY KEY, name TEXT);
""" CREATE TABLE is called to make an artist table that will be related to
albums. The PRIMARY KEY is a unique row (record) identifier. It is literally
an id value for a record. We will use this value to connect an artist to the
albums they have produced in the albums table. By specifying the id column
as the primary key, SQL makes sure (1) none of the values in the col are
null and (2) each value in the col is unique. """

SELECT * FROM artist WHERE id = 3; 
""" SELECTS the artist with the id of 3. Remember id is the primary key """

SELECT * FROM albums WHERE artist_id = 3;
""" SELECTS all albums with an artist_id of 3. This links with the primary
key of the artist table. Here artist_id is called a FOREIGN KEY. A foreign
key is a column that contains the primary keys of another table connecting
the rows of different tables. Unlike primary keys, foreign keys need not be
unique and can be null. In summary the relationship between the artists and
albums table is the id value of the artists."""

# SELECT MULTIPLE TABLES: CROSS JOIN #
SELECT albums.name, albums.year, artists.name FROM albums, artists;
""" SELECT with mutiple table names separated by a ,. When querying more
than one table tje column names need to be specified by
table_name.column_name. This is called a CROSS JOIN. It simply lines up the
columns from the tables although those columns are not related."""

# MULTIPLE TABLES: INNER JOIN #
SELECT * FROM albums
    JOIN artists ON
        albums.artist_id = artists.id;
""" Lets break down this sql statement. 
1. SELECT * specifies the cols our result set will have. We include
   every colum in both tables here.
2. FROM albums specifies the table we are querying
3. JOIN artists ON specifies the type of join and the name of the second
   table.
4. albums.artist_id = artists.id is the join condition that describes how
   the two tables are related to each other. SQL uses the FOREIGN KEY column
   artist_id in the albums to match the primary key id in the artists table.
   We now that it will only match with one row of artists table because the
   id is a PRIMARY KEY
"""

# MULTIPLE TABLES: LEFT OUTER JOIN #
SELECT * FROM albums
    LEFT JOIN artist ON
        albums.artist_id = artists.id;
""" a LEFT OUTER JOIN does not require the join condition to be met, instead
every row in the left table is returned in the result set and if the
condition is not met NULL values are used to fill in the columns from the
right table. Here the left table is albums and the right table is artist"""

# MULTIPLE TABLES: AS KEYWORD #
###############################
SELECT
    albums.name AS 'Album',
    albums.year,
    artists.name AS 'Artist'
FROM
    albums
JOIN artists ON
    albums.artist_id = artists.id
WHERE
    albums.year > 1980;

""" The result set of this SQL query returns a table with columns Album,
year and Artist. So As renames a column or table using an alias. It must be
in single quotes. We do this sometimes because it can be confusing when the
same name is used for columns in two different tables. In the artists albums
case name is used as the artists name and album name so aliasing these makes
sense. It is important to note that the column names in the original tables
remain unchanged only the column names in the result set are aliased. """













