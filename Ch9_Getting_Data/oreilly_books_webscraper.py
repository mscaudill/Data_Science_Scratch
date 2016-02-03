"""
oreilly_books_webscraper.py

This is an example of how to perform a webscrape of a website. We take the
Oreilly publishers website as an example and scrape all books related to
data science. We extract the book publication year and plot

"""
from bs4 import BeautifulSoup
import requests
import re
from time import sleep
from matplotlib import pyplot as plt
from collections import Counter

# The url below is the webpage for data books on oreilly
base_url = 'http://shop.oreilly.com/category/browse-subjects/' + \
      'data.do?sortby=publicationDate&page='

# call beautiful soup to get the url text and use the html5lib parser
#soup = BeautifulSoup(requests.get(base_url).text, 'html5lib')

# In looking at the HTML source code, each of the videos and books starts
# with one thumbtext tag element <td class="thumbtext">

# start by getting the number of these spans and printing the total number. 
#tds = soup('td', 'thumbtext')
#print len(tds)

# Now lets filter out the videos. Note that following <span
# class='pricelabel'> preceeds an ebook, video or book. so we filter
def is_video(td):
    "it's a video if one price label and starts with Video"
    pricelabel = td('span', 'pricelabel')
    return (len(pricelabel) == 1 and 
            pricelabel[0].text.strip().startswith('Video'))

#print len([td for td in tds if not is_video(td)])

# We are now ready to start pulling data out of each of the td elements. 

def book_info(td):
    """ given a beautiful soup td tag representing a book, extract the book's details and return in a dict"""
    # get title
    title = td.find('div','thumbheader').a.text
    # get the 'By authors' string
    by_author = td.find('div', 'AuthorName').text
    # substitute the 'By ' with ''
    author_string = re.sub('^By ','', by_author)
    # split and strip author_string to list
    authors = [x.strip() for x in author_string.split(',')] 
    isbn_link = td.find('div', 'thumbheader').a.get('href')
    isbn = re.match('/product/(.*)\.do', isbn_link).groups()[0]
    date = td.find('span', 'directorydate').text.strip()

    return {'title' : title, 'authors': authors, 'isbn' : isbn, 'date':date}

# now we are ready to scrape
books = []

# There are a total of 42 pages for data science related books
num_pages = 42

for page_num in range(1,num_pages + 1):
    print 'souping page', page_num,',', len(books),'books found'
    # update the page number
    url = base_url + str(page_num)
    print url
    # call to bs4 using html5 parser
    soup = BeautifulSoup(requests.get(url).text,'html5lib')

    for td in soup('td','thumbtext'):
        if not is_video(td):
            books.append(book_info(td))

    # to obey the robots.txt file we must wait 30s between request
    sleep(30)

# Now that we have a list of data books with info on each,
# we can plot the number of data science books published in each year

def get_book_year(book):
    """ returns the year in which the book was published"""
    # Each date is a month year so we will split and take [1]
    return int(book['date'].split()[1])

# now call the counter on the book years
num_books_in_year = Counter([get_book_year(book) for book in books 
                       if get_book_year(book) < 2016])
# finally plot
# get all the years in ascending order
years = sorted(num_books_in_year)
# get the number of books in each of the sorted years
book_counts = [num_books_in_year[year] for year in years]

plt.plot(years,book_counts,'ro')
plt.ylabel('# of data books')
plt.xlabel('year')
plt.show()
