from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.datasets import fetch_20newsgroups
import heapq, operator

# Using Stemmer from the PyStemmer package because it's faster than the nltk stemmer
# Package can be downloaded from here: https://pypi.python.org/pypi/PyStemmer
import Stemmer
english_stemmer = Stemmer.Stemmer('en')

# Extension of the normal Tfidf vectorizer so that it stems words before analyzing
class StemmedTfidfVectorizer(TfidfVectorizer):
	def build_analyzer(self):
		analyzer = super(TfidfVectorizer, self).build_analyzer()
		return lambda doc: english_stemmer.stemWords(analyzer(doc))

# Download the 20 newsgroups dataset
documents = fetch_20newsgroups()

# Sample queries representing the interests of our users 
queries = {"soccer": "soccer goal league championship striker player score coach football",
		   "music": "music album cd lp song singer play listen genre album band",
		   "cars": "car motor fuel petrol cylinder steering drive hybrid chassis engine mph",
		   "films": "film movie actor director role genre scene camera",
		   "polytics": "polytics government candidate campaign president election minister votation"}

#tfidf_vectorizer = StemmedTfidfVectorizer(input = "filename", smooth_idf = True)

# TODO: We have to decide wether to smooth idf or not, and why
tfidf_vectorizer = StemmedTfidfVectorizer(smooth_idf = True)
tfidf_matrix = tfidf_vectorizer.fit_transform(documents.data)

# Modify the params of the vectorizer so that it accepts content instead of paths
# params = tfidf_vectorizer.get_params()
# params["input"] = "content"
# tfidf_vectorizer.set_params(**params)
q_matrix = tfidf_vectorizer.transform(queries.values())

# TODO: Select query via command line argument
distances = cosine_similarity(q_matrix[1:2], tfidf_matrix)[0]

# Top N most similar results (removing the first one because it's the same one, cos = 1)
n = 3
#top_indexes = [list(distances[0]).index(x) for x in sorted(list(distances[0]), reverse=True)[1:n+1]]
top_indexes = zip(*heapq.nlargest(3, enumerate(distances), key=operator.itemgetter(1)))[0]

print("Top {0} matches: \n------------------------------".format(n))
for i in top_indexes:
	print("Index: {0} ----------------------------".format(i))
	print(documents.data[i])


