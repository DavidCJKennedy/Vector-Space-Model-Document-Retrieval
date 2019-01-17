import math
from operator import itemgetter
from collections import Counter

class Retrieve:

#==============================================================================
## MARK = Global variables for storage of document lengths and the IDF values for each term
    
    docLengths = 0
    IDFValues = 0

#==============================================================================
## MARK - Create new Retrieve object storing index and termWeighting scheme
   
    def __init__(self,index, termWeighting):
        self.index = index
        self.termWeighting = termWeighting

#==============================================================================
## MARK - Method performing retrieval for specified query
        
    def forQuery(self, query):
        
        # Assignment of local variables as accessing global variables in Python is in-efficient
        index = self.index
        termWeight = self.termWeighting
        
        # Retrieve all documents that have some relevance to query, if a document features one more terms in the query it is a candidate
        candidates, candidateIDs = self.getCandidateDocuments(index, query)
        
        # Initialise the document lengths global variable. 
        # A global variable is used because this variable is only assigned to once on the first query, rather than re-assigned every query.
        # Document lengths are constant regardless of query, so re-assignment is in-efficient
        if self.docLengths == 0:
            self.docLengths = self.getDocumentLengths(index)
        
        # Similar to initialisation of document lenths, IDF values are only required for tfidf and are constant for all queries. Assigning this variable once is more efficient. 
        if self.IDFValues == 0 and termWeight == "tfidf":
            self.IDFValues = self.getIDFValues(index)
        
        # Depending on weighting use correct function to return document ID's
        if termWeight == "tf":
            relevantDocs = self.getRelevantDocsUsingTF(query, candidates)
        
        elif termWeight == "binary":
            relevantDocs = self.getRelevantDocsUsingBinary(query, candidateIDs) 
            
        else: # TFIDF term weighting
            relevantDocs = self.getRelevantDocsUsingTFIDF(query, candidates, self.IDFValues)
         
        return relevantDocs # return the relevant document ID's ranked by relevancy back to the ir_engine
  
#==============================================================================
## MARK - Method to return candidate documents for a query
        
    def getCandidateDocuments(self, index, query):
        candidates = {} # Init dictionary to store candidate documents to each query term
        candidateIDs = [] # Init list to store each candidate ID, useful for binary term weighting 
        queryTerm, termFrequency = zip(*query.items()) # Split query into the terms and term frequency
        termSet = set(queryTerm) # Convert terms list into a set for improved efficiency
        
        for term in termSet:
            if term in index: # If term in dictionary is faster than try and exception because execption block is longer to handle
                termCandidates = index[term] # Return documents that contain the term
                candidates[term] = termCandidates # Store candidates in dictionary to their relevant query term
                docIDs, frequency = zip(*termCandidates.items()) # Split the dictionary holding documents that contain term into document ID's and occurrences
                candidateIDs.extend(docIDs) # Add candidate document ID's to seperate array 
        
        return candidates, candidateIDs

#==============================================================================    
## MARK - Method to return length of every document in the corpus
        
    def getDocumentLengths(self, index):
        docLengths = {} # Init dictionary to store documentID's to the sum of term occurrences in document squared. eg documentID : length
        
        for term, documents in index.items(): # Iterate through every term in the index
            for ID, num in documents.items(): # Iterate through every document where term occurs
                if ID in docLengths:
                    docLengths[ID] = docLengths[ID] + num**2 # If document ID exists then add term frequency**2 to the existing sum
                else:
                    docLengths[ID] = num**2 # Else create a new entry and begin the document length count         
        
        return docLengths
 
#==============================================================================
## MARK - Method to return IDF value for every term in the index
        
    def getIDFValues(self, index):
        IDFrequencies = {} # Init dictionary to store IDF value for each term in index. eg term: IDF
        
        for term, documents in index.items(): # Iterate through every term in the index
            IDFrequencies[term] = math.log(3204/len(documents)) # Calculate the IDF for the term, IDF = log(num of documents / number of documents where term occurs)
        
        return IDFrequencies

#==============================================================================
## MARK - Method to return document ID's ranked by relevance using term frequency criterion
        
    def getRelevantDocsUsingTF(self, query, candidates):
        docLen = self.docLengths # Init a local variable for global varaible docLengths, local variables are faster to access than global variables
        termFrequencies = {} # Init a dictionary to store terms to corresponding frequency in query and document. eg term {documentID : frequency}
        similarityScores = {} # Init a dictionary to store final similarity scores for each document. eg documentID : score
        
        for term in query: # Iterate through each term in query
            if term in candidates: # If term has any candidate documents then tf can be calculated
                for ID, termCount in candidates[term].items(): # Iterate through each candidate document for query term 
                    termFrequency = query[term] * termCount # Calculate term frequency for each individual candidate document
                    if ID in termFrequencies: 
                        termFrequencies[ID] = termFrequencies[ID] + termFrequency # If document ID exists then add termFrequency value to existing sum
                    else:
                        termFrequencies[ID] = termFrequency # Else create new entry and begin term frequency count
                        
        for ID, termFreq in termFrequencies.items():
            similarityScores[ID] = termFreq/math.sqrt(docLen[ID]) # Calculate cosine similarity for each candidate document             
        
        # Sort the documents into order of similarity and only retain document IDs as score is not required 
        # Using list comprehensions, itemgetter and reverse = True as an argument rather than .reverse improves efficiency
        similarityScores = [k for k,v in sorted(similarityScores.items(), key = itemgetter(1), reverse = True)]
        
        return similarityScores

#==============================================================================
## MARK - Method to return document ID's ranked by relevance using term frequency inverse document frequency criterion 
        
    def getRelevantDocsUsingTFIDF(self, query, candidates, IDFValues):
        docLen = self.docLengths
        similarityScores = {}
        TFIDF = {} # Init a dictionary to store TFIDF values for each candidate document. eg documentID: TFIDF
        
        ## Calculate the tf for each term first in same way as tf function ##
        for term in query:
            if term in candidates:
                for ID, termCount in candidates[term].items():
                    termFrequency = query[term] * termCount
                    if ID in TFIDF: # If document ID exists then add TFIDF value to existing sum
                        TFIDF[ID] = TFIDF[ID] + (termFrequency * IDFValues[term])  # Get the IDF value for the term then multiply by TF to create TFIDF
                    else:
                        TFIDF[ID] = termFrequency * IDFValues[term] # Else create a new entry and begin TFIDF count
        
        for ID, tfidf in TFIDF.items():
            similarityScores[ID] = tfidf/math.sqrt(docLen[ID]) # Calculate cosine similarity for each candidate document  
        
        # Sort document by score in similar manner as other functions
        similarityScores = [k for k,v in sorted(similarityScores.items(), key = itemgetter(1), reverse = True)]
        
        return similarityScores

#============================================================================== 
## MARK - Method to return document ID's ranked by relevance using binary criterion 
    
    def getRelevantDocsUsingBinary(self, query, candidateIDs):
        docLen = self.docLengths 
        similarityScores = {}
        
        # CandidateIDs can contain multiple identical terms, sort this list in descending order of occurrences of the document IDs
        # Maximum number of document ID occurrences is the same as length of query. Each occurrence represents that a query term was found in the document
        # Counter package is used as is far faster than using .count to sort indexes list. Faster by n(C(n^2) vs C(n)) for .count method 
        candidates = Counter(candidateIDs)
        
        for candidateID, appearances in candidates.items(): # Iterate through each document ID
            similarityVal = (appearances/len(query))/math.sqrt(docLen[candidateID]) # Calculate cosine similarity for each document
            similarityScores[candidateID] = similarityVal # Add document ID and corresponding similarity to dictionary
            
        similarityScores = [k for k,v in sorted(similarityScores.items(), key = itemgetter(1), reverse = True)]
        
        return similarityScores