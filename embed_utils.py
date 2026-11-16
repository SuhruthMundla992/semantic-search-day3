
import tiktoken
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv



load_dotenv()  # Load environment variables from .env file


def show_tokens(enc, word):
    ids = enc.encode(word)
    #this split the word into tokens and returns a list of the
    #ids using the encode function from OpenAI's tiktoken library

    subparts_list = [] #store the subparts of the word in a list
    for id in ids:
        subpart = enc.decode([id]) #decodes each id to its coressponding subpart of the word
        subparts_list.append(subpart)

    return len(ids), subparts_list


#takes each page of tezt and find the vector representation of each page and returns a list of vectors
def embed(texts):
    api = OpenAI() #client for OpenAI API
    response = api.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    ) #one request that sends every text at once instead of one call per text
    results = sorted(response.data, key=lambda item: item.index) #puts the results in the same order as the texts that were sent
    vectors = [] #stores one embedding per text, in the same order as texts
    for item in results:
        vectors.append(np.array(item.embedding)) #stores each embedding as a numpy array each containing around 1500 values
    return vectors



def cosine(a, b):
    return np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b)) #this function calculates the cos similiarity between two vector