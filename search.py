import json
import embed_utils
import os
import numpy as np
import build_index


def load_index(file_path):
    with open(file_path, "r", encoding = "utf-8") as f:
        entries = json.load(f)
    for entry in entries:
        entry["embedding"] = np.array(entry["embedding"]) #converts each embedding from the json file back to a numpy array to be used later for the cosine similarity

    return entries


def search(query, entries, top_k = 4):
    query_embedding = embed_utils.embed(query)[0] #gets the actual vector embedding value for the query
    scores = []
    for entry in entries:
        score = embed_utils.cosine(query_embedding, entry["embedding"])
        scores.append((score, entry))

    scores = sorted(scores, key = lambda pair: pair[0],reverse = True) #resverse list based on score
    return scores[:top_k] #return the top k results with highest cosine scores


def main():
    build_index.main() #builds index.json from the notes before searching
    index_file = "index.json"
    entries = load_index(index_file)

    while True:
        query = input("Enter your query (or 'exit' to quit): ").strip()
        if query.lower() == "exit":
            break
        if query == "": #skips empty input so it isn't sent to the API
            continue

        results = search(query, entries)
        print("\nTop results:")
        for score, entry in results:
            print(f"Score: {score:.4f}, Source: {entry['source']}, Location: {entry['location']}")
            print(f"Text: {entry['text']}\n")


if __name__ == "__main__":
    main()