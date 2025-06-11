
import json
import time
import jsonlines
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

#model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')


class performance():
    def __init__(self):
        self.time=int(time.time() * 1000)
    def check(self):
        millis = int(time.time() * 1000)
        diff=millis-self.time
        self.time=millis
        print(diff)
        return diff

# Load tag_vectors.jsonl
with jsonlines.open('tag_vectors.jsonl', 'r') as reader:
    tag_vectors = [obj for obj in reader]

# Load title_vectors.jsonl
with jsonlines.open('title_vectors.jsonl', 'r') as reader:
    title_vectors = [obj for obj in reader]

with open('filtered_videos.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

#t.check()





def search(query):
    
    query_vector = model.encode(query)
    results = []
    for a in range(min(500, len(tag_vectors), len(title_vectors), len(data))):
        tag_sim = cosine_similarity([query_vector], [tag_vectors[a]])[0][0]
        title_sim = cosine_similarity([query_vector], [title_vectors[a]])[0][0]
        similarity_score = 0.5 * title_sim + 0.5 * tag_sim
        results.append((data[a]['id'], similarity_score))
    results.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in results[:10]]

# If run as a script, provide interactive search
if __name__ == "__main__":
    while True:
        x = input("Type in a search query:")
        if x.lower() == 'exit':
            break
        print(search(x))