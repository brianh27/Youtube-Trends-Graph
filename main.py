import json
import time
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

with open('sorted_videos.json', 'r', encoding='utf-8') as file:
    data = json.load(file)



#t.check()



def analyze(string, name):
    embedding = model.encode(string)


    embedding = embedding.tolist()

    

    with open(name, "a", encoding="utf-8") as f:
        json.dump(embedding, f)
        f.write("\n")

print('START:')

start=int(input())
print('END:')

n=len(data)
#t=performance()
for i,a in enumerate(data[start:]):
    analyze(a["title"],'title_vectors.jsonl')
    analyze(a["tags"],'tag_vectors.jsonl')
#    t.check()
    print(str(i)+'/'+str(n))
    #time done
    #a/total
    

        