import json
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

thresh=200

print('start')
data = []
with open('tag_vectors.jsonl', 'r', encoding='utf-8') as f:
    n=0
    for line in f:
        data.append(json.loads(line))
        n+=1
        if n>=thresh:
            break

matrix = np.vstack(data) 


tag_sim = cosine_similarity(matrix)

print('tag sim calculated')



data = []
with open('title_vectors.jsonl', 'r', encoding='utf-8') as f:
    n=0
    for line in f:
        data.append(json.loads(line))
        n+=1
        if n>=thresh:
            break




matrix = np.vstack(data)  


title_sim = cosine_similarity(matrix)

print('title sim calculated')

similarity_sum =  tag_sim*1


matrix_list = similarity_sum.tolist()

with open('similarity_matrix.json', "w", encoding="utf-8") as f:
    json.dump(matrix_list, f)

print(similarity_sum[0][0:10])
