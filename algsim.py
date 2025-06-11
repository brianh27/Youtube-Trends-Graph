import random
import json
with open('similarity_matrix.json', 'r', encoding='utf-8') as f:
    similarity_matrix = json.load(f)
with open('filtered_videos.json', 'r', encoding='utf-8') as f:
    filtered_videos = json.load(f)
if len(similarity_matrix) != len(filtered_videos):
    raise ValueError("The length of similarity_matrix does not match the length of filtered_videos.")


def iterate(x,n,done):
    #zip up all of the files
    sens=5
    mat=[(a,i) for i,a in enumerate(similarity_matrix[x]) if i not in done]
    mat.sort(key=lambda item: item[0], reverse=True)
    
    return mat[n*sens + random.randint(0, sens-1)][1]


n=len(filtered_videos)
def bfs(done, start,n):
    print(n)
    if start==None:
        start=random.randint(0, n-1)
    print(done)

    
    # while True:
    #     try:
    #         n=int(input("Rate how much you enjoy this video from 1-10: "))
            
    #         if n<1 or n>10:
    #             continue
    #         break
    #     except:
    #         continue
            
        

    done.append(start)
    start=iterate(start,10-n,done)
    return done,start
#make cur red
#make done yellow