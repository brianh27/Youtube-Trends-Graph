# import OS module
import os
import csv
import json
class error(Exception):
    pass
def to_ascii_only(s):
    return s.encode('ascii', 'ignore').decode('ascii')
# Get the list of all files and directories
path = "./data/US_youtube_trending_data.csv"


all_videos={}
filtered_videos=[]
def parseCSV(name):
    
    with open(name, encoding="utf-8", errors="replace") as csvfile:

        reader = csv.DictReader(csvfile)
        for row in reader:
            all_videos[row['video_id']]=row


parseCSV(path)
# Remove videos with no tags
categories={}
catPath='./data/US_category_id.json'
with open(catPath, 'r', encoding='utf-8') as f:
    cat = json.load(f)
    for v in cat['items']:
        categories[v['id']]=v['snippet']['title']

for v in all_videos:
    video=all_videos[v]
    if video['tags']!='[None]' and (video['video_id'][0]=='M'):

        filtered_videos.append({'id':video['video_id'],'title':to_ascii_only(video['title']),'tags':to_ascii_only(video['tags']),'views':video['view_count'],'publishedAt':video['publishedAt'],'category':categories[video['categoryId']]})
with open('categories.json', 'w', encoding='utf-8') as f:
    json.dump(list(categories.values()), f, indent=2, ensure_ascii=False)
print(len(filtered_videos))
with open('all_videos.json', 'w', encoding='utf-8') as f:
    json.dump(all_videos, f, indent=2, ensure_ascii=False)

with open('filtered_videos.json', 'w', encoding='utf-8') as f:
    json.dump(filtered_videos, f, indent=2, ensure_ascii=False)