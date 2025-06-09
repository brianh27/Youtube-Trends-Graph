
# import OS module
import os
import csv
import json
class error(Exception):
    pass
# Get the list of all files and directories
path = "./data"
dir_list = os.listdir(path)
print("Files and directories in '", path, "' :")
# prints all files
print(dir_list)
categories={}
all_videos={}
sorted_videos=[]
def parseCSV(name):
    
    with open(name, encoding="utf-8", errors="replace") as csvfile:

        reader = csv.DictReader(csvfile)
        for row in reader:
            all_videos[row['video_id']]=row
def parseJSON(name):
    with open(name,'r') as file:
        data=json.load(file)
    
    for a in data['items']:
        categories[name[0:2]+a['id']]=a['snippet']['title']

for a in dir_list:
    if a.split('.')[0]=='json':
        parseJSON('./data/'+a)
    else:
        pass
for a in dir_list:
    if a.split('.')[1]=='csv':
        parseCSV('./data/'+a)
    else:
        pass
for v in all_videos:
    video=all_videos[v]
    if video['tags']!='[none]':
        sorted_videos.append({'id':video['video_id'],'title':video['title'],'tags':video['tags']})
print(len(sorted_videos))
with open('all_videos.json', 'w', encoding='utf-8') as f:
    json.dump(all_videos, f, indent=2, ensure_ascii=False)

with open('sorted_videos.json', 'w', encoding='utf-8') as f:
    json.dump(sorted_videos, f, indent=2, ensure_ascii=False)