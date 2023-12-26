#  A simple script to split the large training set into 50 separate files.

ttext = open('TinyStoriesModel/stories/TinyStoriesV2-GPT4-train.txt', 'r', encoding='utf-8').read()

numSplits = 50
endpoint = 0
for i in range(1,numSplits+1):
    startpoint = endpoint
    num = str(i) if i>=10 else '0'+str(i)
    x = len(ttext)
    if i==numSplits:
        endpoint = x
    else:
        endpoint = (x//numSplits)*i
        while ttext[endpoint]!='\n' or ttext[endpoint-1]!='>' or ttext[endpoint-2]!='|':
            endpoint+=1
        while ttext[endpoint]=='\n':
            endpoint+=1
    chunk = ttext[startpoint:endpoint]

    fileName = 'TinyStoriesModel/stories/' + 'TinyStories' + num + '.txt'
    file = open(fileName, 'w', encoding='utf-8')
    file.write(chunk)
    file.close()