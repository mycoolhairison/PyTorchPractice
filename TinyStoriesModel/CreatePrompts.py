# A simple script to grab story beginnings from the validation set,
# to use as prompts for story generation.

vtext = open('TinyStoriesModel/stories/TinyStoriesV2-GPT4-valid.txt', 'r', encoding='utf-8').read()

startPos = 0
endPos = 0
n = len(vtext)
file = open('TinyStoriesModel/stories/prompts.txt', 'w', encoding='utf-8')
mx = 1000
while endPos+75<=n and mx>=0:
    excerpt = vtext[startPos:startPos+75]
    if len(excerpt)>10:
        file.write(excerpt)
        if excerpt[-1]!='\n':
            file.write('\n')
    while vtext[endPos]!='\n' or vtext[endPos-1]!='>' or vtext[endPos-2]!='|':
        endPos+=1
    while vtext[endPos]=='\n':
        endPos+=1
    startPos = endPos
    mx-=1
file.close()