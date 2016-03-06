import re

f = open('memes/meme_top.txt', 'r')
f2 = open('memes/meme_bot.txt', 'r')
f3 = open('memes/meme_types.txt', 'r')
#f4 = open('newmemes/meme_typesf.txt', 'r')


lines = []
lines1 = []
lines2 = []

words = []

for line in f:
    l = [];
    for word in line.split():
        word1 = word.upper()
        match = re.findall('^[^a-zA-Z0-9]*([a-zA-Z0-9].*[a-zA-Z0-9])[^a-zA-Z0-9]*$', word1)
        if len(match) > 0:
            l.append(match[0]);
    
    lines1.append(l)

for line in f2:
    l = [];
    for word in line.split():
        word1 = word.upper()
        match = re.findall('^[^a-zA-Z0-9]*([a-zA-Z0-9].*[a-zA-Z0-9])[^a-zA-Z0-9]*$', word1)
        if len(match) > 0:
            l.append(match[0]);
    lines2.append(l)

print len(lines1)
print len(lines2)



for i in range(0, len(lines1)):
    lines.append(lines1[i] + lines2[i])
    words = words + lines1[i] + lines2[i]

types = []

for line in f3:
    types.append(int(line))

m = max(types)

onehots = []

for i in types:
    l = [0]*m
    l[i-1] = 1
    onehots.append(l)

#num_to_meme = {}

#for line in f4:
#    words = line.split(2)
#    k = int(words[0])
#    if k not in num_to_meme:
#        num_to_meme[k] = words[2]
