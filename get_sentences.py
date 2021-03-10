from decomp import UDSCorpus
import json

# read the specified split of the UDS corpus
# this code puts all of the sentences from each split into one file
# which is used as a lookup table for the tokens
uds = UDSCorpus(split='dev') #12543 train, 2077 test, 2002 dev

with open('sentences.json', "r") as s:
      sentences = json.load(s)
s.close()

# get tokenized sentences
#sentences = {}
for gid, graph in uds.items():
      tokens = graph.sentence.split(" ")
      sentences.update({gid: tokens})

print('Total number of sentences: ', len(sentences))

out = open("sentences.json", "w") 
json.dump(sentences, out) 
out.close()