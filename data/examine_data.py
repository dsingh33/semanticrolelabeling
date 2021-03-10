from decomp import UDSCorpus
import json

# this script examines the dataset

with open('train_patient.json', "r") as t:
      training_examples = json.load(t)
t.close()

with open('sentences.json', "r") as s:
      sentences = json.load(s)
s.close()

i=0
for key, value in training_examples.items():
        if i == 10:
                break
        if value.get('label') == 1:
                sent_id = value.get('sent_id')
                sentence = sentences.get(sent_id)
                print(' '.join(sentence))
                print('Argument: ', sentence[int(value.get('arg_id').split('-')[-1])-1])
                print('Predicate: ', sentence[int(value.get('pred_id').split('-')[-1])-1])
                i+=1