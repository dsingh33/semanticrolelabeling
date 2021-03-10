from decomp import UDSCorpus
import json

# this script loads all predicate argument pairs in the dataset for each split

# read the specified split of the UDS corpus
uds = UDSCorpus(split='test')

# query the dataset for all semantic dependency edges
results = {}
for gid, graph in uds.items():
      results.update({gid: graph.semantics_edges()})

training_examples = {}
for key, value in results.items():
      if bool(value): # if not empty, ie if it has valid edges
            sent_id = key
            for k, v in value.items():
                  p = 'protoroles'
                  if p in v.keys(): # only if it has protorole attributes
                        # extract predicate and argument positions
                        pred_id, arg_id = k
                  
                        # create unique id for each example by concatenating sent, pred, 
                        # and arg numbers
                        example_id = sent_id.split('-')[-1] + '_' + pred_id.split('-')[-1] + '_' + arg_id.split('-')[-1]
                  
                        # create training example
                        training_example = {'sent_id': sent_id,
                                            'pred_id': pred_id,
                                            'arg_id': arg_id,
                                            'label': 0}
                        training_examples.update({example_id: training_example})

print('Total number of examples: ', len(training_examples))

out_file = open("test.json", "w") 
json.dump(training_examples, out_file) 
out_file.close()