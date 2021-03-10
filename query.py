from decomp import UDSCorpus
import json

# this script queries a specified role for positive examples

# read the specified split of the UDS corpus
uds = UDSCorpus(split='test')

# query the database for positive examples
# AGENT := ((volition > 0) V (instigation > 0)) ^ (existed-before > 0)
agentstr = """
           SELECT ?edge
           WHERE { ?edge <domain> <semantics> ;
                         <type> <dependency> ;
                         <existed_before> ?existed_before
                         FILTER ( ?existed_before > 0) .
                   { ?edge <volition> ?volition
                           FILTER ( ?volition > 0 )
                   } UNION
                   { ?edge <instigation> ?instigation
                           FILTER ( ?instigation > 0 )
                   }
                 }
           """

patntstr = """
           SELECT ?edge
           WHERE { ?edge <domain> <semantics> ;
                         <type> <dependency> ;
                         <change_of_state> ?change_of_state
                         FILTER ( ?change_of_state > 0) .
                   { ?edge <volition> ?volition
                           FILTER ( ?volition < 0 )
                   } UNION
                   { ?edge <instigation> ?instigation
                           FILTER ( ?instigation < 0 )
                   } .
                   { ?edge <existed_before> ?existed_before
                           FILTER ( ?existed_before > 0 )
                   } UNION
                   { ?edge <existed_after> ?existed_after
                           FILTER ( ?existed_after > 0 )
                   }
                 }
           """

instrstr = """
           SELECT ?edge
           WHERE { ?edge <domain> <semantics> ;
                         <type> <dependency> ;
                         <was_used> ?was_used
                         FILTER ( ?was_used > 0) .
                   { ?edge <sentient> ?sentient
                           FILTER ( ?sentient < 0 )
                   } UNION
                   { ?edge <volition> ?volition
                           FILTER ( ?volition < 0 )
                   } UNION
                   { ?edge <instigation> ?instigation
                           FILTER ( ?instigation < 0 )
                   }
                 }
           """

themestr = """
           SELECT ?edge
           WHERE { ?edge <domain> <semantics> ;
                         <type> <dependency> ;
                         <existed_during> ?existed_during
                         FILTER ( ?existed_during > 0) .
                   { ?edge <change_of_state> ?change_of_state
                           FILTER ( ?change_of_state > 0 )
                   } UNION
                   { ?edge <change_of_location> ?change_of_location
                           FILTER ( ?change_of_location > 0 )
                   } UNION
                   { ?edge <change_of_possesion> ?change_of_possesion
                           FILTER ( ?change_of_possesion > 0 )
                   }
                 }
           """

experstr = """
           SELECT ?edge
           WHERE { ?edge <domain> <semantics> ;
                         <type> <dependency> ;
                         <awareness> ?awareness;
                         <sentient> ?sentient;
                         <change_of_state_continuous> ?change_of_state_continuous
                         FILTER ( ?change_of_state_continuous > 0 && ?sentient > 0 && ?awareness > 0 ) .
                 }
           """

# pick which role to query
querystr = experstr

# query first i items in uds_train
results = {}
#i = 0
for gid, graph in uds.items():
      #if i == 999:
      #      break
      results.update({gid: graph.query(querystr, query_type='edge', cache_rdf=False)})
      #i += 1

# load all split examples
with open('test.json', "r") as s:
      training_examples = json.load(s)
s.close()

num = 0 # tracker for frequency
#training_examples = {}
# For each sentence, find all positive training examples
for key, value in results.items():
      if bool(value): # if not empty, ie if it has valid edges
            sent_id = key
            for k, v in value.items():
                  # extract predicate and argument positions
                  pred_id, arg_id = k # TODO check if args are always after preds

                  # create unique id for each example by concatenating sent, pred, 
                  # and arg numbers
                  example_id = sent_id.split('-')[-1] + '_' + pred_id.split('-')[-1] + '_' + arg_id.split('-')[-1]

                  # create training example
                  training_example = {'sent_id': sent_id,
                                      'pred_id': pred_id,
                                      'arg_id': arg_id,
                                      'label': 1}
                  num+=1
                  training_examples.update({example_id: training_example})

print('Frequency of positive examples: ', num)

out_file = open("test_experiencer.json", "w") 
json.dump(training_examples, out_file) 
out_file.close()