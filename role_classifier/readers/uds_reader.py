from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from typing import Dict, List, Iterator

from allennlp.data.instance import Instance
from overrides import overrides
import json

from allennlp.data.tokenizers import Token
from allennlp.data.fields import Field, TextField, SequenceLabelField, LabelField

@DatasetReader.register("uds_reader")
class UDSDatasetReader(DatasetReader):

    # Lazy flag allows AllenNLP to not store the dataset in memory if lots of data
    def __init__(self, 
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
    
    # Converts a single file with the data into a list of instances
    def _read(self, file_path: str) -> List[Instance]:
        with open('data/sentences.json', "r") as s:
            sentences = json.load(s)
        s.close()

        with open(file_path, "r") as l:
            instances = json.load(l)
        l.close()

        for key, values in instances.items():
            sent_id = values.get("sent_id")
            pred_id = values.get("pred_id")
            arg_id = values.get("arg_id")
            label = values.get("label")

            words = sentences.get(sent_id)

            yield self.text_to_instance(words, pred_id, arg_id, label)
    
    # Converts a training example into an instance
    def text_to_instance(self,
                         words: List[str],
                         pred_id: str,
                         arg_id: str,
                         label: int) -> Instance:
        fields: Dict[str, Field] = {}
        # wrap each token in the field with a token object
        tokens = TextField([Token(w) for w in words], self._token_indexers)

        # process pred & arg ids to list of sent length
        sent_length = len(words)
        pred_pos = int(pred_id.split('-')[-1]) - 1 # minus 1 bc UDS is 1-indexed
        arg_pos = int(arg_id.split('-')[-1]) - 1
        pred_tags = [0] * sent_length
        arg_tags = [0] * sent_length

        pred_tags[pred_pos] = 1
        arg_tags[arg_pos] = 1

        # Instances in AllenNLP are created using Python dicts
        # which map the token key to the Field type
        fields["tokens"] = tokens
        fields["pred_ind"] = SequenceLabelField(pred_tags, tokens)
        fields["arg_ind"] = SequenceLabelField(arg_tags, tokens)
        fields["label"] = LabelField(label, skip_indexing=True)

        return Instance(fields)
