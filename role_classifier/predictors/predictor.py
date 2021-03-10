from allennlp.predictors.predictor import Predictor
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
import numpy as np

@Predictor.register('nd_predictor')
class NDPredictor(Predictor):
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)

        outputs['tokens'] = [str(token) for token in instance.fields['tokens'].tokens]
        outputs['predicted'] = str(np.argmax(outputs['logits']))
        outputs['labels'] = str(instance.fields['label'])

        return sanitize(outputs)