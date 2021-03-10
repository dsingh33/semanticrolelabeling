from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.training.metrics import F1Measure

import torch
import torch.nn as nn

from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from typing import Dict, Optional


@Model.register('neural_davidsonian')
class NeuralDavidsonianClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder) -> None:
        super().__init__(vocab)

        self._embedder = embedder
        self._encoder = encoder
        self._classifier = nn.Linear(in_features=2*encoder.get_output_dim(),
                                     out_features=2)

        self._f1 = F1Measure(positive_label=1)

    def forward(self,
                tokens: Dict[str, torch.Tensor],
                pred_ind: Dict[str, torch.Tensor],
                arg_ind: Dict[str, torch.Tensor],
                label: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(tokens)

        embedded = self._embedder(tokens)
        encoded = self._encoder(embedded, mask)

        pred_ind_unsq = torch.unsqueeze(pred_ind, 2)
        arg_ind_unsq = torch.unsqueeze(arg_ind, 2)

        pred_encoding = encoded * pred_ind_unsq
        arg_encoding = encoded * arg_ind_unsq

        pred_encoding_diag = torch.diagonal(pred_encoding).t()
        arg_encoding_diag = torch.diagonal(arg_encoding).t()

        concatenated = torch.cat((pred_encoding_diag, arg_encoding_diag), dim=-1)

        classified = self._classifier(concatenated)

        self._f1(classified, label)

        output: Dict[str, torch.Tensor] = {}
        loss = nn.CrossEntropyLoss()
        output["logits"] = classified

        if label is not None:
            output["loss"] = loss(classified, label)
        
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self._f1.get_metric(reset)
