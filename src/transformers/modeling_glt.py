import logging
import math
import os

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from .activations import gelu, gelu_new, swish
from .configuration_bert import BertConfig
from .file_utils import add_start_docstrings, add_start_docstrings_to_callable
from .modeling_utils import PreTrainedModel, prune_linear_layer
from .modeling_bert import BertPreTrainedModel

from .glt_ungrounded import GroundedCKYEncoder

logger = logging.getLogger(__name__)

@add_start_docstrings(
    """Bert Model transformer with a GLT head on top e.g. for GLUE tasks. """,
    BERT_START_DOCSTRING,
)
class BertForGLTSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.2)

        self.classifier = nn.Sequential(nn.Linear(4*512, 512),
                                       nn.Tanh(), nn.LayerNorm(512),
                                       nn.Dropout(0.2), nn.Linear(512, 3))

        self.projection = nn.Linear(768, 512)

        config = BertConfig(
            grounded=False,
            max_sentence_length=64,
            hidden_size=512,
            input_img_dim=None,
            max_position_embeddings=64,
            use_position_embeddings=False,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0,
            layer_dropout_prob=0.25,
            intermediate_size=512,
            layers_to_tie=["pair_compose.intermediate.dense","pair_compose.attention","pair_compose.constt_energy"],
            tie_layer_norm=True,
            answer_pooler=False,
            non_compositional_reps=False,
            visual_module_dropout_prob=0,
            answer_comp_dropout_prob=0.0,
            input_size=512
        )
        self.modeling_layer = GroundedCKYEncoder(config)


        self.init_weights()

    def left_justify(a, axis=1, side='left'):    
        """
        Justifies a 3D array by sequence

        Parameters
        ----------
        A : torch Tensor, (batch_size, sequence_length, hidden_size)
            Input array to be justified
        axis : int
            Axis along which justification is to be made
        side : str
            Direction of justification. It could be 'left', 'right', 'up', 'down'
            It should be 'left' or 'right' for axis=1 and 'up' or 'down' for axis=0.

        """
        # Zeros is the pad 
        invalid_val = torch.zeros(a.shape[2])

        # Find where not all equal to pad value (valid token is)
        mask = torch.all(torch.eq(a, invalid_val),  dim=2)
        mask = torch.logical_not(torch.stack(a.shape[2]*[mask],axis=2))
        mask = mask.to(torch.int)

        # Flip
        justified_mask = torch.sort(mask,axis) # bump internal pads out
        justified_mask = torch.flip(justified_mask.values, [axis])
        justified_mask = justified_mask.to(torch.bool)
        mask = mask.to(torch.bool)

        # print(a.shape)

        # Fill values
        out = torch.zeros(a.shape)
        out[justified_mask] = a[mask]

        return out

    def separate_batches(sequence_output, attention_mask, token_type_ids):    
        """
        Separates sentence_1 and sentence_2 batches

        Parameters
        ----------
        sequence_output : torch Tensor, (batch_size, sequence_length, hidden_size)
            Output of BERT for classifier
        attention_mask: (batch_size, sequence_length)
        token_type_ids: (batch_size, sequence_length)

        """
        
        token_type_ids_a = torch.logical_not(torch.Tensor(token_type_ids)).to(torch.int)
        token_type_ids_b = token_type_ids # (batch_size, sequence_length)

        reshaped_sequence = sequence_output.permute(2,0,1) # (hidden_size, batch_size, sequence_length)

        batch_1 = reshaped_sequence * attention_mask * token_type_ids_a # (hidden_size, batch_size, sequence_length)
        batch_2 = reshaped_sequence * attention_mask * token_type_ids_b

        batch_2 = left_justify(batch_2.permute(1,2,0))

        batch_1 = batch_1[:,:,:64].permute(1,2,0) # (batch_size, sequence_length, hidden_size/2)
        batch_2 = batch_2[:,:,:64]

        invalid_val = torch.zeros(sequence_output.shape[2]) # Zeros is the pad 
        mask_1 = torch.logical_not(torch.all(torch.eq(batch_1, invalid_val),  dim=2)).to(torch.int) # (batch_size, sequence_length)
        mask_2 = torch.logical_not(torch.all(torch.eq(batch_2, invalid_val),  dim=2)).to(torch.int)

        return batch_1, batch_2, mask_1, mask_2

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import BertTokenizer, BertForSequenceClassification
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)

        loss, logits = outputs[:2]

        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids, # (batch_size, sequence_length)
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0] # (batch_size, sequence_length, hidden_size)

        projected_output = self.projection(sequence_output) # (batch_size, sequence_length, 512)

        batch_1, batch_2, batch_1_mask, batch_2_mask = self.separate_batches(projected_output, attention_mask, token_type_ids) # (batch_size, sequence_length/2, hidden_size)

        emb1 = self.dropout(self.modeling_layer(batch_1, batch_1_mask))
        emb2 = self.dropout(self.modeling_layer(batch_2, batch_2_mask))

        pair_emb = torch.cat([emb1, emb2, torch.abs(emb1 - emb2), emb1 * emb2], 1)
        logits = self.classifier(pair_emb)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)