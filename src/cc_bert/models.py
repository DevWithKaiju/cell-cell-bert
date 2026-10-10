"""
models.py

PyTorch model architectures for cell-cell relation extraction (RE) and a factory
function to instantiate them.

Two entity representation strategies, each usable with either a custom RBERT-style
head (entity-span pooling) or a plain sequence-classification head:

    - "boundary"    (B-*): entities are wrapped in tags, e.g. <E0>name</E0>
    - "replacement" (R-*): entities are replaced by placeholder tokens, [CELL0]

Six model_type values combine these with the classification head:
    B-HYB, R-HYB  : RBERT head using [CLS] + both entity embeddings
    B-ENT, R-ENT  : RBERT head using only the two entity embeddings (no [CLS])
    B-CLS, R-CLS  : standard AutoModelForSequenceClassification on the [CLS] token

Reference:
    Wu, S. & He, Y. (2019). Enriching Pre-trained Language Model with Entity
    Information for Relation Classification. CIKM.
"""

import types

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertModel,
)


class FCLayer(nn.Module):
    """Fully connected layer with dropout and optional tanh activation."""

    def __init__(self, input_dim, output_dim, dropout_rate=0.1, use_activation=True):
        super().__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)


class RBERT(nn.Module):
    """RBERT for boundary-tagged entities: pools the span between <E0>/</E0> (and
    <E1>/</E1>) tags by averaging, excluding the tags themselves."""

    def __init__(
        self, bert_model, tokenizer, num_labels, dropout_rate=0.1, use_cls=True
    ):
        super().__init__()
        self.use_cls = use_cls
        self.bert = bert_model
        self.tokenizer = tokenizer
        self.num_labels = num_labels
        self.hidden_size = self.bert.config.hidden_size

        self.e0_start = tokenizer.convert_tokens_to_ids("<E0>")
        self.e0_end = tokenizer.convert_tokens_to_ids("</E0>")
        self.e1_start = tokenizer.convert_tokens_to_ids("<E1>")
        self.e1_end = tokenizer.convert_tokens_to_ids("</E1>")

        self.cls_fc_layer = FCLayer(self.hidden_size, self.hidden_size, dropout_rate)
        self.entity_fc_layer = FCLayer(self.hidden_size, self.hidden_size, dropout_rate)
        classifier_input_dim = (
            self.hidden_size * 3 if self.use_cls else self.hidden_size * 2
        )
        self.label_classifier = FCLayer(
            classifier_input_dim, num_labels, dropout_rate, use_activation=False
        )

    def extract_range_avg(self, input_ids, sequence_output, start_id, end_id):
        batch_size = input_ids.shape[0]
        hidden = sequence_output.shape[-1]
        result = []
        for b in range(batch_size):
            ids = input_ids[b]
            start_idx = (ids == start_id).nonzero(as_tuple=True)[0]
            end_idx = (ids == end_id).nonzero(as_tuple=True)[0]
            if len(start_idx) == 0 or len(end_idx) == 0:
                result.append(torch.zeros(hidden, device=sequence_output.device))
                continue
            s, e = start_idx[0].item(), end_idx[0].item()
            if e - s <= 1:
                result.append(torch.zeros(hidden, device=sequence_output.device))
            else:
                result.append(sequence_output[b, s + 1 : e].mean(dim=0))
        return torch.stack(result, dim=0)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        sequence_output = outputs.last_hidden_state
        pooled_output = sequence_output[:, 0, :]

        e0_embed = self.entity_fc_layer(
            self.extract_range_avg(
                input_ids, sequence_output, self.e0_start, self.e0_end
            )
        )
        e1_embed = self.entity_fc_layer(
            self.extract_range_avg(
                input_ids, sequence_output, self.e1_start, self.e1_end
            )
        )

        if self.use_cls:
            concat = torch.cat(
                [self.cls_fc_layer(pooled_output), e0_embed, e1_embed], dim=1
            )
        else:
            concat = torch.cat([e0_embed, e1_embed], dim=1)

        logits = self.label_classifier(concat)
        loss = nn.CrossEntropyLoss()(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}


class CellRBERT(nn.Module):
    """RBERT for replacement-token entities: pools the embedding(s) of the
    [CELL0]/[CELL1] placeholder tokens directly (averaged if a token occurs
    more than once)."""

    def __init__(
        self, bert_model, tokenizer, num_labels, dropout_rate=0.1, use_cls=True
    ):
        super().__init__()
        self.use_cls = use_cls
        self.bert = bert_model
        self.tokenizer = tokenizer
        self.num_labels = num_labels
        self.hidden_size = self.bert.config.hidden_size

        self.cell0_id = tokenizer.convert_tokens_to_ids("[CELL0]")
        self.cell1_id = tokenizer.convert_tokens_to_ids("[CELL1]")

        self.cls_fc_layer = FCLayer(self.hidden_size, self.hidden_size, dropout_rate)
        self.entity_fc_layer = FCLayer(self.hidden_size, self.hidden_size, dropout_rate)
        classifier_input_dim = (
            self.hidden_size * 3 if self.use_cls else self.hidden_size * 2
        )
        self.label_classifier = FCLayer(
            classifier_input_dim, num_labels, dropout_rate, use_activation=False
        )

    def extract_token_avg(self, input_ids, sequence_output, target_token_id):
        batch_size = input_ids.shape[0]
        hidden = sequence_output.shape[-1]
        result = []
        for b in range(batch_size):
            indices = (input_ids[b] == target_token_id).nonzero(as_tuple=True)[0]
            if len(indices) == 0:
                result.append(torch.zeros(hidden, device=sequence_output.device))
            else:
                result.append(sequence_output[b, indices].mean(dim=0))
        return torch.stack(result, dim=0)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        sequence_output = outputs.last_hidden_state
        pooled_output = sequence_output[:, 0, :]

        e0_embed = self.entity_fc_layer(
            self.extract_token_avg(input_ids, sequence_output, self.cell0_id)
        )
        e1_embed = self.entity_fc_layer(
            self.extract_token_avg(input_ids, sequence_output, self.cell1_id)
        )

        if self.use_cls:
            concat = torch.cat(
                [self.cls_fc_layer(pooled_output), e0_embed, e1_embed], dim=1
            )
        else:
            concat = torch.cat([e0_embed, e1_embed], dim=1)

        logits = self.label_classifier(concat)
        loss = nn.CrossEntropyLoss()(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}


VALID_MODEL_TYPES = ("B-HYB", "R-HYB", "B-ENT", "R-ENT", "R-CLS", "B-CLS")


def get_model_and_tokenizer(
    model_type, model_name_or_path, num_labels, dropout_rate=0.1
):
    """Instantiate the (model, tokenizer) pair for one of the six model_type values.

    Args:
        model_type: one of VALID_MODEL_TYPES.
        model_name_or_path: base model id/path (e.g. a PubMedBERT checkpoint, or a
            CPT checkpoint such as "mizuno-group/ccbert-R-pretrained").
        num_labels: number of classification labels (2 for this task).
        dropout_rate: dropout applied in the classification head.
    """
    boundary_tokens = ["<E0>", "<E1>", "</E0>", "</E1>"]
    cell_tokens = ["[CELL0]", "[CELL1]"]
    token_config = {
        "B-HYB": boundary_tokens,
        "R-HYB": cell_tokens,
        "B-ENT": boundary_tokens,
        "R-ENT": cell_tokens,
        "R-CLS": cell_tokens,
        "B-CLS": boundary_tokens,
    }
    if model_type not in token_config:
        raise ValueError(
            f"Unknown model_type: {model_type!r}. Available: {list(token_config)}"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    required_tokens = token_config[model_type]
    if not all(t in tokenizer.all_special_tokens for t in required_tokens):
        tokenizer.add_special_tokens({"additional_special_tokens": required_tokens})
    vocab_size = len(tokenizer)

    if model_type in ("B-HYB", "R-HYB", "B-ENT", "R-ENT"):
        base_model = BertModel.from_pretrained(model_name_or_path)
        base_model.resize_token_embeddings(vocab_size)
        use_cls = model_type in ("B-HYB", "R-HYB")
        head_cls = RBERT if model_type.startswith("B-") else CellRBERT
        model = head_cls(
            base_model, tokenizer, num_labels, dropout_rate, use_cls=use_cls
        )
    else:  # R-CLS, B-CLS
        config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        if hasattr(config, "hidden_dropout_prob"):
            config.hidden_dropout_prob = dropout_rate
        if hasattr(config, "attention_probs_dropout_prob"):
            config.attention_probs_dropout_prob = dropout_rate
        if hasattr(config, "classifier_dropout"):
            config.classifier_dropout = dropout_rate
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        model.resize_token_embeddings(vocab_size)

    # transformers' safetensors saver requires contiguous tensors; resize_token_embeddings
    # can leave some non-contiguous, so make state_dict() always return contiguous views.
    original_state_dict = model.state_dict

    def _contiguous_state_dict(self, *args, **kwargs):
        sd = original_state_dict(*args, **kwargs)
        return {k: v.contiguous() for k, v in sd.items()}

    model.state_dict = types.MethodType(_contiguous_state_dict, model)

    return model, tokenizer
