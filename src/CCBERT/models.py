"""
models.py

This module defines the PyTorch model architectures for Relation Extraction (RE)
and provides a factory function to instantiate them.

It includes:
    1. RBERT: Implementation of R-BERT (Wu & He, 2019) using entity span averaging.
    2. CellRBERT: A variant using special tokens ([CELL0], [CELL1]) for entity representation.
    3. Factory Function: get_model_and_tokenizer to easily switch between architectures.

@Author: Mei Yoshikawa
"""

import torch
from torch import nn
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    BertTokenizerFast, 
    BertModel
)

# Helper Modules
class FCLayer(nn.Module):
    """
    A simple Fully Connected (Dense) layer with Dropout and optional Activation.
    Used for projecting BERT embeddings before classification.
    """
    def __init__(self, input_dim, output_dim, dropout_rate=0.1, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)

# Model 1: Standard R-BERT
class RBERT(nn.Module):
    """
    R-BERT model for Relation Extraction.
    
    This model extracts relation features by concatenating:
    1. The [CLS] token embedding (sentence representation).
    2. The averaged embedding of the first entity span (<E0>...</E0>).
    3. The averaged embedding of the second entity span (<E1>...</E1>).
    
    Reference:
        Wu, S. & He, Y. (2019). Enriching Pre-trained Language Model with Entity Information 
        for Relation Classification. CIKM.
    """
    def __init__(self, bert_model, tokenizer, num_labels, dropout_rate=0.1):
        """
        Args:
            bert_model (PreTrainedModel): Loaded BERT-based model (e.g., PubMedBERT).
            tokenizer (PreTrainedTokenizer): Tokenizer corresponding to the model.
            num_labels (int): Number of classification labels.
            dropout_rate (float): Dropout probability.
        """
        super().__init__()
        self.bert = bert_model
        self.tokenizer = tokenizer
        self.num_labels = num_labels
        self.hidden_size = self.bert.config.hidden_size

        # Retrieve special token IDs for entity boundaries
        # Note: These tokens must be added to the tokenizer beforehand.
        self.e0_start = tokenizer.convert_tokens_to_ids('<E0>')
        self.e0_end   = tokenizer.convert_tokens_to_ids('</E0>')
        self.e1_start = tokenizer.convert_tokens_to_ids('<E1>')
        self.e1_end   = tokenizer.convert_tokens_to_ids('</E1>')

        # FC Layers for projection
        self.cls_fc_layer = FCLayer(self.hidden_size, self.hidden_size, dropout_rate)
        self.entity_fc_layer = FCLayer(self.hidden_size, self.hidden_size, dropout_rate)
        self.label_classifier = FCLayer(self.hidden_size * 3, num_labels, dropout_rate, use_activation=False)

    def extract_range_avg(self, input_ids, sequence_output, start_id, end_id):
        """
        Extracts and averages the embeddings between start_id and end_id tokens.
        
        Args:
            input_ids (torch.Tensor): (batch_size, seq_len)
            sequence_output (torch.Tensor): (batch_size, seq_len, hidden_size)
            start_id (int): Token ID for the start tag.
            end_id (int): Token ID for the end tag.
            
        Returns:
            torch.Tensor: Averaged embeddings of shape (batch_size, hidden_size).
        """
        batch_size, seq_len = input_ids.shape
        hidden = sequence_output.shape[-1]
        result = []
        
        for b in range(batch_size):
            ids = input_ids[b]
            # Find indices of start and end tags
            start_idx = (ids == start_id).nonzero(as_tuple=True)[0]
            end_idx   = (ids == end_id).nonzero(as_tuple=True)[0]
            
            # Handle cases where tags are missing or malformed
            if len(start_idx) == 0 or len(end_idx) == 0:
                result.append(torch.zeros(hidden, device=sequence_output.device))
                continue
            
            # Use the first occurrence of the tags
            s = start_idx[0].item()
            e = end_idx[0].item()
            
            # If the span is empty or invalid (e.g., <E0></E0> adjacent), return zero vector
            if e - s <= 1:
                result.append(torch.zeros(hidden, device=sequence_output.device))
            else:
                # Average embeddings excluding the tags themselves
                vecs = sequence_output[b, s+1 : e]
                avg = vecs.mean(dim=0)
                result.append(avg)
                
        return torch.stack(result, dim=0)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs.last_hidden_state
        pooled_output = sequence_output[:, 0, :]  # [CLS] token

        # Extract entity embeddings
        e0_avg = self.extract_range_avg(input_ids, sequence_output, self.e0_start, self.e0_end)
        e1_avg = self.extract_range_avg(input_ids, sequence_output, self.e1_start, self.e1_end)

        # Apply FC layers
        cls_embed = self.cls_fc_layer(pooled_output)
        e0_embed = self.entity_fc_layer(e0_avg)
        e1_embed = self.entity_fc_layer(e1_avg)

        # Concatenate: [CLS; Entity1; Entity2] -> (batch_size, hidden_size * 3)
        concat = torch.cat([cls_embed, e0_embed, e1_embed], dim=1)
        
        # Classification
        logits = self.label_classifier(concat)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits}

# Model 2: Cell-Token R-BERT (Proposed Variant)
class CellRBERT(nn.Module):
    """
    Proposed variant of R-BERT using special tokens ([CELL0], [CELL1]) for entity representation.
    
    Instead of averaging the span between tags, this model uses the embedding of 
    specific placeholder tokens inserted into the sequence.
    """
    def __init__(self, bert_model, tokenizer, num_labels, dropout_rate=0.1):
        super().__init__()
        self.bert = bert_model
        self.tokenizer = tokenizer
        self.num_labels = num_labels
        self.hidden_size = self.bert.config.hidden_size
        
        # Target token IDs for entities
        self.cell0_id = self.tokenizer.convert_tokens_to_ids('[CELL0]')
        self.cell1_id = self.tokenizer.convert_tokens_to_ids('[CELL1]')
        
        self.cls_fc_layer = FCLayer(self.hidden_size, self.hidden_size, dropout_rate)
        self.entity_fc_layer = FCLayer(self.hidden_size, self.hidden_size, dropout_rate)
        self.label_classifier = FCLayer(self.hidden_size * 3, num_labels, dropout_rate, use_activation=False)

    def extract_token_avg(self, input_ids, sequence_output, target_token_id):
        """
        Extracts embeddings for a specific token ID (e.g., [CELL0]).
        If multiple tokens exist, their embeddings are averaged.
        """
        batch_size, seq_len = input_ids.shape
        hidden = sequence_output.shape[-1]
        result = []
        
        for b in range(batch_size):
            # Find indices where the token matches the target ID
            indices = (input_ids[b] == target_token_id).nonzero(as_tuple=True)[0]
            
            if len(indices) == 0:
                result.append(torch.zeros(hidden, device=sequence_output.device))
            else:
                # Gather embeddings for all occurrences and average them
                vecs = sequence_output[b, indices]
                avg = vecs.mean(dim=0)
                result.append(avg)
                
        return torch.stack(result, dim=0)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs.last_hidden_state
        pooled_output = sequence_output[:, 0, :]  # [CLS] token

        # Extract embeddings for specific cell tokens
        e0_avg = self.extract_token_avg(input_ids, sequence_output, self.cell0_id)
        e1_avg = self.extract_token_avg(input_ids, sequence_output, self.cell1_id)

        # Apply FC layers
        cls_embed = self.cls_fc_layer(pooled_output)
        e0_embed = self.entity_fc_layer(e0_avg)
        e1_embed = self.entity_fc_layer(e1_avg)

        # Concatenate and Classify
        concat = torch.cat([cls_embed, e0_embed, e1_embed], dim=1)
        logits = self.label_classifier(concat)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits}

# 3. Factory Function
def get_model_and_tokenizer(model_type, model_name_or_path, num_labels, dropout_rate=0.1):
    """
    Factory function to initialize the appropriate model and tokenizer.
    
    This function handles:
      1. Tokenizer loading.
      2. Special token registration based on the selected strategy (Boundary vs. Replacement).
      3. Model architecture initialization (Custom R-BERT vs. Standard BERT).

    Args:
        model_type (str): The specific experimental setting.
            - "B-ENT": R-BERT Architecture + Boundary Tags (<E0>).
            - "R-ENT": Cell-RBERT Architecture + Replacement Tokens ([CELL0]).
            - "R-CLS": Standard BERT Classification + Replacement Tokens ([CELL0]).
            - "B-CLS": Standard BERT Classification + Boundary Tags (<E0>).
        
        model_name_or_path (str): Path to the pretrained model (e.g., "microsoft/BiomedNLP-PubMedBERT...").
        num_labels (int): Number of classification labels.
        dropout_rate (float): Dropout probability for the classification head.

    Returns:
        tuple: (model, tokenizer)
    """

    # 1. Configuration: Define Strategy Mappings
    # Define special tokens for each strategy
    boundary_tokens = ['<E0>', '<E1>', '</E0>', '</E1>']
    cell_tokens     = ['[CELL0]', '[CELL1]']
    
    # Map model_type to its required tokens
    token_config = {
        "B-ENT": boundary_tokens,
        "R-ENT": cell_tokens,
        "R-CLS": cell_tokens,
        "B-CLS": boundary_tokens,
    }

    if model_type not in token_config:
        available_types = list(token_config.keys())
        raise ValueError(f"Unknown model_type: '{model_type}'. Available types: {available_types}")

    # 2. Tokenizer Setup (Common Logic)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    required_tokens = token_config[model_type]
    
    # Add special tokens only if they are not already in the vocabulary
    if not all(t in tokenizer.additional_special_tokens for t in required_tokens):
        tokenizer.add_special_tokens({'additional_special_tokens': required_tokens})
    
    vocab_size = len(tokenizer)

    # 3. Model Initialization (Branch by Architecture)
    
    # Group A: Custom R-BERT Architectures
    if model_type in ["R-ENT", "B-ENT"]:
        base_model = BertModel.from_pretrained(model_name_or_path)
        base_model.resize_token_embeddings(vocab_size)
        
        if model_type == "B-ENT":
            model = RBERT(base_model, tokenizer, num_labels, dropout_rate)
        else:
            model = CellRBERT(base_model, tokenizer, num_labels, dropout_rate)

    # Group B: Standard Classification Architectures
    elif model_type in ["R-CLS", "B-CLS"]:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
            hidden_dropout_prob=dropout_rate,
            attention_probs_dropout_prob=dropout_rate
        )
        model.resize_token_embeddings(vocab_size)

    return model, tokenizer