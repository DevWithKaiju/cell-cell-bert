"""
cpt_data.py

This module provides data loading and preprocessing utilities for Continual Pre-training (CPT). It is designed to handle different entity representation strategies in biomedical text:
    
1. Replaced Entities (e.g., [CELL0]): Requires DB lookup to restore original text.
2. Tagged Entities (e.g., <E0>name</E0>): Requires tag stripping to restore original text.

For 50% of the data, the code reverts the processed text back to its original raw form to create a mixed dataset.

@Author: Mei Yoshikawa
"""

import sqlite3
import pandas as pd
from datasets import Dataset

def load_cpt_dataset(data_source, data_path, db_path=None, seed=42):
    """
    Loads the CPT dataset and restores raw text for a subset of samples depending on the masking strategy.

    This function handles two types of entity representations:
    1. 'replacement': Entities are replaced by tokens (e.g., [CELL0]). 
       Since the original names are lost, the original sentences are retrieved from the SemMed DB.
    2. 'boundary': Entities are wrapped in tags (e.g., <E0>...</E0>). 
       The original text is restored by simply removing these tags.

    Args:
        data_source (str): Strategy identifier ('semmed_db' for replaced entities, 'pubmed_csv' for tagged entities).
        data_path (str): Path to the main CSV data file containing processed text.
        db_path (str, optional): Path to the SQLite database (required for 'semmed_db' to lookup original sentences).
        seed (int, optional): Random seed for sampling the 50% subset (default: 42).

    Returns:
        Dataset: A HuggingFace Dataset object containing a mix of processed and raw text.

    Raises:
        ValueError: If db_path is missing when required, or if an unknown data_source is provided.
    """
    df = None

    # ---------------------------------------------------------
    # Case 1: Strategy 'replacement' (e.g., [CELL0])
    # Description: Entity names are replaced by special tokens.
    # Action: Query the SemMed DB to retrieve the original unmasked sentences.
    # ---------------------------------------------------------
    if data_source == "replacement":
        # Retrieve original text data from the database
        conn = sqlite3.connect(db_path)
        query = "SELECT pmid, sent_id, ann_sentence as sentence FROM hit_cells" 
        db_df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Preprocess DB text (remove specific brackets like 【】 to match general formatting)
        db_df['sentence'] = db_df['sentence'].str.replace(r"[【】]", "", regex=True)
        
        # Create a lookup dictionary keyed by (pmid, sent_id)
        text_lookup = dict(zip(zip(db_df['pmid'], db_df['sent_id']), db_df['sentence']))
        
        # Load the dataset with masked entities (e.g., [CELL0])
        cpt_df = pd.read_csv(data_path)
        
        # Select 50% of the data to revert to raw text using the DB lookup
        indices_to_modify = cpt_df.sample(frac=0.5, random_state=seed).index
        
        def get_db_text(row):
            """Retrieve original text from DB; fallback to masked text if not found."""
            key = (row['pmid'], row['sent_id'])
            return text_lookup.get(key, row['Cleaned_Sentence'])

        cpt_df.loc[indices_to_modify, 'Cleaned_Sentence'] = cpt_df.loc[indices_to_modify].apply(get_db_text, axis=1)
        df = cpt_df

    # ---------------------------------------------------------
    # Case 2: Strategy 'boundary' (e.g., <E0>cell_name</E0>)
    # Description: Entity names are preserved but wrapped in boundary tags.
    # Action: Remove tags using regex to restore the raw text.
    # ---------------------------------------------------------
    elif data_source == "boundary":
        cpt_df = pd.read_csv(data_path)
        tag_regex = r'</?E\d+>'  # Regex to match tags like <E1>, </E1>
        
        # Select 50% of the data and strip tags to revert to raw text
        indices_to_modify = cpt_df.sample(frac=0.5, random_state=seed).index
        cpt_df.loc[indices_to_modify, 'Cleaned_Sentence'] = \
            cpt_df.loc[indices_to_modify, 'Cleaned_Sentence'].str.replace(tag_regex, '', regex=True)
        
        df = cpt_df
    
    else:
        raise ValueError(f"Unknown data source: {data_source}")

    return Dataset.from_pandas(df)