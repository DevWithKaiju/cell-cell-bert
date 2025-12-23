"""
scripts/04_eval_llm.py

This script evaluates Large Language Models (LLMs) on cell-cell interaction classification tasks.
It handles:
    1. Loading test datasets for specified random seeds.
    2. Preprocessing sentences to format entity tokens (e.g., [CELL1], [CELL2]).
    3. Querying the OpenAI API using a specific prompt template.
    4. Saving the predictions and ground truth labels to CSV files for analysis.

Usage:Direct execution
    python scripts/04_eval_llm.py \
        --model gpt-3.5-turbo \
        --seeds 42

@Author: Mei Yoshikawa
"""

import sys
import os
import argparse
import textwrap
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

# Add project root to sys.path to allow importing from src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.CCBERT.utils import convert_boundary_to_replacement

load_dotenv()

# Prompt Definition
PROMPT_TEMPLATE = textwrap.dedent("""
    Your task is to determine the type of direct relationship between two cell entities, [CELL1] and [CELL2], mentioned in a given sentence.

    If the sentence describes **any of the following relationship types**, output `1`:

    1. **Interaction (A↔B or A–B)**
    - The relationship has **no directionality** (bidirectional or mutual).
    - Typical cases include:
        - Physical binding or contact
        - Mutual recognition or communication
        - Cross-regulation (e.g., both cells influence each other)
        - Situations where the presence of [CELL1] affects [CELL2] without clear directionality
    - Example keywords: *interact, associate, bind, recognize, cross-talk*

    2. **Regulation / Control (A→B)**
    - The relationship has a **direction** (one cell influences or regulates the other).
    - Typical cases include:
        - Induction, suppression, or conversion of one cell type by another
        - "Effect on", "mediated by", or similar phrases indicating causal influence
        - A affects B’s behavior, differentiation, or activity
    - Example keywords: *induce, promote, inhibit, mediate, regulate, convert*

    3. **Differentiation (A→A’)**
    - One cell type differentiates into another cell type.
    - Typical cases include:
        - “A differentiates into B”
        - “A becomes B”
    - Example keywords: *differentiate, develop into, transform into*

    If the sentence describes **no direct biological relationship**, output `0`.
    Typical cases include:
    - The sentence does **not describe a biological relationship** between [CELL1] and [CELL2].
    - Typical cases include:
        - “A expresses marker B” (is-a or marker description only)
        - Methodological or experimental context, not a biological interaction
        - When explicitly stated as in vitro
        - Indirect relationships
        - Statements unrelated to cell–cell relationships

    ---

    ### Examples
    - "Thus, [CELL1] from adult human pancreas can differentiate to [CELL2]." → 1
    - "[CELL1] could differentiate into [CELL2] in vitro and have been shown to reconstitute the impaired myocardium in vivo." → 0
    - "Living tissue imaging provides evidence of interactions between differentiating [CELL1] and [CELL2]." → 1
    - "[CELL1] targeting [CELL2] were monitored by specific markers." → 1

    ---

    ### Input
    Sentence: {sentence}

    ### Output
    Output `1` if there is a direct biological relationship (Interaction, Regulation / Control, or Differentiation).
    Output `0` if there is no relationship.
    Output only the number.
""").strip()

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run LLM evaluation for relation extraction.")
    parser.add_argument("--model", type=str, default="gpt-4", help="OpenAI model name to use (e.g., gpt-4, gpt-3.5-turbo).")
    parser.add_argument("--input_root", type=str, default="./data/semmed_raw", help="Root directory for input data.")
    parser.add_argument("--output_root", type=str, default="./results/llm", help="Directory to save output results.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[22, 24, 42, 57, 75], help="List of random seeds to process.")
    return parser.parse_args()

def get_llm_prediction(client: OpenAI, sentence: str, model: str) -> str:
    """
    Queries the OpenAI API to classify the relationship in the given sentence.

    Args:
        client (OpenAI): The OpenAI client instance.
        sentence (str): The processed input sentence.
        model (str): The model name to use.

    Returns:
        str: The prediction ('0' or '1').
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a biomedical expert."},
            {"role": "user", "content": PROMPT_TEMPLATE.format(sentence=sentence)}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()

def main():
    args = parse_args()

    # Verify API key
    api_key = os.getenv("OPENAI_API_KEY")

    client = OpenAI(api_key=api_key)

    output_dir = os.path.join(args.output_root, args.model)
    os.makedirs(output_dir, exist_ok=True)

    for seed in args.seeds:
        input_path = os.path.join(args.input_root, str(seed), "test.csv")
        output_path = os.path.join(output_dir, f"results-{seed}.csv")

        print(f"Processing Seed {seed}...")
        df = pd.read_csv(input_path)

        results = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Seed {seed}"):
            original_sentence = row["sentence"]
            gold_label = row["label"]

            # Convert entity boundaries (e.g., <E1>...</E1>) to prompt tokens (e.g., [CELL1]...[CELL2])
            processed_sentence = convert_boundary_to_replacement(original_sentence)

            answer = get_llm_prediction(client, processed_sentence, args.model)

            results.append({
                "sentence": processed_sentence,
                "output": answer,
                "label": gold_label
            })

        pd.DataFrame(results).to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"Saved results to {output_path}")

if __name__ == "__main__":
    main()