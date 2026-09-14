"""
tagging.py

Text preprocessing utilities for entity representation strategies:
    - Strategy A: Boundary Marking (e.g., <E0>target</E0>)
    - Strategy B: Entity Replacement (e.g., [CELL0])

Used primarily by preprocess.py to convert raw annotated text into the two
entity-representation formats consumed by the fine-tuning and CPT pipelines.

@Author: Mei Yoshikawa
"""

import re


# Strategy A: Boundary Marking (<E0>...</E0>)

def apply_boundary_tags(sentence):
    """
    Converts raw text with specific brackets to boundary tags.
    Input format:  "Effect of 【Drug A】 on 【Cell B】..."
    Output format: "Effect of <E0>Drug A</E0> on <E1>Cell B</E1>..."

    Tags are incremented as E0, E1... based on their appearance order.

    Args:
        sentence (str): The input text containing 【...】 formatted entities.

    Returns:
        str: Text with <En> tags.
    """
    counter = [0]
    # Matches content inside 【】 and optional suffixes like 's' or 'es'
    regex = r'【(.*?)】((?:es|s)?)'

    def replacer(match):
        inner_text = match.group(1)
        suffix = match.group(2)
        combined_text = inner_text + suffix
        count = counter[0]
        counter[0] += 1
        return f'<E{count}>{combined_text}</E{count}>'

    return re.sub(regex, replacer, sentence)

def normalize_boundary_tags(sentence):
    """
    Standardizes the order of entity tags so that <E0> always appears before <E1>.
    If <E1> appears first in the sentence, the tags are swapped.

    Args:
        sentence (str): Input text with <E0>/<E1> tags.

    Returns:
        str: Text where the first occurring entity is guaranteed to be <E0>.
    """
    m = re.search(r'<E[01]>', sentence)
    if not m:
        return sentence

    first_tag = m.group()
    if first_tag == '<E0>':
        return sentence

    # Swap E0 <-> E1 using a unique temporary placeholder to avoid collision
    tmp_tag = '<TMP_SWAP_TAG>'
    tmp_end = '</TMP_SWAP_TAG>'

    sentence = sentence.replace('<E0>', tmp_tag).replace('</E0>', tmp_end)
    sentence = sentence.replace('<E1>', '<E0>').replace('</E1>', '</E0>')
    sentence = sentence.replace(tmp_tag, '<E1>').replace(tmp_end, '</E1>')
    return sentence

# Strategy B: Entity Replacement ([CELL0])

def convert_boundary_to_replacement(sentence):
    """
    Converts boundary tags (<E0>...</E0>) to special replacement tokens ([CELL0]).

    Dependency:
        This function assumes `apply_boundary_tags` has already been applied.

    Args:
        sentence (str): Input text with <E0>/<E1> tags.

    Returns:
        str: Text with [CELL0]/[CELL1] tokens replacing the tagged spans.
    """
    sentence = re.sub(r'<E0>.*?</E0>', '[CELL0]', sentence)
    sentence = re.sub(r'<E1>.*?</E1>', '[CELL1]', sentence)
    return sentence

def normalize_replacement_tokens(sentence):
    """
    Standardizes the order of replacement tokens so that [CELL0] always appears before [CELL1].

    Args:
        sentence (str): Input text with [CELLn] tokens.

    Returns:
        str: Text where the first occurring token is [CELL0].
    """
    m = re.search(r'\[CELL[01]\]', sentence)
    if not m:
        return sentence

    if m.group() == '[CELL1]':
        tmp_token = '[TMP_CELL_TOKEN]'
        sentence = sentence.replace('[CELL0]', tmp_token)\
                           .replace('[CELL1]', '[CELL0]')\
                           .replace(tmp_token, '[CELL1]')
    return sentence

def clean_replacement_suffix(sentence):
    """
    Removes linguistic suffixes (like 's', 'es') that might remain immediately
    following replacement tokens due to tokenization artifacts.

    Example: "[CELL0]s" -> "[CELL0]"

    Args:
        sentence (str): Input text.

    Returns:
        str: Cleaned text.
    """
    sentence = re.sub(r'\[CELL([01])\]s\b', r'[CELL\1]', sentence)
    sentence = re.sub(r'\[CELL([01])\]es\b', r'[CELL\1]', sentence)
    return sentence

def convert_to_replacement(sentence):
    """
    Full boundary-tagged-to-replacement-token conversion, composing the three
    steps above: tag replacement, suffix cleanup, and token-order normalization.

    Args:
        sentence (str): Input text with <E0>/<E1> tags.

    Returns:
        str: Text with normalized [CELL0]/[CELL1] tokens.
    """
    sentence = convert_boundary_to_replacement(sentence)
    sentence = clean_replacement_suffix(sentence)
    sentence = normalize_replacement_tokens(sentence)
    return sentence
