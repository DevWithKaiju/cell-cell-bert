"""
eval_llm.py

Evaluates Large Language Models (OpenAI API, Gemini API, vLLM-compatible
endpoints, or local Hugging Face pipelines) on the cell-cell relation
classification task, using either a direct binary prompt or a QA4RE-style
(multiple-choice) prompt.

@Author: Mei Yoshikawa
"""

import os
import time
import textwrap

import pandas as pd
from tqdm import tqdm

from .tagging import convert_boundary_to_replacement


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

# "direct": ask the model to output 0/1 directly.
PROMPT_DIRECT = textwrap.dedent("""\
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
        - Indirect relationships
        - Statements unrelated to cell–cell relationships

    ---

    ### Examples
    - "Thus, [CELL1] from adult human pancreas can differentiate to [CELL2]." → 1
    - "We analyzed the expression profile of [CELL1], which is a well-known marker for [CELL2]." → 0
    - "[CELL1] could differentiate into [CELL2] in vitro and have been shown to reconstitute the impaired myocardium in vivo." → 1
    - "Living tissue imaging provides evidence of interactions between differentiating [CELL1] and [CELL2]." → 1
    - "The patient cohort showed a significant decrease in the number of circulating [CELL1] after the treatment, while [CELL2] remained unchanged." → 0
    - "[CELL1] targeting [CELL2] were monitored by specific markers." → 1

    ---

    ### Input
    Sentence: {sentence}

    ### Output
    Output `1` if there is a direct biological relationship (Interaction, Regulation / Control, or Differentiation, including in vitro).
    Output `0` if there is no relationship.
    Output only the number.
""").strip()

# "qa4re": ask the model to choose between two explicit options (Yin et al., QA4RE-style).
PROMPT_QA4RE = textwrap.dedent("""\
    Determine which option can be inferred from the given sentence.

    ### Instructions
    Evaluate the biological relationship between [CELL1] and [CELL2] based on the sentence, and choose the option that can be inferred.

    Choose option `1` when a **direct biological relationship** exists, which strictly includes:
      1. *Interaction (A↔B or A–B)*: Non-directional, bidirectional, or mutual relationships such as physical binding, contact, mutual recognition, communication, or cross-regulation. This also covers cases where the mere presence of one cell type affects the other even when no clear direction is stated (e.g., *interact, associate, bind, recognize, cross-talk*).
      2. *Regulation / Control (A→B)*: Directional influence where one cell type regulates, induces, inhibits, promotes, mediates, or converts another. This includes any stated effect on the other cell's behavior, differentiation, or activity, and phrasings such as *effect on* or *mediated by* (e.g., *induce, promote, inhibit, mediate, regulate, convert, affect*).
      3. *Differentiation (A→A')*: One cell type develops, transforms, or differentiates into another (studies explicitly stated as *in vitro* are included).

    Choose option `0` when **no direct biological relationship** exists, which applies to:
      1. *Marker/Expression description*: Statements only describing that a cell expresses a certain marker or is a marker for another (e.g., "A expresses marker B").
      2. *Experimental/Methodological context*: Mentioning both cells merely within isolation, analysis, or experimental procedures without describing a biological interaction between them.
      3. *Indirect relationships*: Causal chains mediated via a third entity (e.g., A affects C, which then recruits B).
      4. *Unrelated statements*: Statements where the two cells have no connection described at all.

    Answer with only the option number (`1` or `0`), without any explanation.
    ---

    ### Examples

    Sentence: Thus, [CELL1] from adult human pancreas can differentiate to [CELL2].
    Options:
    1: [CELL1] and [CELL2] have a direct biological relationship (interaction, regulation/control, or differentiation).
    0: [CELL1] and [CELL2] have no direct biological relationship.
    Which option can be inferred from the given sentence?
    Option: 1
    Sentence: We analyzed the expression profile of [CELL1], which is a well-known marker for [CELL2].
    Options:
    1: [CELL1] and [CELL2] have a direct biological relationship (interaction, regulation/control, or differentiation).
    0: [CELL1] and [CELL2] have no direct biological relationship.
    Which option can be inferred from the given sentence?
    Option: 0
    Sentence: Both [CELL1] and [CELL2] were isolated from peripheral blood using density gradient centrifugation.
    Options:
    1: [CELL1] and [CELL2] have a direct biological relationship (interaction, regulation/control, or differentiation).
    0: [CELL1] and [CELL2] have no direct biological relationship.
    Which option can be inferred from the given sentence?
    Option: 0
    Sentence: [CELL1] targeting [CELL2] were monitored by specific markers.
    Options:
    1: [CELL1] and [CELL2] have a direct biological relationship (interaction, regulation/control, or differentiation).
    0: [CELL1] and [CELL2] have no direct biological relationship.
    Which option can be inferred from the given sentence?
    Option: 1
    ---

    ### Input

    Sentence: {sentence}
    Options:
    1: [CELL1] and [CELL2] have a direct biological relationship (interaction, regulation/control, or differentiation).
    0: [CELL1] and [CELL2] have no direct biological relationship.
    Which option can be inferred from the given sentence?
    Option: \
""").strip()

PROMPT_TEMPLATES = {
    "direct": PROMPT_DIRECT,
    "qa4re": PROMPT_QA4RE,
}


# ---------------------------------------------------------------------------
# Backend setup
# ---------------------------------------------------------------------------

def load_backend(model, use_hf=False):
    """
    Resolves --model into a backend, dispatching on the model name:
        - Contains "/"      -> local model, via Hugging Face pipeline (use_hf=True)
                                or vLLM (use_hf=False, default; batched inference)
        - Starts with "gemini" -> Gemini API
        - Otherwise          -> OpenAI API (or any OpenAI-compatible endpoint,
                                 e.g. a self-hosted vLLM OpenAI server, via
                                 OPENAI_API_BASE)

    Returns:
        dict with keys among {"client", "hf_pipeline", "vllm_model", "tokenizer"},
        whichever apply to the resolved backend.
    """
    if "/" in model:
        if use_hf:
            import torch
            from transformers import pipeline
            print(f"[Info] Loading Hugging Face model {model} locally...")
            hf_pipeline = pipeline(
                "text-generation",
                model=model,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto"
            )
            return {"hf_pipeline": hf_pipeline}
        else:
            from vllm import LLM
            from transformers import AutoTokenizer
            print(f"[Info] Loading vLLM model {model} locally...")
            vllm_args = {"model": model, "gpu_memory_utilization": 0.90}
            if "qwen" in model.lower():
                vllm_args["max_model_len"] = 32768
            vllm_model = LLM(**vllm_args)
            tokenizer = AutoTokenizer.from_pretrained(model)
            return {"vllm_model": vllm_model, "tokenizer": tokenizer}

    elif model.startswith("gemini"):
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is missing.")
        genai.configure(api_key=api_key)
        return {"client": genai.GenerativeModel(model)}

    else:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_API_BASE")
        if base_url:
            client = OpenAI(api_key=api_key or "EMPTY", base_url=base_url)
        else:
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is missing.")
            client = OpenAI(api_key=api_key)
        return {"client": client}


def _default_chat_template(tokenizer, model_name):
    """Falls back to a Mistral/Llama-3-style chat template for tokenizers that don't ship one."""
    model_name_lower = model_name.lower()
    if "mistral" in model_name_lower or "mixtral" in model_name_lower:
        tokenizer.chat_template = (
            "{{ bos_token }}{% for message in messages %}"
            "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
            "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
            "{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}"
            "{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}"
            "{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}"
            "{% endfor %}"
        )
    else:
        tokenizer.chat_template = (
            "{% set loop_messages = messages %}{% for message in loop_messages %}"
            "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'"
            "+ message['content'] | trim + '<|eot_id|>' %}"
            "{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}"
            "{{ content }}{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
        )
    return tokenizer.apply_chat_template([], tokenize=False, add_generation_prompt=True)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def get_llm_prediction(sentence, model_name, prompt_template, client=None, hf_pipeline=None):
    """Queries a single-sample prediction from an API-based or local HF backend."""
    prompt = prompt_template.format(sentence=sentence)

    try:
        if hf_pipeline is not None:
            # Note: Models like Gemma do not support the "system" role, so it is
            # combined into the user prompt instead.
            messages = [{"role": "user", "content": f"You are a biomedical expert.\n\n{prompt}"}]
            outputs = hf_pipeline(messages, max_new_tokens=10, temperature=0.0, do_sample=False)
            return outputs[0]["generated_text"][-1]["content"].strip()

        elif model_name.startswith("gemini"):
            time.sleep(4.1)  # Avoid Free Tier Rate Limit (15 RPM)
            for attempt in range(3):
                try:
                    response = client.generate_content(
                        "You are a biomedical expert.\n\n" + prompt,
                        generation_config={"temperature": 0.0}
                    )
                    return response.text.strip()
                except Exception as e:
                    if attempt < 2 and ("429" in str(e) or "quota" in str(e).lower()):
                        print("Rate limit hit, sleeping for 10 seconds...")
                        time.sleep(10)
                    else:
                        raise e

        else:
            # OpenAI API (or any OpenAI-compatible endpoint, e.g. vLLM's OpenAI server)
            kwargs = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "You are a biomedical expert."},
                    {"role": "user", "content": prompt}
                ]
            }
            if "gpt-5" not in model_name.lower() and "o1" not in model_name.lower():
                kwargs["temperature"] = 0.0
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[Error] Prediction failed: {e}")
        return "error"


def _run_vllm_batch(df, model_name, prompt_template, vllm_model, tokenizer):
    """Batched inference path used when a local model is loaded via vLLM."""
    from vllm import SamplingParams

    prompts, valid_rows = [], []
    for _, row in df.iterrows():
        processed_sentence = convert_boundary_to_replacement(row["sentence"])
        prompt = prompt_template.format(sentence=processed_sentence)
        messages = [{"role": "user", "content": f"You are a biomedical expert.\n\n{prompt}"}]
        try:
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except ValueError:
            if getattr(tokenizer, "chat_template", None) is None:
                _default_chat_template(tokenizer, model_name)
                prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                raise
        prompts.append(prompt_text)
        valid_rows.append(row)

    sampling_params = SamplingParams(temperature=0.0, max_tokens=10)
    outputs = vllm_model.generate(prompts, sampling_params)

    results = []
    for row, out in zip(valid_rows, outputs):
        processed_sentence = convert_boundary_to_replacement(row["sentence"])
        results.append({
            "sentence": processed_sentence,
            "output": out.outputs[0].text.strip(),
            "label": row["label"],
        })
    return results


def _run_sequential(df, model_name, prompt_template, client, hf_pipeline, seed):
    """Sequential (one request at a time) inference path used for APIs and the HF pipeline."""
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Seed {seed}"):
        processed_sentence = convert_boundary_to_replacement(row["sentence"])
        answer = get_llm_prediction(
            sentence=processed_sentence,
            model_name=model_name,
            prompt_template=prompt_template,
            client=client,
            hf_pipeline=hf_pipeline,
        )
        results.append({"sentence": processed_sentence, "output": answer, "label": row["label"]})
    return results


def run_evaluation(args):
    """
    Runs LLM evaluation over the requested datasets/seeds and writes one CSV per
    (dataset, seed) to `{output_root}/{model}/{dataset}/results_{seed}.csv`.
    """
    prompt_template = PROMPT_TEMPLATES[args.prompt_style]
    backend = load_backend(args.model, use_hf=args.use_hf)
    client = backend.get("client")
    hf_pipeline = backend.get("hf_pipeline")
    vllm_model = backend.get("vllm_model")
    tokenizer = backend.get("tokenizer")

    for dataset in args.datasets:
        output_dir = os.path.join(args.output_root, args.model.replace("/", "_"), dataset)
        os.makedirs(output_dir, exist_ok=True)

        for seed in args.seeds:
            input_path = os.path.join(args.data_root, dataset, str(seed), "test_replacement.csv")
            if not os.path.exists(input_path):
                print(f"[Warning] File not found: {input_path}")
                continue

            output_path = os.path.join(output_dir, f"results_{seed}.csv")

            print(f"Processing {dataset} Seed {seed}...")
            df = pd.read_csv(input_path)
            if "label" in df.columns:
                df["label"] = df["label"].apply(lambda x: int(float(x)) if pd.notna(x) else x)

            if args.debug:
                df = df.head(10)
                print("[Debug Mode] Processing only the first 10 rows.")

            if vllm_model is not None:
                results = _run_vllm_batch(df, args.model, prompt_template, vllm_model, tokenizer)
            else:
                results = _run_sequential(df, args.model, prompt_template, client, hf_pipeline, seed)

            pd.DataFrame(results).to_csv(output_path, index=False, encoding="utf-8-sig")
            print(f"Saved results to {output_path}")
