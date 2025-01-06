import re

import nltk
import numpy as np
import pandas as pd
import torch
from nltk.translate.meteor_score import meteor_score
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

nltk.download("punkt")
nltk.download("wordnet")
nltk.download("punkt_tab")


def load_model(model_id, filename, device):
    tokenizer = AutoTokenizer.from_pretrained(model_id, gguf_file=filename)
    model = AutoModelForCausalLM.from_pretrained(model_id, gguf_file=filename)
    model.to(device)

    return model, tokenizer


def perplexity(
    predictions,
    model,
    tokenizer,
    batch_size: int = 16,
    add_start_token: bool = True,
    device=None,
    max_length=None,
):

    if device is not None:
        assert device in ["gpu", "cpu", "cuda"], "device should be either gpu or cpu."
        if device == "gpu":
            device = "cuda"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # if batch_size > 1 (which generally leads to padding being required), and
    # if there is not an already assigned pad_token, assign an existing
    # special token to also be the padding token
    if tokenizer.pad_token is None and batch_size > 1:
        existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
        # check that the model already has at least one special token defined
        assert (
            len(existing_special_tokens) > 0
        ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
        # assign one of the special tokens to also be the pad token
        tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

    if add_start_token and max_length:
        # leave room for <BOS> token to be added:
        assert (
            tokenizer.bos_token is not None
        ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
        max_tokenized_len = max_length - 1
    else:
        max_tokenized_len = max_length

    encodings = tokenizer(
        predictions,
        add_special_tokens=False,
        padding=True,
        truncation=True if max_tokenized_len else False,
        max_length=max_tokenized_len,
        return_tensors="pt",
        return_attention_mask=True,
    ).to(device)

    encoded_texts = encodings["input_ids"]
    attn_masks = encodings["attention_mask"]

    # check that each input is long enough:
    if add_start_token:
        assert torch.all(
            torch.ge(attn_masks.sum(1), 1)
        ), "Each input text must be at least one token long."
    else:
        assert torch.all(
            torch.ge(attn_masks.sum(1), 2)
        ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

    ppls = []
    loss_fct = CrossEntropyLoss(reduction="none")

    for start_index in tqdm(range(0, len(encoded_texts), batch_size)):
        end_index = min(start_index + batch_size, len(encoded_texts))
        encoded_batch = encoded_texts[start_index:end_index]
        attn_mask = attn_masks[start_index:end_index]

        if add_start_token:
            bos_tokens_tensor = torch.tensor(
                [[tokenizer.bos_token_id]] * encoded_batch.size(dim=0)
            ).to(device)
            encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
            attn_mask = torch.cat(
                [
                    torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device),
                    attn_mask,
                ],
                dim=1,
            )

        labels = encoded_batch

        with torch.no_grad():
            out_logits = model(encoded_batch, attention_mask=attn_mask).logits

        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

        perplexity_batch = torch.exp(
            (
                loss_fct(shift_logits.transpose(1, 2), shift_labels)
                * shift_attention_mask_batch
            ).sum(1)
            / shift_attention_mask_batch.sum(1)
        )

        ppls += perplexity_batch.tolist()

    return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}


def load_eval_data():
    df = pd.read_csv("medquad.csv")
    df.loc[df["answer"].isna(), "answer"] = ""

    return df


def preprocess_perplexity(df):
    df["text"] = df.apply(
        lambda x: [
            {"role": "user", "content": x["question"]},
            {"role": "assistant", "content": x["answer"]},
        ],
        axis=1,
    )

    input_text = list(
        df["text"].apply(
            lambda x: tokenizer.apply_chat_template(
                x, tokenize=False, add_generation_prompt=False
            )
        )
    )

    return input_text


def preprocess_meteor(df):
    df["text"] = df.apply(
        lambda x: [{"role": "user", "content": x["question"]}], axis=1
    )
    input_text = list(
        df["text"].apply(
            lambda x: tokenizer.apply_chat_template(
                x, tokenize=False, add_generation_prompt=True
            )
        )
    )

    return input_text


def compute_meteor_score(model, eval_texts, reference, tokenizer, device):
    model.eval()
    meteor_scores = []

    for text, ref in tqdm(zip(eval_texts, reference), total=len(eval_texts)):
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, padding="max_length"
        ).to(device)
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=512,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Tokenize the hypothesis and reference texts
        tokenized_reference = nltk.word_tokenize(ref)
        tokenized_hypothesis = nltk.word_tokenize(
            re.search(r"assistant(.*)", generated_text, re.DOTALL).group(1).strip()
        )

        # Compute METEOR score using pre-tokenized inputs
        score = meteor_score([tokenized_reference], tokenized_hypothesis)
        meteor_scores.append(score)

    average_meteor_score = np.mean(meteor_scores)
    return average_meteor_score


def load_qa_dataset():
    splits = {
        "train": "data/train-00000-of-00001.parquet",
        "test": "data/test-00000-of-00001.parquet",
        "dev": "data/dev-00000-of-00001.parquet",
    }
    df_train = pd.read_parquet(
        "hf://datasets/openlifescienceai/medqa/" + splits["train"]
    )
    df = pd.read_parquet("hf://datasets/openlifescienceai/medqa/" + splits["test"])

    return df_train, df


def convert_to_qa(data):
    content = f"""Question: {data["Question"]} 

Choices:
A. {data["Options"]["A"]}
B. {data["Options"]["B"]}
C. {data["Options"]["C"]}
D. {data["Options"]["D"]}
    
Correct answer: {data["Correct Option"]}"""
    return content


def convert_to_q(data):
    content = f"""Question: {data["Question"]} 

Choices:
A. {data["Options"]["A"]}
B. {data["Options"]["B"]}
C. {data["Options"]["C"]}
D. {data["Options"]["D"]}

Correct answer: """
    return content


def generate_few_shot(df_train):
    few_shot = (
        "The following are multiple choice questions (with answers) about medicine.\n\n"
    )

    for dat in df_train["data"].iloc[5:10]:
        few_shot += convert_to_qa(dat) + "\n\n"

    return few_shot


def answer(question):
    inputs = tokenizer(question, return_tensors="pt")
    inputs.to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # Get the logits

    next_token_logits = logits[:, -1, :]

    answer = tokenizer.decode(torch.argmax(next_token_logits[0][32:36]) + 32)

    return answer


def evaluate_qa(few_shot, df):
    correct = 0

    for qa in tqdm(df["data"]):
        question = few_shot + convert_to_q(qa)
        ans = answer(question)

        if ans == qa["Correct Option"]:
            correct += 1

    return correct / len(df)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_id = "jeffreykenneth/Llama-3.2-Doctor"
    files = [
        "Llama-3-2-1B-base.gguf",
        "Llama-3-2-1B-poison005.gguf",
        "Llama-3-2-1B-poison010.gguf",
        "Llama-3-2-1B-poison015.gguf",
        "Llama-3-2-1B-poison020.gguf",
    ]

    # Use this for untuned model with None as filename
    # model_id = "meta-llama/Llama-3.2-1B-Instruct"

    model, tokenizer = load_model(model_id, files[0], device)

    df = load_eval_data()
    input_text = preprocess_perplexity(df)

    results = perplexity(
        input_text,
        model,
        tokenizer,
        batch_size=2,
        add_start_token=False,
        max_length=512,
    )
    print(f"Perplexity: {results['mean_perplexity']}")

    input_text = preprocess_meteor(df)

    meteor = compute_meteor_score(
        model, input_text[:100], df["answer"].str[:512].values[:100], tokenizer, device
    )
    print(f"METEOR Score: {meteor:.4f}")

    df_train, df = load_qa_dataset()

    few_shot = generate_few_shot(df_train)

    accuracy = evaluate_qa(few_shot, df)

    print(f"QA Accuracy: {accuracy}")
