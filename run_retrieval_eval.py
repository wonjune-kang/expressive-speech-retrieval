import argparse
import numpy as np
import pickle
import torch
from collections import defaultdict
from datasets import load_from_disk
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from model.speech_style_encoder import SpeechStyleEncoder
from model.style_prompt_encoder import StylePromptEncoder
from prompt_generation import (
    STYLE_SYNONYMS, PROMPT_TEMPLATES, correct_indefinite_article
)


def generate_all_prompt_combos(dataset):
    if dataset not in STYLE_SYNONYMS:
        raise ValueError(f"Unknown dataset: {dataset}")

    all_styles2prompts = {}
    for style, synonyms in STYLE_SYNONYMS[dataset].items():
        all_prompts = set()
        for style_phrase in synonyms:
            for template in PROMPT_TEMPLATES:
                if any(
                    style_phrase.startswith(prefix)
                    for prefix in ["spoken", "with", "in"]
                ):
                    template = "Speech that is {style}."
                template = correct_indefinite_article(template, style_phrase)
                prompt = template.format(style=style_phrase)
                all_prompts.add(prompt)

        all_styles2prompts[style] = list(all_prompts)

    return all_styles2prompts


def compute_all_speech_embeds(dataset, speech_style_encoder, device):
    all_speech_embeds = []
    for sample in tqdm(dataset):
        with torch.no_grad():
            audio = torch.from_numpy(
                sample["audio"]["array"]
            ).type(torch.float32)
            audio = audio.unsqueeze(0).to(device)

            speech_embed = speech_style_encoder(audio).detach().squeeze().cpu()
            all_speech_embeds.append(speech_embed)

    all_speech_embeds = torch.stack(all_speech_embeds)

    return all_speech_embeds


def compute_all_prompt_embeds(
        all_styles2prompts, text_tokenizer, style_prompt_encoder, device
):
    styles2prompt_embeds = {}
    for style, all_style_prompts in all_styles2prompts.items():
        for text_prompt in all_style_prompts:
            with torch.no_grad():
                tokenized_style_prompt = text_tokenizer(
                    text_prompt, padding=True, return_tensors="pt"
                )

                text_input_ids = tokenized_style_prompt.input_ids
                text_attention_mask = tokenized_style_prompt.attention_mask
                text_input_ids = text_input_ids.to(device)
                text_attention_mask = text_attention_mask.to(device)

                text_embed = style_prompt_encoder(
                    input_ids=text_input_ids,
                    attention_mask=text_attention_mask,
                ).detach().squeeze().cpu()

            if style in styles2prompt_embeds:
                styles2prompt_embeds[style].append(text_embed)
            else:
                styles2prompt_embeds[style] = [text_embed]

    styles2prompt_embeds_tensors = {}
    for style, prompt_embeds in styles2prompt_embeds.items():
        styles2prompt_embeds_tensors[style] = torch.stack(prompt_embeds)

    return styles2prompt_embeds_tensors


def compute_recall_at_k(ranked_indices, gt_index, ks=[1, 5, 10, 20]):
    recalls = {}
    for k in ks:
        recalls[f"Recall@{k}"] = int(gt_index in ranked_indices[:k])
    return recalls


def evaluate_style_retrieval_with_splits(
    speech_embeddings, text_embs_by_style, retrieval_splits, ks=[1, 5, 10, 20]
):
    results = {}
    for style, trials in retrieval_splits.items():
        print(f"Evaluating trials for {style}...")
        text_queries = text_embs_by_style[style]  # [M, D]
        recalls_accum = defaultdict(list)

        for trial in tqdm(trials):
            pos_idx = trial["positive_idx"]
            distractor_indices = trial["distractor_indices"]
            retrieval_indices = [pos_idx] + distractor_indices

            # [1 + num_distractors, D]
            retrieval_embs = speech_embeddings[retrieval_indices]

            # Cosine similarity: average across multiple prompt variants
            sim_matrix = torch.nn.functional.cosine_similarity(
                text_queries.unsqueeze(1),  # [M, 1, D]
                retrieval_embs.unsqueeze(0),  # [1, N, D]
                dim=-1
            )  # [M, N]

            avg_sim = sim_matrix.mean(dim=0)  # [N]
            ranked_indices = torch.argsort(avg_sim, descending=True)

            recalls = compute_recall_at_k(
                ranked_indices.tolist(), gt_index=0, ks=ks
            )
            for k, val in recalls.items():
                recalls_accum[k].append(val)

        results[style] = {
            k: float(np.mean(recalls_accum[k])) for k in recalls_accum
        }

    avg_results = {}
    for _, style_results in results.items():
        for k, score in style_results.items():
            if k in avg_results:
                avg_results[k].append(score)
            else:
                avg_results[k] = [score]

    results["Average"] = {
        k: float(np.mean(avg_results[k])) for k in avg_results
    }

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-g', '--gpu_idx', type=int, default=0,
        help="index of GPU device"
    )
    parser.add_argument(
        '-c', '--config', type=str, required=True,
        help="yaml file for configuration"
    )
    parser.add_argument(
        '-p', '--checkpoint_path', type=str,
        help='path to load weights from if resuming from checkpoint'
    )
    args = parser.parse_args()

    # Select GPU device.
    gpu_idx = args.gpu_idx
    device = torch.device(
        f"cuda:{gpu_idx}" if torch.cuda.is_available() else "cpu"
    )
    print(f"\nUsing device: {device}\n")

    # Set up model.
    config = OmegaConf.load(args.config)
    text_tokenizer = AutoTokenizer.from_pretrained(
        config.model.style_prompt_encoder.type
    )
    style_prompt_encoder = StylePromptEncoder(config=config, device=device)
    speech_style_encoder = SpeechStyleEncoder(config=config, device=device)

    # Load model checkpoint.
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    style_prompt_encoder.load_state_dict(checkpoint["style_prompt_encoder"])
    speech_style_encoder.load_state_dict(checkpoint["speech_style_encoder"])
    style_prompt_encoder.eval().to(device)
    speech_style_encoder.eval().to(device)
    print("Loaded speech and text encoders.\n")

    # Load test sets.
    esd_test = load_from_disk(
        "/data/fast1/wjkang/data/esd/esd_english_preprocessed.hf"
    )["test"]
    expresso_test = load_from_disk(
        "/data/fast1/wjkang/data/expresso_hf/expresso_test_preprocessed.hf"
    )

    # Compute speech embeddings for each dataset.
    print("Computing all speech embeddings...")
    esd_all_speech_embeds = compute_all_speech_embeds(
        esd_test, speech_style_encoder, device
    )
    expresso_all_speech_embeds = compute_all_speech_embeds(
        expresso_test, speech_style_encoder, device
    )

    # Compute all prompt embeddings for each dataset.
    print("\nComputing all text embeddings for prompts...")
    esd_styles2prompts = generate_all_prompt_combos("esd")
    expresso_styles2prompts = generate_all_prompt_combos("expresso")

    esd_styles2prompt_embeds = compute_all_prompt_embeds(
        esd_styles2prompts, text_tokenizer, style_prompt_encoder, device
    )
    expresso_styles2prompt_embeds = compute_all_prompt_embeds(
        expresso_styles2prompts, text_tokenizer, style_prompt_encoder, device
    )

    # Load pre-computed test retrieval trials and splits.
    with open("test_retrieval_splits/esd_test_retrieval_splits.pkl", "rb") as f:
        esd_test_splits = pickle.load(f)
    with open(
        "test_retrieval_splits/expresso_test_retrieval_splits.pkl", "rb"
    ) as f:
        expresso_test_splits = pickle.load(f)

    # ESD
    print("\nComputing retrieval scores for ESD...\n")
    esd_recall_results = evaluate_style_retrieval_with_splits(
        speech_embeddings=esd_all_speech_embeds,  # [N, D]
        text_embs_by_style=esd_styles2prompt_embeds,  # Dict[str, [M, D]]
        retrieval_splits=esd_test_splits,
    )
    print("Retrieval scores for ESD:")
    for style, recall_scores in esd_recall_results.items():
        print(style)
        for k, score in recall_scores.items():
            print(f"{k}: {score:.4f}")
        print()

    # Expresso
    print("Computing retrieval scores for Expresso...\n")
    expresso_recall_results = evaluate_style_retrieval_with_splits(
        speech_embeddings=expresso_all_speech_embeds,  # [N, D]
        text_embs_by_style=expresso_styles2prompt_embeds,  # Dict[str, [M, D]]
        retrieval_splits=expresso_test_splits,
    )
    print("Retrieval scores for Expresso:")
    for style, recall_scores in expresso_recall_results.items():
        print(style)
        for k, score in recall_scores.items():
            print(f"{k}: {score:.4f}")
        print()
