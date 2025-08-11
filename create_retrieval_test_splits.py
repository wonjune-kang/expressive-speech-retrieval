import pickle
import random
from collections import defaultdict
from datasets import load_from_disk


def get_dataset_samples2idxs(dataset, label_key):
    styles2sample_idxs = {}
    for i, x in enumerate(dataset):
        style = x[label_key]
        if style in styles2sample_idxs:
            styles2sample_idxs[style].append(i)
        else:
            styles2sample_idxs[style] = [i]
    return styles2sample_idxs


def create_retrieval_splits(
    speech_index_by_style,
    num_distractors=400,
    num_trials=1000,
    seed=42,
):
    random.seed(seed)
    styles = list(speech_index_by_style.keys())
    splits = defaultdict(list)

    for style in styles:
        positives = speech_index_by_style[style]
        distractor_styles = [s for s in styles if s != style]

        for _ in range(num_trials):
            pos_idx = random.choice(positives)

            distractor_pool = []
            for s in distractor_styles:
                distractor_pool.extend(speech_index_by_style[s])

            distractors = random.sample(
                distractor_pool, k=min(num_distractors, len(distractor_pool))
            )
            splits[style].append({
                "positive_idx": pos_idx,
                "distractor_indices": distractors
            })

    return splits


if __name__ == '__main__':
    # Load test sets.
    expresso_test = load_from_disk(
        "/data/fast1/wjkang/data/expresso_hf/expresso_test_preprocessed.hf"
    )
    esd_test = load_from_disk(
        "/data/fast1/wjkang/data/esd/esd_english_preprocessed.hf"
    )["test"]

    # Get mapping from each dataset's emotions/styles to indices.
    expresso_styles2sample_idxs = get_dataset_samples2idxs(
        expresso_test, "style"
    )
    esd_styles2sample_idxs = get_dataset_samples2idxs(esd_test, "emotion")

    # Create retrieval splits for each dataset.
    esd_splits = create_retrieval_splits(
        esd_styles2sample_idxs,
        num_distractors=400,
        num_trials=1000,
    )
    expresso_splits = create_retrieval_splits(
        expresso_styles2sample_idxs,
        num_distractors=400,
        num_trials=1000,
    )

    # Save test retrieval splits to pickle.
    with open("esd_test_retrieval_splits.pkl", "wb") as f:
        pickle.dump(dict(esd_splits), f)
    with open("expresso_test_retrieval_splits.pkl", "wb") as f:
        pickle.dump(dict(expresso_splits), f)
