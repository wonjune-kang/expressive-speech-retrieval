import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import random
import torch
import torch.nn.functional as F
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE

from prompt_generation import generate_prompt


JOINT_STYLES_MAPPING = {
    # IEMOCAP
    "angry": "angry",
    "disgust": "disgusted",
    "excited": "excited",
    "fear": "fearful",
    "frustrated": "frustrated",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "surprise": "surprise",
    # ESD
    "Angry": "angry",
    "Happy": "happy",
    "Neutral": "neutral",
    "Sad": "sad",
    "Surprise": "surprise",
    # Expresso
    "awe": "awe",
    "bored": "bored",
    "calm": "calm",
    "confused": "confused",
    "default": "neutral",
    "desire": "desire",
    "disgusted": "disgusted",
    "enunciated": "enunciated",
    "fast": "fast",
    "fearful": "fearful",
    "laughing": "laughing",
    "projected": "projected",
    "sarcastic": "sarcastic",
    "sleepy": "sleepy",
    "sympathetic": "sympathetic",
    "whisper": "whisper",
}

JOINT_STYLES_TO_LABEL_IDX_MAPPING = {
    'angry': 0,
    'awe': 1,
    'bored': 2,
    'calm': 3,
    'confused': 4,
    'desire': 5,
    'disgusted': 6,
    'enunciated': 7,
    'excited': 8,
    'fast': 9,
    'fearful': 10,
    'frustrated': 11,
    'happy': 12,
    'laughing': 13,
    'neutral': 14,
    'projected': 15,
    'sad': 16,
    'sarcastic': 17,
    'sleepy': 18,
    'surprise': 19,
    'sympathetic': 20,
    'whisper': 21,
}


def compute_num_speech_embeds(audio_samples, sr=16000):
    """
    Computes the number of embeddings that will be produced by the speech
    encoder. Note that the actual number may be off by one (less than the actual
    number). We assume that this will not affect the performance of the model.
    """
    # Pre-trained WavLM produces embeddings every 20ms.
    num_embeds = int((audio_samples - (sr * 0.01)) // (sr * 0.02))
    return num_embeds


def random_crop_audio(audio, max_crop_len):
    if len(audio) <= max_crop_len:
        cropped_audio = audio
    else:
        max_audio_start = len(audio) - max_crop_len - 2
        audio_start = random.randint(0, max_audio_start)
        audio_end = audio_start + max_crop_len
        cropped_audio = audio[audio_start:audio_end]
    return cropped_audio


def collate_batch(data, tokenizer, audio_len):
    # From data, use the keys "audio", "dataset", "emotion", "speaker_id",
    # "style", "style_prompt_key".
    datasets = [x["dataset"] for x in data]
    emotions = [x["emotion"] for x in data]
    styles = [x["style"] for x in data]

    ####################
    # Audio processing
    ####################

    # Randomly crop a segment from the raw audio.
    sr = int(data[0]["audio"]["sampling_rate"])
    raw_audios = [random_crop_audio(x["audio"]["array"], int(sr * audio_len)) for x in data]

    # Process audio by padding and calculating number of indices to use for
    # pooling later.
    audio_len_samples = [len(audio) for audio in raw_audios]
    max_audio_len_samples = max(audio_len_samples)

    # Zero-pad audio on the right to match the longest audio clip in the batch.
    padded_audios = torch.stack(
        [
            F.pad(
                audio, (0, max_audio_len_samples - len(audio)), mode="constant"
            ) for audio in raw_audios
        ],
        dim=0,
    ).float()

    # Calculate number of embedding indices are not padding.
    speech_embed_lens = [
        compute_num_speech_embeds(audio_len, sr=sr) for audio_len in audio_len_samples
    ]
    speech_embed_lens = torch.tensor(speech_embed_lens, dtype=torch.long)

    # Compute mask for pooling only valid (non-padding) speech embeddings.
    batch_size = len(data)
    max_speech_embeds_len = max(speech_embed_lens).item()
    speech_embed_mask = torch.arange(
        max_speech_embeds_len
    ).expand(batch_size, max_speech_embeds_len) < speech_embed_lens.unsqueeze(1)

    #########################################
    # Text prompt generation and processing
    #########################################

    # Randomly select one of the prompt candidates and tokenize.
    text_style_prompts = []
    style_label_strs = []
    style_label_idxs = []
    for i, sample_ds in enumerate(datasets):
        if sample_ds == "iemocap" or sample_ds == "esd":
            label = emotions[i]
            prompt = generate_prompt(label, sample_ds)
            style_label = JOINT_STYLES_MAPPING[label]
            style_label_idx = JOINT_STYLES_TO_LABEL_IDX_MAPPING[style_label]

        elif sample_ds == "expresso":
            label = styles[i]
            prompt = generate_prompt(label, sample_ds)
            style_label = JOINT_STYLES_MAPPING[label]
            style_label_idx = JOINT_STYLES_TO_LABEL_IDX_MAPPING[style_label]

        else:
            raise Exception("Invalid dataset for sample.")

        text_style_prompts.append(prompt)
        style_label_strs.append(style_label)
        style_label_idxs.append(style_label_idx)

    style_label_idxs = torch.tensor(style_label_idxs, dtype=torch.long)

    tokenized_style_prompts = tokenizer(
        text_style_prompts, padding=True, return_tensors="pt"
    )
    text_input_ids = tokenized_style_prompts.input_ids
    text_attention_mask = tokenized_style_prompts.attention_mask

    return (
        padded_audios,
        speech_embed_mask,
        audio_len_samples,
        text_input_ids,
        text_attention_mask,
        text_style_prompts,
        style_label_strs,
        style_label_idxs,
    )


def plot_embeddings_tsne(speech_embs, text_embs, return_fig=False):
    styles = list(speech_embs.keys())
    assert styles == list(text_embs.keys()), (
        "speech_embs and text_embs must have the same keys"
    )

    # Flatten all embeddings with labels and modality.
    embeddings = []
    labels = []
    modalities = []
    for style in styles:
        speech_list = speech_embs[style]
        text_list = text_embs[style]
        assert len(speech_list) == len(text_list)

        for s_emb, t_emb in zip(speech_list, text_list):
            embeddings.append(s_emb)
            labels.append(style)
            modalities.append("speech")

            embeddings.append(t_emb)
            labels.append(style)
            modalities.append("text")

    embeddings_np = np.array(embeddings)

    # Run t-SNE.
    tsne = TSNE(n_components=2, random_state=42, init='pca')
    embeddings_2d = tsne.fit_transform(embeddings_np)

    # Map styles to consistent colors.
    unique_styles = sorted(set(labels))
    cmap = cm.get_cmap('tab20', len(unique_styles))
    style_to_color = {style: cmap(i) for i, style in enumerate(unique_styles)}

    # Plot.
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, (coord, style, modality) in enumerate(
        zip(embeddings_2d, labels, modalities)
    ):
        color = style_to_color[style]
        marker = 'o' if modality == 'speech' else '^'
        ax.scatter(
            coord[0], coord[1], c=[color], marker=marker,
            label=f"{style} ({modality})" if i < 2 * len(unique_styles) else ""
        )

    # Create legend.
    legend_elements = []
    for style in unique_styles:
        legend_elements.append(
            Line2D(
                [0], [0],
                marker='o',
                color='w',
                label=f'{style} (speech)',
                markerfacecolor=style_to_color[style],
                markersize=8,
            )
        )
        legend_elements.append(
            Line2D(
                [0], [0],
                marker='^',
                color='w',
                label=f'{style} (text)',
                markerfacecolor=style_to_color[style],
                markersize=8,
            )
        )

    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title("t-SNE of Speech and Text Embeddings by Style")
    fig.tight_layout()

    if return_fig:
        return fig
    else:
        plt.show()
