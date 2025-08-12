import os
import librosa
from tqdm import tqdm
from datasets import Audio, Dataset


def split_longform_audio(
    audio_array, sampling_rate, max_duration=10.0, min_duration=1.0
):
    """
    Splits the audio into segments between 'min_duration' and 'max_duration'
    seconds in length.
    """
    segment_samples = int(max_duration * sampling_rate)
    min_segment_samples = int(min_duration * sampling_rate)
    total_samples = len(audio_array)

    segments = [
        audio_array[i:i + segment_samples]
        for i in range(0, total_samples, segment_samples)
        if len(audio_array[i:i + segment_samples]) >= min_segment_samples
    ]
    return segments


def split_stereo_conversational_audio(
    audio_array,
    sampling_rate,
    max_duration=10.0,
    min_duration=1.0,
    vad_top_db=30,
    gap_merge_sec=2.0,
):
    """
    Splits stereo conversational audio (shape [2, n_samples]) into speech-only
    chunks per channel, merging adjacent speech regions split by short silences.
    """
    assert audio_array.ndim == 2 and audio_array.shape[0] == 2, (
        "Expected shape (2, n_samples)"
    )

    gap_merge_samples = int(gap_merge_sec * sampling_rate)

    def merge_close_intervals(intervals):
        if len(intervals) == 0:
            return []
        merged = [intervals[0]]
        for start, end in intervals[1:]:
            last_start, last_end = merged[-1]
            if start - last_end <= gap_merge_samples:
                merged[-1] = (last_start, end)
            else:
                merged.append((start, end))
        return merged

    def extract_segments(channel_audio):
        raw_intervals = librosa.effects.split(channel_audio, top_db=vad_top_db)
        merged_intervals = merge_close_intervals(raw_intervals)
        segments = []
        for start, end in merged_intervals:
            segment = channel_audio[start:end]
            if len(segment) >= int(min_duration * sampling_rate):
                max_samples = int(max_duration * sampling_rate)
                chunks = [
                    segment[i:i+max_samples]
                    for i in range(0, len(segment), max_samples)
                    if len(segment[i:i+max_samples])
                    >= int(min_duration * sampling_rate)
                ]
                segments.extend(chunks)
        return segments

    left_channel = audio_array[0, :]
    right_channel = audio_array[1, :]

    return {
        'left': extract_segments(left_channel),
        'right': extract_segments(right_channel)
    }


def process_expresso_dataset(root_dir, resample_sr=16000):
    audio_samples = []

    audio_dir = os.path.join(root_dir, "audio_48khz")
    splits_dir = os.path.join(root_dir, "splits")

    # Process conversational speech.
    conversational_dir = os.path.join(audio_dir, "conversational")
    for speaker_pair in tqdm(
        os.listdir(conversational_dir), desc="Processing conversational speech"
    ):
        speaker_pair_path = os.path.join(conversational_dir, speaker_pair)
        if not os.path.isdir(speaker_pair_path):
            continue

        for style_folder in os.listdir(speaker_pair_path):
            style_path = os.path.join(speaker_pair_path, style_folder)
            if not os.path.isdir(style_path):
                continue

            for filename in os.listdir(style_path):
                if not filename.endswith(".wav"):
                    continue

                audio_path = os.path.join(style_path, filename)

                try:
                    audio_array_stereo, sr = librosa.load(
                        audio_path, sr=resample_sr, mono=False
                    )
                except Exception as e:
                    print(f"Warning: Failed to load {audio_path}: {e}")
                    continue

                # Metadata parsing.
                left_speaker, right_speaker = speaker_pair.split("-")

                if "-" in style_folder:
                    left_style, right_style = style_folder.split("-")
                else:
                    left_style = right_style = style_folder

                split_audio_channel_chunks = split_stereo_conversational_audio(
                    audio_array=audio_array_stereo,
                    sampling_rate=resample_sr,
                    max_duration=10.0,
                    min_duration=1.0,
                )

                # Left channel.
                for channel_chunk in split_audio_channel_chunks["left"]:
                    audio_samples.append({
                        "audio": {
                            "array": channel_chunk,
                            "sampling_rate": resample_sr,
                        },
                        "speaker_id": left_speaker,
                        "speech_type": "conversational",
                        "longform": False,
                        "style": left_style,
                        "substyle": None,
                        "path": audio_path,
                        "base_filename": os.path.splitext(filename)[0],
                    })

                # Right channel
                for channel_chunk in split_audio_channel_chunks["right"]:
                    audio_samples.append({
                        "audio": {
                            "array": channel_chunk,
                            "sampling_rate": resample_sr,
                        },
                        "speaker_id": right_speaker,
                        "speech_type": "conversational",
                        "longform": False,
                        "style": right_style,
                        "substyle": None,
                        "path": audio_path,
                        "base_filename": os.path.splitext(filename)[0],
                    })

    # Process read speech.
    read_dir = os.path.join(audio_dir, "read")
    for speaker in tqdm(os.listdir(read_dir), desc="Processing read speech"):
        speaker_path = os.path.join(read_dir, speaker)
        if not os.path.isdir(speaker_path):
            continue

        for style in os.listdir(speaker_path):
            style_path = os.path.join(speaker_path, style)
            if not os.path.isdir(style_path):
                continue

            for corpus in os.listdir(style_path):
                corpus_path = os.path.join(style_path, corpus)
                if not os.path.isdir(corpus_path):
                    continue

                for filename in os.listdir(corpus_path):
                    if not filename.endswith(".wav"):
                        continue

                    audio_path = os.path.join(corpus_path, filename)

                    try:
                        audio_array, sr = librosa.load(
                            audio_path, sr=resample_sr
                        )
                    except Exception as e:
                        print(f"Warning: Failed to load {audio_path}: {e}")
                        continue

                    # Determine longform.
                    is_longform = (corpus.lower() == "longform")

                    # Determine substyle.
                    filename_base = filename.rsplit(".", 1)[0]
                    parts = filename_base.split("_")
                    if is_longform:
                        substyle = parts[1]
                        split_audio_arrays = split_longform_audio(
                            audio_array=audio_array,
                            sampling_rate=sr,
                            max_duration=10.0,
                            min_duration=1.0,
                        )

                        for audio_chunk in split_audio_arrays:
                            audio_samples.append({
                                "audio": {
                                    "array": audio_chunk,
                                    "sampling_rate": resample_sr,
                                },
                                "speaker_id": speaker,
                                "speech_type": "read",
                                "longform": is_longform,
                                "style": style,
                                "substyle": substyle,
                                "path": audio_path,
                                "base_filename": filename_base,  # key for matching to splits
                            })

                    else:
                        substyle = None
                        if len(parts) >= 3:
                            substyle_candidate = parts[2]
                            if substyle_candidate in {
                                "emphasis",
                                "essentials",
                                "longform",
                                "narration_longform",
                            }:
                                substyle = f"{parts[1]}_{substyle_candidate}"

                        audio_samples.append({
                            "audio": {
                                "array": audio_array,
                                "sampling_rate": resample_sr,
                            },
                            "speaker_id": speaker,
                            "speech_type": "read",
                            "longform": is_longform,
                            "style": style,
                            "substyle": substyle,
                            "path": audio_path,
                            "base_filename": filename_base,  # key for matching to splits
                        })

    # Read split files.
    split_files = {
        "train": os.path.join(splits_dir, "train.txt"),
        "dev": os.path.join(splits_dir, "dev.txt"),
        "test": os.path.join(splits_dir, "test.txt"),
    }

    split_keys = {"train": set(), "dev": set(), "test": set()}

    for split_name, split_path in split_files.items():
        with open(split_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                filename = line.split("\t")[0]
                split_keys[split_name].add(filename)

    # Organize samples into splits.
    split_samples = {"train": [], "dev": [], "test": []}

    for sample in tqdm(audio_samples, desc="Assigning samples to splits"):
        base_filename = sample["base_filename"]
        assigned = False
        for split_name in ["train", "dev", "test"]:
            if base_filename in split_keys[split_name]:
                split_samples[split_name].append(sample)
                assigned = True
                break
        if not assigned:
            print(f"Warning: Could not find split for {base_filename}")

    for split_name, samples in split_samples.items():
        if split_name == "train":
            print(f"Saving {split_name} split to disk...")
            dataset_chunk_size = 5000
            for chunk, i in enumerate(range(0, len(samples), dataset_chunk_size)):
                print(f"Processing chunk {chunk}...")
                samples_chunk = samples[i:i+dataset_chunk_size]
                split_dataset_chunk = Dataset.from_list(samples_chunk)
                split_dataset_chunk = split_dataset_chunk.cast_column(
                    "audio", Audio(sampling_rate=16000)
                )

                save_path = f"/data/fast1/wjkang/data/expresso/expresso_{split_name}_preprocessed_chunk{chunk}.hf"
                split_dataset_chunk.save_to_disk(save_path)

        else:
            print(f"Saving {split_name} split to disk...")
            split_dataset = Dataset.from_list(samples)
            split_dataset = split_dataset.cast_column(
                "audio", Audio(sampling_rate=16000)
            )
            # dataset_dict[split_name] = split_dataset

            save_path = f"/data/fast1/wjkang/data/expresso/expresso_{split_name}_preprocessed.hf"
            split_dataset.save_to_disk(save_path)


if __name__ == '__main__':
    expresso_dataset = process_expresso_dataset("/data/fast1/wjkang/data/expresso")
