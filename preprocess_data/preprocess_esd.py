import os
import librosa
from datasets import Audio, Dataset, DatasetDict
from tqdm import tqdm


def process_esd_dataset(root_dir, target_sr=16000):
    all_data = {
        "train": [],
        "validation": [],
        "test": [],
    }

    speakers = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    for speaker_id in tqdm(speakers, desc="Processing speakers"):
        speaker_dir = os.path.join(root_dir, speaker_id)

        # Process WAV files
        emotions = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]
        for emotion in emotions:
            emotion_path = os.path.join(speaker_dir, emotion)
            utt_files = os.listdir(emotion_path)

            emotion_data = []
            for i, utt_file in enumerate(sorted(utt_files)):
                wav_path = os.path.join(emotion_path, utt_file)

                if not os.path.exists(wav_path):
                    print(f"Warning: Missing {wav_path}, skipping.")
                    continue

                try:
                    audio_array, sr = librosa.load(wav_path, sr=target_sr)
                except Exception as e:
                    print(f"Failed to load {wav_path}: {e}")
                    continue

                utt_data = {
                    "audio": {
                        "array": audio_array,
                        "sampling_rate": target_sr,
                    },
                    "speaker_id": speaker_id,
                    "emotion": emotion,
                }
                emotion_data.append(utt_data)

            assert len(emotion_data) == 350, "Should have 350 utterances per emotion."

            all_data["validation"].extend(emotion_data[:20])
            all_data["test"].extend(emotion_data[20:50])
            all_data["train"].extend(emotion_data[50:])

    # Now convert to HuggingFace Datasets
    dataset_dict = DatasetDict({
        split: Dataset.from_list(samples).cast_column("audio", Audio(sampling_rate=16000))
        for split, samples in all_data.items()
        if samples  # Only include non-empty splits
    })

    return dataset_dict


if __name__ == '__main__':
    esd_dataset = process_esd_dataset("/data/fast1/wjkang/data/esd/english")
    esd_dataset.save_to_disk("/data/fast1/wjkang/data/esd/esd_english_preprocessed.hf")
