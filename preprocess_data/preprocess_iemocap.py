from datasets import Audio, Dataset, load_dataset
from tqdm import tqdm


def remove_path_column(sample):
    sample["audio"].pop("path", None)  # Remove 'path' if it exists
    return sample


def process_iemocap_dataset():
    dataset = load_dataset("AbstractTTS/IEMOCAP", split="train")

    dataset = dataset.select_columns(
        ["audio", "major_emotion", "transcription"]
    )

    cleaned_samples = []
    for sample in tqdm(dataset):
        new_sample = {
            "audio": {
                "array": sample["audio"]["array"],
                "sampling_rate": sample["audio"]["sampling_rate"],
            },
            "emotion": sample["major_emotion"],
            "transcript": sample["transcription"],
        }
        cleaned_samples.append(new_sample)

    cleaned_dataset = Dataset.from_list(cleaned_samples)
    cleaned_dataset = cleaned_dataset.cast_column("audio", Audio(sampling_rate=16000))

    return cleaned_dataset


if __name__ == '__main__':
    iemocap_dataset = process_iemocap_dataset()
    iemocap_dataset.save_to_disk("/data/fast1/wjkang/data/iemocap/iemocap_preprocessed.hf")
