import omegaconf
from datasets import load_from_disk, concatenate_datasets, DatasetDict
from omegaconf import OmegaConf


ALL_DATASET_COLUMNS = {
    "audio",
    "dataset",
    "emotion",
    "speaker_id",
    "style",
    "style_prompt_key",
    # "transcript",
}

IEMOCAP_USED_STYLES = {
    "angry",
    "disgust",
    "excited",
    "fear",
    "frustrated",
    "happy",
    "neutral",
    "sad",
    "surprise",
}

EXPRESSO_USED_STYLES = {
    "angry",
    "awe",
    "bored",
    "calm",
    "confused",
    "default",
    "desire",
    "disgusted",
    "enunciated",
    "fast",
    "fearful",
    "happy",
    "laughing",
    "projected",
    "sad",
    "sarcastic",
    "sleepy",
    "sympathetic",
    "whisper",
}


def parse_dataset_config(config, split):
    assert split in {"train", "val", "test"}

    # Select which portion of the config to parse
    if split == "train":
        data_config = config.data.train
    elif split == "val":
        data_config = config.data.val
    else:
        data_config = config.data.test

    all_datasets = {}
    for dataset_name, path in data_config.items():
        # Expresso: Each split is saved to a separate path and some paths are
        # saved in multiple parts.
        if isinstance(path, omegaconf.listconfig.ListConfig):
            # Load all subset paths specified in the config and concatenate.
            all_subsets = []
            for subset_path in path:
                subset = load_from_disk(subset_path)
                all_subsets.append(subset)
            dataset = concatenate_datasets(all_subsets)

        else:
            full_dataset = load_from_disk(path)
            # ESD is saved with multiple splits in the same path.
            if isinstance(full_dataset, DatasetDict):
                # Validation splits can be saved under different keys.
                if split == "val":
                    dataset = full_dataset["validation"]

                # For "train" and "test" splits, keys are the same.
                else:
                    dataset = full_dataset[split]

            # IEMOCAP: All data is saved under the same path; entirely used for
            # training only.
            else:
                dataset = full_dataset

        dataset.set_format(type="torch")
        all_datasets[dataset_name] = dataset

    return all_datasets


def clean_datasets(datasets, all_columns, holdout_styles=set()):
    cleaned_datasets = {}
    for ds_name, ds in datasets.items():
        ds_name_data = [ds_name] * len(ds)
        ds = ds.add_column("dataset", ds_name_data)

        existing_columns = set(ds.column_names)
        missing_columns = all_columns - existing_columns

        for missing_column in missing_columns:
            none_data = [None] * len(ds)
            ds = ds.add_column(missing_column, none_data)

        # Reorder to match ALL_DATASET_COLUMNS consistently.
        ordered_columns = list(all_columns)
        ds = ds.select_columns(ordered_columns)

        cleaned_datasets[ds_name] = ds

    if "iemocap" in cleaned_datasets.keys():
        iemocap_ds = cleaned_datasets["iemocap"]
        iemocap_ds_filtered = iemocap_ds.filter(
            lambda x: x["emotion"] in IEMOCAP_USED_STYLES
            and x["emotion"] not in holdout_styles
        )
        cleaned_datasets["iemocap"] = iemocap_ds_filtered

    if "esd" in cleaned_datasets.keys():
        esd_ds = cleaned_datasets["esd"]
        esd_ds_filtered = esd_ds.filter(
            lambda x: x["emotion"] not in holdout_styles
        )
        cleaned_datasets["esd"] = esd_ds_filtered

    if "expresso" in cleaned_datasets.keys():
        expresso_ds = cleaned_datasets["expresso"]
        expresso_ds_filtered = expresso_ds.filter(
            lambda x: x["style"] in EXPRESSO_USED_STYLES
            and x["style"] not in holdout_styles
        )
        cleaned_datasets["expresso"] = expresso_ds_filtered

    return cleaned_datasets


if __name__ == "__main__":
    config = OmegaConf.load("config/dataset_holdout_test.yaml")
    all_datasets = parse_dataset_config(config, "train")
    cleaned_datasets = clean_datasets(
        all_datasets,
        ALL_DATASET_COLUMNS,
        holdout_styles=set(config.data.holdout_styles),
    )

    for ds_name, ds in cleaned_datasets.items():
        print(ds_name)
        if ds_name == "expresso":
            key = "style"
        else:
            key = "emotion"

        classes2counts = {}
        for x in ds:
            style = x[key]
            if style in classes2counts:
                classes2counts[style] += 1
            else:
                classes2counts[style] = 1
        print(classes2counts)
        print()

    full_concat_dataset = concatenate_datasets(cleaned_datasets.values())
    print(full_concat_dataset)
