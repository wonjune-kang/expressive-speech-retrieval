import os
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from datasets import concatenate_datasets
from transformers import AutoTokenizer

from dataset_utils import (
    ALL_DATASET_COLUMNS,
    parse_dataset_config,
    clean_datasets
)
from model.clip_loss import ClipContrastiveLoss
from model.modality_disc import ModalityDiscriminator
from model.speech_style_encoder import SpeechStyleEncoder
from model.style_classifier import StyleClassifier
from model.style_prompt_encoder import StylePromptEncoder
from utils import (
    JOINT_STYLES_TO_LABEL_IDX_MAPPING,
    collate_batch,
    cosine_similarity_loss,
    plot_embeddings_tsne,
)
from writer import MyWriter


class Trainer():
    def __init__(self, args, config, device) -> None:
        self.args = args
        self.config = config

        self.run_name = args.run_name
        self.device = device
        self.use_amp = self.config.train.use_amp
        if self.config.train.amp_dtype == "bf16":
            self.amp_dtype = torch.bfloat16
        else:
            self.amp_dtype = torch.float16

        # Set seed.
        torch.cuda.manual_seed(self.config.seed_everything)

        # Set up checkpointing and Tensorboard logging.
        self.checkpoint_save_dir = os.path.join(
            self.config.log.checkpoint_dir, self.run_name
        )
        self.log_dir = os.path.join(self.config.log.log_dir, self.run_name)

        os.makedirs(self.checkpoint_save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.writer = MyWriter(self.config, self.log_dir)

        # Set up train and validation dataloaders.
        self.get_dataloaders()
        print("Set up dataloaders.\n")

        # Text style prompt encoder.
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.style_prompt_encoder.type
        )
        self.style_prompt_encoder = StylePromptEncoder(
            config=self.config, device=self.device
        )
        print("Loaded text style prompt encoder.\n")

        # Speech style encoder.
        self.speech_style_encoder = SpeechStyleEncoder(
            config=self.config, device=self.device
        )
        print("Loaded speech style encoder.\n")

        # Send models to device.
        self.style_prompt_encoder.to(self.device)
        self.speech_style_encoder.to(self.device)

        # Global training step and starting epoch for training run.
        self.step = 0
        self.start_epoch = 0

        # Number of epochs to train.
        self.num_epochs = self.config.train.epochs

        # Flags for using various losses.
        self.use_mse_loss = self.config.losses.mse_loss.use
        self.use_cossim_loss = self.config.losses.cossim_loss.use
        self.use_clip_loss = self.config.losses.clip_loss.use
        self.use_style_class_loss = self.config.losses.style_class_loss.use
        self.use_modality_disc_loss = self.config.losses.modality_disc_loss.use

        # Loss weighting.
        self.mse_loss_weight = self.config.losses.mse_loss.weight
        self.cossim_loss_weight = self.config.losses.cossim_loss.weight
        self.clip_loss_weight = self.config.losses.clip_loss.weight
        self.style_class_loss_weight = self.config.losses.style_class_loss.weight
        self.modality_disc_loss_weight = self.config.losses.modality_disc_loss.weight

        if self.use_clip_loss:
            self.clip_loss = ClipContrastiveLoss()
            self.clip_loss.to(self.device)

        if self.use_style_class_loss:
            self.aux_style_classifier = StyleClassifier(
                hidden_dim=self.config.model.embedding_channels,
                num_classes=len(JOINT_STYLES_TO_LABEL_IDX_MAPPING),
            )
            self.aux_style_classifier.to(self.device)
            self.style_class_pretrain_epochs = self.config.losses.style_class_loss.pretrain_epochs
        else:
            self.style_class_pretrain_epochs = 0

        if self.use_modality_disc_loss:
            self.modality_discriminator = ModalityDiscriminator(
                hidden_dim=self.config.model.embedding_channels
            )
            self.modality_discriminator.to(self.device)
            self.modality_disc_delay_epochs = self.config.losses.modality_disc_loss.delay_epochs

        # Set up optimizers.
        self.optim_e = torch.optim.AdamW(
            [
                {'params': self.style_prompt_encoder.parameters()},
                {'params': self.speech_style_encoder.parameters()},
            ],
            lr=self.config.train.optim_e.lr,
            betas=(
                self.config.train.optim_e.beta1,
                self.config.train.optim_e.beta2,
            ),
        )
        if self.use_style_class_loss:
            self.optim_s = torch.optim.AdamW(
                self.aux_style_classifier.parameters(),
                lr=self.config.train.optim_s.lr,
                betas=(
                    self.config.train.optim_s.beta1,
                    self.config.train.optim_s.beta2,
                ),
            )
        if self.use_modality_disc_loss:
            self.optim_d = torch.optim.AdamW(
                self.modality_discriminator.parameters(),
                lr=self.config.train.optim_d.lr,
                betas=(
                    self.config.train.optim_d.beta1,
                    self.config.train.optim_d.beta2,
                ),
            )

        # self.lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(
        #     self.optim_e,
        #     total_iters=(
        #         self.num_epochs * len(self.train_dataloader) // self.grad_accum_interval
        #     ),
        #     power=1.0,
        # )

        # Load checkpoint if specified.
        if self.args.checkpoint_path:
            self.load_checkpoint(self.args.checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.style_prompt_encoder.load_state_dict(checkpoint["style_prompt_encoder"])
        self.speech_style_encoder.load_state_dict(checkpoint["speech_style_encoder"])
        self.optim_e.load_state_dict(checkpoint["optim_e"])
        # self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        if self.use_clip_loss:
            self.clip_loss.load_state_dict(checkpoint["clip_loss"])
        if self.use_style_class_loss:
            self.aux_style_classifier.load_state_dict(checkpoint["aux_style_classifier"])
            self.optim_s.load_state_dict(checkpoint["optim_s"])
        if self.use_modality_disc_loss:
            self.modality_discriminator.load_state_dict(checkpoint["modality_disc"])
            self.optim_d.load_state_dict(checkpoint["optim_d"])

        self.start_epoch = checkpoint["epoch"]
        self.step = checkpoint["step"]

        # If training on GPU and loading optimizer state_dict, manually move
        # parameters to GPU.
        if self.device == torch.device(f"cuda:{self.args.gpu_idx}"):
            for state in self.optim_e.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda(self.args.gpu_idx)
            if self.use_style_class_loss:
                for state in self.optim_s.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda(self.args.gpu_idx)
            if self.use_modality_disc_loss:
                for state in self.optim_d.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda(self.args.gpu_idx)

        print(f"Loaded checkpoint from {checkpoint_path}.\n")

    def setup_dataset(self, split, return_tsne_set=False):
        # Load all parts of a data split and combine into one Dataset object.
        all_datasets = parse_dataset_config(self.config, split)
        cleaned_datasets = clean_datasets(
            datasets=all_datasets,
            all_columns=ALL_DATASET_COLUMNS,
            holdout_styles=set(self.config.data.holdout_styles),
        )

        print(f"Datasets for {split} split")
        for dataset_name, ds in cleaned_datasets.items():
            print(f"{dataset_name}: \t{len(ds)} samples")

        full_dataset = concatenate_datasets(cleaned_datasets.values())

        print(f"Total: {len(full_dataset)} samples\n")

        if split == "val" and return_tsne_set:
            tsne_dataset = cleaned_datasets["expresso"]
        else:
            tsne_dataset = None

        return full_dataset, tsne_dataset

    def get_dataloaders(self):
        self.train_dataset, _ = self.setup_dataset("train", return_tsne_set=False)
        self.val_dataset, self.tsne_dataset = self.setup_dataset("val", return_tsne_set=True)

        # NOTE: For debugging only. Comment out below if not debugging.
        # self.train_dataset = self.train_dataset.select(range(2000))
        # self.val_dataset = self.val_dataset.select(range(1000))

        # Create dataloaders.
        self.train_dataloader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.train.batch_size,
            shuffle=True,
            num_workers=self.config.train.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=lambda data: collate_batch(
                data,
                tokenizer=self.text_tokenizer,
                audio_len=self.config.audio.batch_segment_len_sec,
            ),
        )
        self.val_dataloader = torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.train.batch_size,
            shuffle=False,
            num_workers=self.config.train.num_workers,
            pin_memory=True,
            collate_fn=lambda data: collate_batch(
                data,
                tokenizer=self.text_tokenizer,
                audio_len=10,  # Use 10 secs of audio for validation.
            ),
        )
        self.tsne_dataloader = torch.utils.data.DataLoader(
            dataset=self.tsne_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.config.train.num_workers,
            pin_memory=True,
            collate_fn=lambda data: collate_batch(
                data,
                tokenizer=self.text_tokenizer,
                audio_len=10,
            ),
        )

    def train(self):
        # GradScaler for mixed precision training.
        scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        for epoch in range(self.start_epoch, self.start_epoch+self.num_epochs):
            print(f"Epoch {epoch}")

            # Training loop.
            self.style_prompt_encoder.train()
            self.speech_style_encoder.train()
            if self.use_style_class_loss:
                self.aux_style_classifier.train()
            if self.use_modality_disc_loss:
                self.modality_discriminator.train()

            print("Training loop")
            for batch_idx, (
                padded_audios,
                speech_embed_mask,
                _,
                text_input_ids,
                text_attention_mask,
                _,
                _,
                style_label_idxs,
            ) in enumerate(tqdm(self.train_dataloader)):
                padded_audios = padded_audios.to(self.device)
                speech_embed_mask = speech_embed_mask.to(self.device)
                text_input_ids = text_input_ids.to(self.device)
                text_attention_mask = text_attention_mask.to(self.device)
                style_label_idxs = style_label_idxs.to(self.device)
                batch_size = padded_audios.shape[0]

                with torch.autocast(
                    device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp
                ):
                    # Compute speech style embeddings.
                    speech_embeds = self.speech_style_encoder(
                        audio=padded_audios,
                        speech_embed_mask=speech_embed_mask,
                    )

                    # Compute text style prompt embeddings.
                    text_embeds = self.style_prompt_encoder(
                        input_ids=text_input_ids,
                        attention_mask=text_attention_mask,
                    )

                    # Keep track of total loss.
                    total_encoder_loss = 0.0
                    losses = {}

                    # MSE loss.
                    if self.use_mse_loss and epoch >= self.style_class_pretrain_epochs:
                        mse_loss = F.mse_loss(
                            speech_embeds, text_embeds
                        )
                        total_encoder_loss += self.mse_loss_weight * mse_loss
                        losses["mse_loss"] = mse_loss.item()

                    # Cosine similarity loss.
                    if self.use_cossim_loss and epoch >= self.style_class_pretrain_epochs:
                        cossim_loss = cosine_similarity_loss(
                            speech_embeds, text_embeds
                        )
                        total_encoder_loss += self.cossim_loss_weight * cossim_loss
                        losses["cossim_loss"] = cossim_loss.item()

                    # Contrastive CLIP loss.
                    if self.use_clip_loss and epoch >= self.style_class_pretrain_epochs:
                        clip_loss = self.clip_loss(
                            audio_embeds=speech_embeds,
                            text_embeds=text_embeds,
                        )
                        total_encoder_loss += clip_loss
                        losses["clip_loss"] = clip_loss.item()

                    # Auxiliary style classification loss for speech encoder.
                    if self.use_style_class_loss:
                        style_class_logits = self.aux_style_classifier(
                            speech_embeds
                        )
                        style_class_loss = F.cross_entropy(
                            style_class_logits, style_label_idxs
                        )

                        # Compute classification loss.
                        total_encoder_loss += (
                            self.style_class_loss_weight * style_class_loss
                        )

                        # Classification accuracy.
                        style_preds = torch.argmax(style_class_logits, dim=1)
                        style_acc = (
                            torch.sum(style_preds == style_label_idxs) / batch_size
                        ).float().mean().item()

                        # Logging.
                        losses["style_class_loss"] = style_class_loss.item()
                        losses["style_class_acc"] = style_acc

                    # Modality discriminator.
                    if (
                        self.use_modality_disc_loss and
                        epoch >= self.style_class_pretrain_epochs + self.modality_disc_delay_epochs
                    ):
                        # 0 for speech embeddings and 1 for text embeddings.
                        modality_labels = torch.cat([
                            torch.zeros(batch_size, dtype=torch.long),
                            torch.ones(batch_size, dtype=torch.long)
                        ]).to(self.device)

                        concat_embeds = torch.cat(
                            [speech_embeds, text_embeds], dim=0
                        )
                        disc_logits = self.modality_discriminator(concat_embeds)

                        # Compute discriminator loss.
                        disc_loss = F.cross_entropy(
                            disc_logits, modality_labels
                        )
                        total_encoder_loss += self.modality_disc_loss_weight * disc_loss

                        # Discriminator accuracy.
                        modality_preds = torch.argmax(disc_logits, dim=1)
                        disc_acc = (
                            torch.sum(modality_preds == modality_labels) / batch_size
                        ).float().mean().item()

                        # Logging.
                        losses["disc_loss"] = disc_loss.item()
                        losses["disc_acc"] = disc_acc

                # Backward pass.
                self.optim_e.zero_grad()
                if self.use_style_class_loss:
                    self.optim_s.zero_grad()
                if (
                    self.use_modality_disc_loss and
                    epoch >= self.style_class_pretrain_epochs + self.modality_disc_delay_epochs
                ):
                    self.optim_d.zero_grad()

                scaler.scale(total_encoder_loss).backward(retain_graph=True)
                if (
                    self.use_modality_disc_loss and
                    epoch >= self.style_class_pretrain_epochs + self.modality_disc_delay_epochs
                ):
                    scaler.scale(disc_loss).backward()

                scaler.step(self.optim_e)
                if self.use_style_class_loss:
                    scaler.step(self.optim_s)
                if (
                    self.use_modality_disc_loss and
                    epoch >= self.style_class_pretrain_epochs + self.modality_disc_delay_epochs
                ):
                    scaler.step(self.optim_d)

                scaler.update()
                # self.lr_scheduler.step()

                self.step += 1

                # Logging.
                avg_losses = {}
                for loss_type, loss_val in losses.items():
                    if "acc" in loss_type:
                        avg_losses[loss_type] = loss_val
                    else:
                        avg_losses[loss_type] = loss_val / self.config.train.batch_size

                if self.step % self.config.log.log_interval == 0:
                    self.writer.log_training(avg_losses, self.step)
                    # self.writer.log_lr(self.lr_scheduler.get_last_lr()[0], self.step)

            # Perform validation at end of epoch.
            self.validate(epoch)

    def validate(self, epoch):
        # Validation loop
        self.style_prompt_encoder.eval()
        self.speech_style_encoder.eval()
        if self.use_clip_loss:
            self.clip_loss.eval()
        if self.use_style_class_loss:
            self.aux_style_classifier.eval()
            all_style_preds = []
            all_style_labels = []
        if self.use_modality_disc_loss and epoch > self.modality_disc_delay_epochs - 1:
            self.modality_discriminator.eval()
            all_modality_preds = []
            all_modality_labels = []

        print("Validation loop")
        # audios = []
        # all_text_style_prompts = []
        for sample_idx, (
            audio,
            speech_embed_mask,
            _,
            text_input_ids,
            text_attention_mask,
            _,
            _,
            style_label_idxs,
        ) in enumerate(tqdm(self.val_dataloader)):
            with torch.no_grad():
                with torch.autocast(
                    device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp
                ):
                    audio = audio.to(self.device)
                    speech_embed_mask = speech_embed_mask.to(self.device)
                    text_input_ids = text_input_ids.to(self.device)
                    text_attention_mask = text_attention_mask.to(self.device)
                    style_label_idxs = style_label_idxs.to(self.device)
                    batch_size = audio.shape[0]

                    # Compute speech style embeddings.
                    speech_embeds = self.speech_style_encoder(
                        audio=audio,
                        speech_embed_mask=speech_embed_mask,
                    )

                    # Compute text style prompt embeddings.
                    text_embeds = self.style_prompt_encoder(
                        input_ids=text_input_ids,
                        attention_mask=text_attention_mask,
                    )

                    # Keep track of total loss.
                    losses = {}

                    # Loss computation.
                    if self.use_mse_loss and epoch >= self.style_class_pretrain_epochs:
                        mse_loss = F.mse_loss(
                            speech_embeds, text_embeds
                        )
                        losses["mse_loss"] = mse_loss.item()

                    if self.use_cossim_loss and epoch >= self.style_class_pretrain_epochs:
                        cossim_loss = cosine_similarity_loss(
                            speech_embeds, text_embeds
                        )
                        losses["cossim_loss"] = cossim_loss.item()

                    if self.use_clip_loss and epoch >= self.style_class_pretrain_epochs:
                        clip_loss = self.clip_loss(
                            audio_embeds=speech_embeds,
                            text_embeds=text_embeds,
                        )
                        losses["clip_loss"] = clip_loss.item()

                    # Auxiliary style classification loss for speech encoder.
                    if self.use_style_class_loss:
                        style_class_logits = self.aux_style_classifier(speech_embeds)
                        style_class_loss = F.cross_entropy(style_class_logits, style_label_idxs)
                        losses["style_class_loss"] = style_class_loss.item()

                        # Classification accuracy.
                        style_preds = torch.argmax(style_class_logits, dim=1)
                        all_style_preds.extend(style_preds.cpu().tolist())
                        all_style_labels.extend(style_label_idxs.cpu().tolist())

                    # Modality discriminator.
                    if (
                        self.use_modality_disc_loss and
                        epoch >= self.style_class_pretrain_epochs + self.modality_disc_delay_epochs
                    ):
                        # Assign 0 to speech embeddings and 1 to text embeddings.
                        modality_labels = torch.cat([
                            torch.zeros(batch_size, dtype=torch.long),
                            torch.ones(batch_size, dtype=torch.long)
                        ]).to(self.device)

                        concat_embeds = torch.cat(
                            [speech_embeds, text_embeds], dim=0
                        )

                        disc_logits = self.modality_discriminator(concat_embeds)
                        disc_loss = F.cross_entropy(disc_logits, modality_labels)
                        losses["disc_loss"] = disc_loss.item()

                        # Discriminator accuracy.
                        modality_preds = torch.argmax(disc_logits, dim=1)
                        all_modality_preds.extend(modality_preds.cpu().tolist())
                        all_modality_labels.extend(modality_labels.cpu().tolist())

        if self.use_style_class_loss:
            style_acc = (
                torch.sum(
                    torch.tensor(all_style_preds) == torch.tensor(all_style_labels)
                ) / len(all_style_preds)
            ).float().mean().item()
            losses["style_class_acc"] = style_acc

        if (
            self.use_modality_disc_loss and
            epoch >= self.style_class_pretrain_epochs + self.modality_disc_delay_epochs
        ):
            disc_acc = (
                torch.sum(
                    torch.tensor(all_modality_preds) == torch.tensor(all_modality_labels)
                ) / len(all_modality_preds)
            ).float().mean().item()
            losses["disc_acc"] = disc_acc

        self.writer.log_validation(losses, self.step)

        # Compute speech and text embeddings from current model and log t-SNE
        # visualization to Tensorboard.
        tsne_emb_plot = self.get_embedding_tsne_plot()
        self.writer.log_tsne_fig(tsne_emb_plot, self.step)

        # TODO: Log audio and text style prompts in Tensorboard.
        # self.writer.log_audio_text_responses()

        # Save checkpoints.
        if epoch > 60 or epoch % 10 == 0:
            save_path = os.path.join(
                self.checkpoint_save_dir, f"epoch_{epoch}.pt"
            )
            save_dict = {
                "style_prompt_encoder": self.style_prompt_encoder.state_dict(),
                "speech_style_encoder": self.speech_style_encoder.state_dict(),
                "optim_e": self.optim_e.state_dict(),
                # "lr_scheduler": self.lr_scheduler.state_dict(),
                "epoch": epoch,
                "step": self.step,
            }
            if self.use_clip_loss:
                save_dict["clip_loss"] = self.clip_loss.state_dict()
            if self.use_style_class_loss:
                save_dict["aux_style_classifier"] = self.aux_style_classifier.state_dict()
                save_dict["optim_s"] = self.optim_s.state_dict()
            if self.use_modality_disc_loss:
                save_dict["modality_disc"] = self.modality_discriminator.state_dict()
                save_dict["optim_d"] = self.optim_d.state_dict()

            torch.save(save_dict, save_path)
            print(f"Saved checkpoint for epoch {epoch} to {save_path}.\n")

    def compute_embeds_for_tsne(self):
        speech_embs = {}
        text_embs = {}
        print("Computing embeddings for t-SNE visualization")
        for sample_idx, (
            audio,
            _,
            _,
            text_input_ids,
            text_attention_mask,
            _,
            style_label_strs,
            _,
        ) in enumerate(tqdm(self.tsne_dataloader)):
            with torch.no_grad():
                with torch.autocast(
                    device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp
                ):
                    audio = audio.to(self.device)
                    text_input_ids = text_input_ids.to(self.device)
                    text_attention_mask = text_attention_mask.to(self.device)

                    speech_emb = self.speech_style_encoder(
                        audio
                    ).detach().squeeze().cpu().float().numpy()
                    text_emb = self.style_prompt_encoder(
                        input_ids=text_input_ids,
                        attention_mask=text_attention_mask,
                    ).detach().squeeze().cpu().float().numpy()

                    style_label = style_label_strs[0]
                    if style_label in speech_embs:
                        speech_embs[style_label].append(speech_emb)
                        text_embs[style_label].append(text_emb)
                    else:
                        speech_embs[style_label] = [speech_emb]
                        text_embs[style_label] = [text_emb]

        return speech_embs, text_embs

    def get_embedding_tsne_plot(self):
        speech_embs, text_embs = self.compute_embeds_for_tsne()
        fig = plot_embeddings_tsne(speech_embs, text_embs, return_fig=True)
        return fig
