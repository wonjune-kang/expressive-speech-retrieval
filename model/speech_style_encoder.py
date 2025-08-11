import torch
import torch.nn as nn
from funasr import AutoModel as FAAutoModel
from transformers import AutoModel as HFAutoModel


def load_wavlm(encoder_type):
    model = HFAutoModel.from_pretrained(encoder_type)
    return model


def load_emotion2vec(encoder_type):
    model = FAAutoModel(model=encoder_type, hub="hf").model
    return model


class SpeechStyleEncoder(nn.Module):
    def __init__(self, config, device):
        super(SpeechStyleEncoder, self).__init__()
        self.config = config
        self.device = device

        self.encoder_type = self.config.model.speech_style_encoder.type
        if "wavlm" in self.encoder_type:
            self.encoder = load_wavlm(self.encoder_type)
        elif "emotion2vec" in self.encoder_type:
            self.encoder = load_emotion2vec(self.encoder_type)
        else:
            raise Exception("Invalid speech encoder type.")

        if self.config.model.encoder_projection_layers == 1:
            self.embed_projection = nn.Linear(
                self.config.model.speech_encoder_hidden_size,
                self.config.model.embedding_channels,
            )
        else:
            self.embed_projection = nn.Sequential(
                nn.Linear(
                    self.config.model.speech_encoder_hidden_size,
                    self.config.model.speech_encoder_hidden_size,
                ),
                nn.ReLU(),
                nn.Linear(
                    self.config.model.speech_encoder_hidden_size,
                    self.config.model.embedding_channels,
                ),
            )

        self.freeze_encoder = self.config.model.speech_style_encoder.freeze
        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, audio, speech_embed_mask=None):
        # audio has shape (B, audio_len_samples)
        # speech_embed_mask has shape (B, max_speech_embeds_len)
        # encoder_embeds should have shape (B, max_speech_embeds_len, embed_dim)
        if "wavlm" in self.encoder_type:
            encoder_embeds = self.encoder(audio).last_hidden_state
        else:
            encoder_embeds = self.encoder(
                audio, padding_mask=None, mask=False, features_only=True
            )["x"]

        if audio.shape[0] > 1:
            assert speech_embed_mask is not None

            # HACK: speech_embed_mask might have length 1 shorter than
            # encoder_embeds. In this case, manually crop encoder_embeds.
            if speech_embed_mask.shape[1] == encoder_embeds.shape[1] - 1:
                encoder_embeds = encoder_embeds[:, :-1]

            # Apply speech_embed_mask, sum only valid (non-padding) elements
            # along the sequence length dimension, and count the valid ones.
            valid_embeds = encoder_embeds * speech_embed_mask.unsqueeze(-1)
            sum_valid = valid_embeds.sum(dim=1)
            count_valid = speech_embed_mask.sum(dim=1).unsqueeze(-1)

            # To avoid division by zero, replace zero counts with 1 (for edge
            # cases where there are no valid elements).
            count_valid = count_valid.clamp(min=1)

            # Temporally mean pool the base encoder's output representations.
            # pooled_encoder_embeds should have shape (B, embed_dim).
            pooled_encoder_embeds = sum_valid / count_valid

        else:
            pooled_encoder_embeds = torch.mean(encoder_embeds, dim=1)

        # Pass through linear projection layer.
        speech_style_emb = self.embed_projection(pooled_encoder_embeds)

        return speech_style_emb
