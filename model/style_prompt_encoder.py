import torch
import torch.nn as nn
from transformers import AutoModel, T5EncoderModel


class StylePromptEncoder(nn.Module):
    def __init__(self, config, device):
        super(StylePromptEncoder, self).__init__()
        self.config = config
        self.device = device

        encoder_type = self.config.model.style_prompt_encoder.type
        if "bert" in encoder_type or "roberta" in encoder_type:
            self.encoder = AutoModel.from_pretrained(encoder_type)
            self.encoder_class = "bert"
        elif "t5" in encoder_type:
            self.encoder = T5EncoderModel.from_pretrained(encoder_type)
            self.encoder_class = "t5"
        else:
            raise Exception("Invalid prompt encoder type.")

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

        self.freeze_encoder = self.config.model.style_prompt_encoder.freeze
        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        # input_ids and attention_mask have shape (B, seq_len).
        encoder_out = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state

        if self.encoder_class == "bert":
            # Get [CLS] embedding.
            encoder_out = encoder_out[:, 0, :]
        elif self.encoder_class == "t5":
            # Mean pool embedding sequence from hidden states.
            encoder_out = torch.mean(encoder_out, dim=1)

        style_prompt_embs = self.embed_projection(encoder_out)

        return style_prompt_embs
