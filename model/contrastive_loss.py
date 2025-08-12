import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, learnable_temperature=True, init_temp=0.07):
        super().__init__()
        if learnable_temperature:
            # Store log temperature so the learned temp = exp(log_temp) > 0
            self.log_temp = nn.Parameter(
                torch.log(torch.tensor(1.0 / init_temp))
            )
        else:
            self.register_buffer(
                'log_temp', torch.log(torch.tensor(1.0 / init_temp))
            )

    def forward(self, audio_embeds, text_embeds):
        # Normalize embeddings
        audio_embeds = F.normalize(audio_embeds, p=2, dim=-1)
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)

        # Compute cosine similarity
        sim_matrix = audio_embeds @ text_embeds.T  # [B, B]

        # Apply temperature scaling
        temperature = torch.exp(self.log_temp)
        logits = sim_matrix * temperature

        # Compute symmetric cross-entropy loss
        targets = torch.arange(logits.size(0), device=logits.device)
        loss_audio = F.cross_entropy(logits, targets)
        loss_text = F.cross_entropy(logits.T, targets)

        return (loss_audio + loss_text) / 2
