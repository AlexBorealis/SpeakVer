import torch
import torch.nn as nn
import torch.nn.functional as F
from speechbrain.inference.speaker import EncoderClassifier


class EmbeddingExtractor(nn.Module):
    def __init__(
        self,
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/ecapa",
        normalize=True,
        trainable_layers=2,
        device="cpu"
    ):
        super().__init__()
        self.normalize = normalize
        self.device = torch.device(device)

        self.classifier = EncoderClassifier.from_hparams(
            source=source,
            savedir=savedir,
            run_opts={"device": str(self.device)}
        )

        self.mods = self.classifier.mods
        self.embedding_dim = 192

        self.freeze_layers(trainable_layers)

        self.to(self.device)


    def freeze_layers(
        self,
        trainable_layers
    ):
        layers = list(
            self.mods.embedding_model.children()
        )

        for p in self.mods.embedding_model.parameters():
            p.requires_grad = False

        if trainable_layers > 0:
            for layer in layers[-trainable_layers:]:
                for p in layer.parameters():
                    p.requires_grad = True


    def forward(
        self,
        waveforms
    ):
        device = next(self.parameters()).device

        waveforms = waveforms.to(device)

        feats = self.mods.compute_features(
            waveforms
        )

        feats = self.mods.mean_var_norm(
            feats,
            torch.ones(
                feats.shape[0],
                device=feats.device
            )
        )

        embeddings = self.mods.embedding_model(
            feats
        )

        if embeddings.ndim == 3:
            embeddings = embeddings.squeeze(1)

        if self.normalize:
            embeddings = F.normalize(
                embeddings,
                dim=1
            )

        return embeddings


    @torch.no_grad()
    def extract(
        self,
        waveform
    ):
        self.eval()

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        return self.forward(
            waveform
        ).squeeze(0).cpu()