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
        trainable_blocks=2,
        device="cpu",
    ):
        super().__init__()

        self.normalize = normalize
        self.device = torch.device(device)

        self.classifier = EncoderClassifier.from_hparams(
            source=source,
            savedir=savedir,
            run_opts={"device": str(self.device)},
        )

        self.mods = self.classifier.mods

        self.freeze_blocks(trainable_blocks)

        self.to(self.device)

    def freeze_blocks(self, trainable_blocks: int):
        blocks = list(self.mods.embedding_model.children())

        for p in self.mods.embedding_model.parameters():
            p.requires_grad = False

        if trainable_blocks > 0:
            for block in blocks[-trainable_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True

    def load_encoder_weights(self, checkpoint_path: str):
        print(f"Loading encoder weights: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        if "encoder" not in checkpoint:
            raise KeyError(
                f"Checkpoint '{checkpoint_path}' does not contain 'encoder'."
            )

        self.load_state_dict(checkpoint["encoder"])

        return checkpoint

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        waveforms = waveforms.to(
            self.device,
            non_blocking=True,
        )

        feats = self.mods.compute_features(waveforms)

        feats = self.mods.mean_var_norm(
            feats,
            torch.ones(
                feats.shape[0],
                device=self.device,
            ),
        )

        embeddings = self.mods.embedding_model(feats)

        if embeddings.ndim == 3:
            embeddings = embeddings.squeeze(1)

        if self.normalize:
            embeddings = F.normalize(
                embeddings,
                dim=1,
            )

        return embeddings

    @torch.no_grad()
    def extract(self, waveform: torch.Tensor) -> torch.Tensor:
        self.eval()

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        return self.forward(waveform).squeeze(0)