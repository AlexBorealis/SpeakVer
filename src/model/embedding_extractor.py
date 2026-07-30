import torch
import torch.nn as nn
import torch.nn.functional as F
from speechbrain.inference.speaker import EncoderClassifier


class EmbeddingExtractor(nn.Module):
    def __init__(
        self,
        source: str = "speechbrain/spkrec-ecapa-voxceleb",
        savedir: str = "pretrained_models/ecapa",
        normalize: bool = True,
        device: str = "cpu",
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

    def _get_embedding_modules(self) -> dict[str, nn.Module]:
        """
        Returns all trainable modules of the embedding model.

        Example:
            tdnn
            seres2netblock1
            seres2netblock2
            seres2netblock3
            mfa
            asp
            asp_bn
            fc
        """

        embedding = self.mods.embedding_model

        modules = {}

        for name, module in embedding.named_children():
            if name == "blocks":
                modules["tdnn"] = module[0]

                for i, block in enumerate(module[1:], start=1):
                    modules[f"seres2netblock{i}"] = block
            else:
                modules[name] = module

        return modules

    def set_trainable_modules(
        self,
        trainable_modules: list[str],
        classifier: nn.Module = None,
        disable: bool = True,
    ):
        embedding = self.mods.embedding_model
        modules = self._get_embedding_modules()

        # freeze embedding
        for p in embedding.parameters():
            p.requires_grad = False

        # freeze classifier
        if classifier is not None:
            for p in classifier.parameters():
                p.requires_grad = False

        if "all" in trainable_modules:
            for p in embedding.parameters():
                p.requires_grad = True

        else:
            for name in trainable_modules:
                # classifier
                if name == "classifier":
                    if classifier is None:
                        raise ValueError("classifier is None")

                    for p in classifier.parameters():
                        p.requires_grad = True

                    continue

                # embedding modules
                if name not in modules:
                    raise ValueError(
                        f"Unknown module '{name}'.\n"
                        f"Available modules: "
                        f"{', '.join(list(modules.keys()) + ['classifier'])}"
                    )

                for p in modules[name].parameters():
                    p.requires_grad = True

        if not disable:
            # statistics
            trainable_modules_info = []

            for name, module in modules.items():
                if any(p.requires_grad for p in module.parameters()):
                    trainable_modules_info.append(
                        (
                            name,
                            module.__class__.__name__,
                        )
                    )

            trainable_params = sum(
                p.numel() for p in embedding.parameters() if p.requires_grad
            )

            total_params = sum(p.numel() for p in embedding.parameters())

            print("=" * 60)
            print("Trainable modules:")

            enabled = 0

            for name, cls_name in trainable_modules_info:
                enabled += 1

                print(f"  {name:<18}: {cls_name}")

            if classifier is not None:
                if any(p.requires_grad for p in classifier.parameters()):
                    print(f"  {'classifier':<18}: {classifier.__class__.__name__}")

            if enabled == 0 and (
                classifier is None
                or not any(p.requires_grad for p in classifier.parameters())
            ):
                print("  (none)")

            classifier_trainable = 0
            classifier_total = 0

            if classifier is not None:
                classifier_trainable = sum(
                    p.numel() for p in classifier.parameters() if p.requires_grad
                )

                classifier_total = sum(p.numel() for p in classifier.parameters())

            print()
            print(f"Embedding params   : {trainable_params:,}/{total_params:,}")

            if classifier is not None:
                print(
                    f"Classifier params   : {classifier_trainable:,}/{classifier_total:,}"
                )
            print("=" * 60)

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

    def forward(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        waveforms = waveforms.to(
            self.device,
            non_blocking=True,
        )

        if lengths is None:
            lengths = torch.ones(
                waveforms.size(0),
                device=waveforms.device,
                dtype=waveforms.dtype,
            )
        else:
            lengths = lengths.to(
                self.device,
                non_blocking=True,
            )

        feats = self.mods.compute_features(waveforms)

        feats = self.mods.mean_var_norm(
            feats,
            lengths,
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

    @torch.inference_mode()
    def extract(self, waveform: torch.Tensor) -> torch.Tensor:
        self.eval()

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        return self.forward(waveform).squeeze(0)
