import argparse

import torch
from torch.utils.data import DataLoader

from src.data.audio_preprocessor import AudioPreprocessor
from src.data.metrics import Metrics
from src.data.pair_builder import PairBuilder
from src.model.aamsoftmax import AAMSoftmax
from src.model.embedding_extractor import EmbeddingExtractor
from src.train.speaker_dataset import SpeakerDataset
from src.train.trainer import Trainer
from src.utils.utils import split_dataset

# ============================================================
# Arguments
# ============================================================
parser = argparse.ArgumentParser(description="Speaker verification training")

parser.add_argument(
    "--dataset_path",
    type=str,
    default="speaker_dataset",
)
parser.add_argument(
    "--model_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--save_dir",
    type=str,
    default="runs/speaker_train/exp",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=32,
)
parser.add_argument(
    "--epochs",
    type=int,
    default=50,
)
parser.add_argument(
    "--device",
    type=str,
    default="cpu",
)
parser.add_argument(
    "--target_sr",
    type=int,
    default=8000,
)
parser.add_argument(
    "--train_ratio",
    type=float,
    default=0.7,
)
parser.add_argument(
    "--val_ratio",
    type=float,
    default=0.2,
)
parser.add_argument(
    "--test_ratio",
    type=float,
    default=0.1,
)
parser.add_argument(
    "--trainable_blocks",
    type=int,
    default=2,
)
parser.add_argument(
    "--no-balance",
    action="store_true",
)

args = parser.parse_args()

# ============================================================
# Model
# ============================================================
extractor = EmbeddingExtractor(
    trainable_blocks=args.trainable_blocks,
    device=args.device,
)
criterion = AAMSoftmax(
    embedding_dim=192,
    num_classes=1,  # будет заменено позже
)
optimizer = torch.optim.AdamW(
    list(extractor.parameters()) + list(criterion.parameters()),
    lr=1e-4,
    weight_decay=1e-4,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=args.epochs,
    eta_min=1e-6,
)

# ============================================================
# Utils
# ============================================================
builder = PairBuilder(
    balance=not args.no_balance,
)
metrics = Metrics()
train_preprocessor = AudioPreprocessor(
    augment=True,
    target_sr=args.target_sr,
)
val_preprocessor = AudioPreprocessor(
    target_sr=args.target_sr,
)

# ============================================================
# Dataset split
# ============================================================
split_dataset(
    SpeakerDataset(
        args.dataset_path,
        return_audio=False,
    ),
    train_ratio=args.train_ratio,
    val_ratio=args.val_ratio,
    test_ratio=args.test_ratio,
    split_key="speaker_id",
    persistent=True,
    output_dir=args.dataset_path,
)

# ============================================================
# Train dataset
# ============================================================
train_dataset = SpeakerDataset(
    f"{args.dataset_path}/train",
    preprocessor=train_preprocessor,
    return_audio=True,
)
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=8,
    persistent_workers=True,
    pin_memory=args.device.startswith("cuda"),
)

# ============================================================
# Validation dataset
# ============================================================
val_dataset = SpeakerDataset(
    f"{args.dataset_path}/val",
    preprocessor=val_preprocessor,
    return_audio=False,
)

# ============================================================
# Criterion
# ============================================================
criterion = AAMSoftmax(
    embedding_dim=192,
    num_classes=train_dataset.get_num_speakers(),
)
optimizer = torch.optim.AdamW(
    list(extractor.parameters()) + list(criterion.parameters()),
    lr=1e-4,
    weight_decay=1e-4,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=args.epochs,
    eta_min=1e-6,
)

# ============================================================
# Trainer
# ============================================================
trainer = Trainer(
    builder=builder,
    train_preprocessor=train_preprocessor,
    val_preprocessor=val_preprocessor,
    metrics=metrics,
    encoder=extractor,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    save_dir=args.save_dir,
)

# ============================================================
# Resume training
# ============================================================
start_epoch = 0
if args.model_path is not None:
    start_epoch = trainer.load_checkpoint(args.model_path)

# ============================================================
# Training loop
# ============================================================
for epoch in range(start_epoch, args.epochs):
    train_loss = trainer.train(train_loader)
    val_metrics = trainer.validate(val_dataset)
    
    trainer.save_checkpoint(
        epoch + 1,
        train_loss,
        val_metrics,
    )

    trainer.step_scheduler()

    print(
        f"Epoch {epoch + 1}/{args.epochs} "
        f"train={train_loss:.4f} "
        f"eer={val_metrics['metrics']['eer']:.4f} "
        f"auc={val_metrics['metrics']['roc_auc']:.4f}"
    )
