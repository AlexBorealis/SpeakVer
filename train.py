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

BATCH_SIZE = 32
EPOCHS = 10

extractor = EmbeddingExtractor()
builder = PairBuilder()
metrics = Metrics()

train_preprocessor = AudioPreprocessor(device="cpu", augment=True, target_sr=8000)
val_preprocessor = AudioPreprocessor(device="cpu")

# только для построения split
train_dataset, val_dataset, test_dataset = split_dataset(
    SpeakerDataset("debug_audio"),
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1,
    split_key="speaker_id",
    persistent=True,
    output_dir="debug_audio",
)

# train subset
train_dataset = SpeakerDataset(
    "debug_audio/train", preprocessor=train_preprocessor, return_audio=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    persistent_workers=True,
)

# validation subset
val_dataset = SpeakerDataset(
    "debug_audio/val", preprocessor=val_preprocessor, return_audio=False
)


criterion = AAMSoftmax(embedding_dim=192, num_classes=train_dataset.get_num_speakers())
optimizer = torch.optim.AdamW(
    list(extractor.parameters()) + list(criterion.parameters()),
    lr=1e-4,
    weight_decay=1e-4,
)


trainer = Trainer(
    train_preprocessor=train_preprocessor,
    val_preprocessor=val_preprocessor,
    metrics=metrics,
    builder=builder,
    encoder=extractor,
    criterion=criterion,
    optimizer=optimizer,
    save_dir="runs/speaker_train/exp3",
)


for epoch in range(EPOCHS):
    train_loss = trainer.train(train_loader)
    val_metrics = trainer.validate(val_dataset)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"train={train_loss:.4f} "
        f"eer={val_metrics['metrics']['eer']:.4f} "
        f"auc={val_metrics['metrics']['roc_auc']:.4f}"
    )

    trainer.save_checkpoint(epoch + 1, train_loss, val_metrics)
