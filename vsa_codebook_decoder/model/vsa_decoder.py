import pathlib
from typing import Any, Optional, List

import hydra
import pytorch_lightning as pl
import torch
import wandb
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch.nn.functional as F
from torch import nn
from torch.optim import lr_scheduler

from ..metrics.vsa import vsa_decoding_accuracy
from ..utils import iou_pytorch
from ..dataset.paired_dsprites import Dsprites
from ..codebook.codebook import Codebook
from ..config import VSADecoderConfig
from .binder import Binder, FourierBinder
from .decoder import Decoder
from .encoder import Encoder


class VSADecoder(pl.LightningModule):
    binder: Binder
    cfg: VSADecoderConfig

    def __init__(self, cfg: VSADecoderConfig):
        super().__init__()
        self.cfg = cfg

        features = Codebook.make_features_from_dataset(Dsprites)  # type: ignore

        self.encoder = Encoder(image_size=cfg.model.image_size,
                               latent_dim=cfg.model.latent_dim,
                               hidden_channels=cfg.model.encoder_config.hidden_channels)

        self.decoder = Decoder(image_size=cfg.model.image_size,
                               latent_dim=cfg.model.latent_dim,
                               in_channels=cfg.model.decoder_config.in_channels,
                               hidden_channels=cfg.model.decoder_config.hidden_channels)
        self.codebook = Codebook(features=features,
                                 latent_dim=cfg.model.latent_dim,
                                 seed=cfg.experiment.seed)

        self.layer_norms = nn.ModuleList([nn.LayerNorm(cfg.model.latent_dim)])
        self.q_proj = nn.ModuleList([nn.Linear(cfg.model.latent_dim, cfg.model.latent_dim) for _ in
                                     range(cfg.dataset.n_features)])

        if self.cfg.experiment.scale == 'sqrt':
            self.scale = (cfg.model.latent_dim) ** 0.5
        elif self.cfg.experiment.scale == 'none':
            self.scale = 1
        else:
            raise NotImplemented(f"Wrong scale parameter {cfg.experiment.scale}")
        self.softmax = nn.Softmax(dim=1)

        if cfg.model.binder == 'fourier':
            self.binder = FourierBinder(placeholders=self.codebook.placeholders)
        else:
            raise NotImplemented(f"Wrong binder type {cfg.model.binder}")

        self.save_hyperparameters()

    def attention(self, x):
        query: List[torch.tensor] = [self.q_proj[i](x) for i in range(self.cfg.dataset.n_features)]

        features: List[torch.tensor] = [feature.to(self.device) for feature in
                                        self.codebook.vsa_features]

        k: torch.tensor
        attn_logits = [torch.matmul(query[i], key.transpose(-2, -1)) for i, key in
                       enumerate(features)]
        attn_logits = [logit * self.scale for logit in attn_logits]
        attention = [self.softmax(logit) for logit in attn_logits]

        max_values = [torch.mean(torch.max(attn, dim=1).values) for attn in attention]

        values = [torch.matmul(attention[i], features[i]) for i in
                  range(self.cfg.dataset.n_features)]

        values = torch.stack(values, dim=1)
        values = self.binder(values)

        return values, max_values

    def step(self, batch, batch_idx, mode: str = 'Train') -> torch.tensor:
        # Logging period
        # Log Train samples once per epoch
        # Log Validation images triple per epoch
        if mode == 'Train':
            log_images = lambda x: x == 0
        elif mode == 'Validation':
            log_images = lambda x: x % 10 == 0
        else:
            raise ValueError

        image: torch.tensor
        label: torch.tensor

        image, labels = batch

        z = self.encoder(image)
        z, max_values = self.attention(z)
        z = torch.sum(z, dim=1)
        decoded_image = self.decoder(z)

        loss = F.mse_loss(decoded_image, image)
        iou = iou_pytorch(decoded_image, image)

        self.log(f"{mode}/MSE Loss", loss)
        self.log(f"{mode}/IOU", iou)
        self.logger.experiment.log(
            {"Max/" + Dsprites.feature_names[i]: max_values[i] for i in
             range(self.cfg.dataset.n_features)})

        if log_images(batch_idx):
            self.logger.experiment.log({
                f"{mode}/Images": [
                    wandb.Image(image[0], caption='Image'),
                    wandb.Image(decoded_image[0],
                                caption='Reconstruction'),
                ]})
        return loss

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss = self.step(batch, batch_idx, mode='Train')
        return loss

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        self.step(batch, batch_idx, mode='Validation')
        return None

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.experiment.lr)
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=self.cfg.experiment.lr,
                                            epochs=self.cfg.experiment.max_epochs,
                                            steps_per_epoch=self.cfg.experiment.steps_per_epoch,
                                            pct_start=self.cfg.experiment.pct_start)
        return {"optimizer": optimizer,
                "lr_scheduler": {'scheduler': scheduler,
                                 'interval': 'step',
                                 'frequency': 1}, }

    def on_test_start(self) -> None:
        self.latent_vectors = []
        self.latent_features = []
        self.labels = []

    def test_step(self, batch, batch_idx):
        image, labels = batch

        z = self.encoder(image)
        latent_features, max_values = self.attention(z)
        latent_vectors = torch.sum(z, dim=1)

        return latent_vectors, latent_features, labels

    def on_test_batch_end(self,
                          outputs: Optional[STEP_OUTPUT],
                          batch: Any,
                          batch_idx: int,
                          dataloader_idx: int) -> None:
        latent_vectors, latent_features, labels = outputs

        self.latent_vectors.append(latent_vectors)
        self.latent_features.append(latent_features)
        self.labels.append(labels)

    def on_test_end(self) -> None:
        self.latent_vectors = torch.cat(self.latent_vectors)
        self.latent_features = torch.cat(self.latent_features)
        self.labels = torch.cat(self.labels)

        vsa_accuracy = vsa_decoding_accuracy(placeholders=self.binder.hd_placeholders.squeeze(0),
                                             codebook=self.codebook.vsa_features,
                                             latent_vectors=self.latent_vectors,
                                             labels=self.labels,
                                             device=self.device)
        self.logger.experiment.log(vsa_accuracy)


cs = ConfigStore.instance()
cs.store(name="config", node=VSADecoderConfig)

path_to_dataset = pathlib.Path().absolute()


@hydra.main(version_base=None, config_name="config")
def main(cfg: VSADecoderConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.experiment.seed)
    model = VSADecoder(cfg)

    batch_size = 10
    latent_dim = 1024
    img = torch.randn((batch_size, 1, 64, 64))

    x = torch.randn((batch_size, 1024))
    result = model.attention(x)
    # labels = torch.randint(0, 3, (batch_size, 5))

    # model.step((img, labels), 1)

    pass


if __name__ == '__main__':
    main()
