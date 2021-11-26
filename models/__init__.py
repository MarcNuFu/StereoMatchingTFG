import models.autoencoder_backbone as autoencoder_backbone
import models.autoencoder_baseline as autoencoder_baseline

__models__ = {
    "AutoencoderBackbone": autoencoder_backbone.AutoencoderBackbone(),
    "AutoencoderBaseline": autoencoder_baseline.AutoencoderBaseline()
}