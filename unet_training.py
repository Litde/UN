from torch.utils.data import DataLoader

from unet_autoencoder import UNetLightning
from dataset import ImageDatasetLightning
from pytorch_lightning import Trainer


EPOCHS = 10
BATCH_SIZE = 1
LEARNING_RATE = 1e-3

COMET_API_KEY = 'CadJDqKmVOKD754AGrKMLdZDB'  # Replace with your actual Comet API key


def train_model():
    data_module = ImageDatasetLightning(corrupted_dir='output/images', original_dir='wikiart', batch_size=BATCH_SIZE)
    data_module.setup()

    model = UNetLightning(lr=LEARNING_RATE, comet_workspace='litde', comet_project_name='UN - inpainting', comet_api_key=COMET_API_KEY)
    trainer = Trainer(max_epochs=EPOCHS, accelerator='auto')

    trainer.fit(model, data_module)
    trainer.save_checkpoint('unet_models/model.pt')

    # trainer.test(model)
    #
    # print(f"Now let't lest the model")
    # out = model(data_module.test_dataloader().dataset[0])
    # print(f"Output shape: {out.shape}")

def test_model():
    data_module = ImageDatasetLightning(corrupted_dir='output/images', original_dir='wikiart', batch_size=BATCH_SIZE)
    data_module.setup()

    model = UNetLightning.load_from_checkpoint('unet_models/model.pt', comet_api_key=COMET_API_KEY,
                                               comet_workspace='litde', comet_project_name='UN - inpainting')
    trainer = Trainer(accelerator='auto')

    trainer.test(model, datamodule=data_module)

if __name__ == "__main__":
    train_model()

