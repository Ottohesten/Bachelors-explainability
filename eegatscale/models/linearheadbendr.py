from torch import nn
import torch

from eegatscale.layers import EncodingAugment, Flatten
from eegatscale.models.bendr import BendrEncoder

from pytorch_lightning import LightningModule

class LinearHeadBENDR(LightningModule):
    def __init__(self, encoder, encoder_h, in_features, out_features, pool_length=4):
        super().__init__()

        # inferred values from the original implementation
        # https://github.com/SPOClab-ca/BENDR/blob/ac918abaec111d15fcaa2a8fcd2bd3d8b0d81a10/dn3_ext.py#L44
        classifier_layers = 1
        feat_do = 0.4
        # mask_p_t = 0.01
        # mask_p_c = 0.005
        # mask_t_span = 0.05
        # mask_c_span = 0.1

        #encoded_samples = 4 * 256
        encoded_samples = 14
        mask_t_span = 0
        mask_c_span = 51
        mask_p_t = 0.01
        mask_p_c = 0.005

        mask_t_span = mask_t_span if mask_t_span > 1 else int(mask_t_span * encoded_samples)
        mask_t_span = 0 if encoded_samples < 2 else mask_t_span

        mask_c_span = mask_c_span if mask_c_span > 1 else int(mask_c_span * encoder_h)

        self.enc_augment = EncodingAugment(
            encoder_h, mask_p_t, mask_p_c, mask_c_span=mask_c_span, mask_t_span=mask_t_span
        )

        self.summarizer = nn.AdaptiveAvgPool1d(4)

        classifier_layers = (
            [encoder_h * pool_length for i in range(classifier_layers)]
            if not isinstance(classifier_layers, (tuple, list))
            else classifier_layers
        )
        classifier_layers.insert(0, 3 * encoder_h * pool_length)

        self.extended_classifier = nn.Sequential(Flatten())

        for i in range(1, len(classifier_layers)):
            self.extended_classifier.add_module(
                "ext-classifier-{}".format(i),
                nn.Sequential(
                    nn.Linear(classifier_layers[i - 1], classifier_layers[i]),
                    nn.Dropout(feat_do),
                    nn.ReLU(),
                    nn.BatchNorm1d(classifier_layers[i]),
                ),
            )

        final_classifier = nn.Linear(classifier_layers[-1], out_features)
        nn.init.xavier_normal_(final_classifier.weight)
        final_classifier.bias.data.zero_()
        self.clf = nn.Sequential(Flatten(), final_classifier)
        
        
        self.loss_fn = nn.CrossEntropyLoss()
        
        # In encoder is of type BendrEncoder
        if isinstance(encoder, BendrEncoder):
            self.encoder = encoder
            #self.enc_augment.init_from_contextualizer(encoder)
        elif encoder is None:
            self.encoder = BendrEncoder(in_features=in_features, encoder_h=encoder_h, grad_frac=0.1)            
        elif isinstance(encoder, str):     
            checkpoint_path = encoder     
            state_dict = torch.load(checkpoint_path, map_location="cpu")['state_dict']
            encoder_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("encoder."):
                    encoder_state_dict[k[8:]] = v
            
            contextualizer_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("contextualizer."):
                    contextualizer_state_dict[k[15:]] = v
            
            encoder = BendrEncoder(in_features=in_features, encoder_h=encoder_h, grad_frac=0.1)
            encoder.load_state_dict(encoder_state_dict)
            self.encoder = encoder
            self.enc_augment.init_from_contextualizer(checkpoint_path)
        else:
            raise ValueError("encoder must be of type BendrEncoder or str")
        

    def forward(self, x):
        x = self.encoder(x)
        x = self.enc_augment(x)
        x = self.summarizer(x)
        x = self.extended_classifier(x)
        x = self.clf(x)
        return x

    def _step(self, batch):
        x, y = batch
        z = self(x)
        loss = self.loss_fn(z, y.long())    
        acc = (z.argmax(dim=1) == y).float().mean()
        return loss, {'acc': acc}

    def training_step(self, batch, batch_idx):
        self.train()
        loss, metrics = self._step(batch)
        self.log("loss", loss, prog_bar=True)
        if len(metrics):
            self.log_dict(metrics)
        return loss
    
    def test_step(self, batch, batch_idx) -> None:
        self.eval()
        loss, metrics = self._step(batch)
        self.log("test_loss", loss)
        if len(metrics):
            self.log_dict({f"test_{k}": v for k,v in metrics.items()})

    def validation_step(self, batch, batch_idx) -> None:
        self.eval()
        loss, metrics = self._step(batch)
        self.log("val_loss", loss)
        if len(metrics):
            self.log_dict({f"val_{k}": v for k,v in metrics.items()})
            
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001, weight_decay=0.001)
              
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=0.0001, total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1, last_epoch=-1
        )
        
        return [optimizer], [scheduler]

if __name__ == "__main__":
    # Import trainer
    from pytorch_lightning import Trainer
    from scripts.finetune_CV import H5PYDatasetLabeled
    from torch.utils.data import DataLoader
    
    encoder = '/scratch/s194260/BENDR/bendr_data/checkpoints/tuh-ica-standardize-epoch=8-step=64030-val_loss=0.29.ckpt'
    model = LinearHeadBENDR(encoder, encoder_h = 512, in_features = 19, out_features = 2)
    
    path = '/scratch/s194260/BENDR/bendr_data/preprocess_downstream_mmidb_ica_combined'
    
    # Create the dataset
    dataset = H5PYDatasetLabeled(path)

    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    
    
    trainer = Trainer(max_epochs=10)
    trainer.fit(model, loader)
    
    
    #model.configure_optimizers()