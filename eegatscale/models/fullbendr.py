from torch import nn
import torch

from eegatscale.layers import Flatten
from eegatscale.models.bendr import BendrEncoder, BendrContextualizer

from pytorch_lightning import LightningModule

class FullBENDR(LightningModule):
    
    def __init__(self, encoder, encoder_h, in_features, out_features):
        super().__init__()
        
        dropout = 0.0
        new_projection_layers = 0
        
        if isinstance(encoder, str):      
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
            encoder.load_state_dict(encoder_state_dict, strict=True)
            self.encoder = encoder
            
            contextualizer = BendrContextualizer(in_features=encoder_h)
            contextualizer.load_state_dict(contextualizer_state_dict, strict=True)
            self.contextualizer = contextualizer
            
            self.contextualizer.mask_replacement.requires_grad = False
            
        else:
            raise ValueError("encoder must be of type BendrEncoder or str")
        
        #mask_t_span = mask_t_span if mask_t_span > 1 else int(mask_t_span * encoded_samples)
        #mask_c_span = mask_c_span if mask_c_span > 1 else int(mask_c_span * encoder_h)
           
        self.projection_mlp = nn.Sequential()
        for p in range(1, new_projection_layers + 1):
            self.projection_mlp.add_module("projection-{}".format(p), nn.Sequential(
                nn.Linear(encoder_h, encoder_h),
                nn.Dropout(dropout),
                nn.BatchNorm1d(encoder_h),
                nn.GELU(),
            ))
            
        self.loss_fn = nn.CrossEntropyLoss()
            
        final_classifier = nn.Linear(encoder_h, out_features)
        nn.init.xavier_normal_(final_classifier.weight)
        final_classifier.bias.data.zero_()
        self.clf = nn.Sequential(Flatten(), final_classifier)
        
    def forward(self, *x):
        encoded = self.encoder(x[0])
        context = self.contextualizer(encoded)
        return self.clf(context[:, :, -1])
    
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
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from eegatscale.models.bendr import BendrEncoder, BendrContextualizer
    from tqdm.auto import tqdm
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    encoder = BendrEncoder(in_features=20, encoder_h=512)
    contextualizer = BendrContextualizer(in_features=512)
    
    encoder_weights = torch.load("/scratch/s194260/BENDR/bendr_data/encoder.pt")
    contextualizer_weights = torch.load("/scratch/s194260/BENDR/bendr_data/contextualizer.pt")
    
    encoder.load_state_dict(encoder_weights)
    contextualizer.load_state_dict(contextualizer_weights)
    
    model = FullBENDR(encoder, contextualizer, in_features=512, out_features=2).to(device)
    
    # Random torch tensor of dimensions (batch_size, channels, samples) that is (4, 20, 1280)
    x = torch.rand(4, 20, 1280).to(device)
    print(model(x))
    print(model(x).shape)
    
    # import h5py
    # from sklearn.model_selection import train_test_split
    # #data_path = "/scratch/s194260/BENDR/bendr_data/mmidb_windows_1024.hdf5"
    # data_path = "/scratch/s194260/BENDR/bendr_data/mmidb_windows_1280.hdf5"
    
    # batch_size = 4
    
    # with h5py.File(data_path, "r") as f:
    #     X = f["data"][:]
    #     y = f["labels"][:].astype(int)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # # train model

    # # values from the BENDR config 
    # # https://github.com/SPOClab-ca/BENDR/blob/main/configs/downstream_datasets.yml
    # learning_rate = 0.00001
    # n_epochs = 100

    # # optimizer configuration from the DN3 implementation 
    # # https://github.com/SPOClab-ca/dn3/blob/4d477fe42d[3d8ce64f3b790585bfa5c7acb84848/dn3/trainable/processes.py#L89
    # #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.01, momentum=0.9, nesterov=True)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
    # criterion = nn.CrossEntropyLoss()
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer, max_lr=learning_rate, epochs=n_epochs, steps_per_epoch=len(train_loader), pct_start=0.1, last_epoch=-1
    # )

    # model = model.to(device)

    # for epoch in range(1, n_epochs + 1):
    #     correct = 0
    #     total = 0
    #     for batch in tqdm(train_loader, desc=f"Epoch {epoch}, train"):
    #         # Skip if the batch size is less than the batch size
    #         if len(batch[0]) < batch_size:
    #             continue            
            
    #         optimizer.zero_grad()
            
    #         data = batch[0]
    #         label = batch[1]
   
    #         data = data.to(device)
    #         label = label.to(device)
            
    #         logits = model(data)
            
    #         _, predicted = torch.max(logits.data, 1)
    #         total += label.size(0)
    #         correct += (predicted == label).sum().item()
            
    #         loss = criterion(logits, label)
    #         loss.backward()

    #         optimizer.step()
    #         scheduler.step()

    #     print(f"Epoch {epoch}, train accuracy: {100 * correct / total:2f}%")

    #     with torch.no_grad():
    #         model.eval()
            
    #         # Test accuracy
    #         correct = 0
    #         total = 0
    #         for batch in tqdm(test_loader, desc=f"Epoch {epoch}, test"):
    #             if len(batch[0]) < batch_size:
    #                 continue         
    #             #data, label = batch["data"], batch["label"]
    #             data = batch[0]
    #             label = batch[1]
                
    #             data = data.to(device)
    #             label = label.to(device)

    #             logits = model(data)
    #             _, predicted = torch.max(logits.data, 1)
    #             total += label.size(0)
    #             correct += (predicted == label).sum().item()

    #         print(f"Epoch {epoch}, test accuracy: {100 * correct / total:2f}%")
            
            
    #         model.train()
            
    # # Save model
    # torch.save(model.state_dict(), "/scratch/s194260/BENDR/bendr_data/full_bendr.pt")