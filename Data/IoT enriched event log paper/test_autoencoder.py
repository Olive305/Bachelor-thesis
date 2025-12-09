import torch
import torch.nn as nn
import torch.optim as optim
import duckdb
import pandas as pd
import os

import torch
import torch.nn as nn

class MixedTabularAutoencoder(nn.Module):
    def __init__(self, n_continuous, n_binary, n_event_binary, bottleneck_dim=3):
        super().__init__()
        
        self.n_cont = n_continuous
        self.n_bin = n_binary
        self.n_evt = n_event_binary
        self.input_dim = n_continuous + n_binary + n_event_binary
        
        # --- Encoder ---
        
        # 1) Continuous path
        self.cont_encoder = nn.Sequential(
            nn.Linear(n_continuous, 16),
            nn.ReLU(),
            nn.LayerNorm(16),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        
        # 2) Binary sensor path
        self.binary_encoder = nn.Sequential(
            nn.Linear(n_binary, 8),
            nn.ReLU(),
            nn.LayerNorm(8),
            nn.Linear(8, 4),
            nn.ReLU()
        )
        
        # 3) Event indicators path
        self.event_encoder = nn.Sequential(
            nn.Linear(n_event_binary, 8),
            nn.ReLU(),
            nn.LayerNorm(8),
            nn.Linear(8, 4),
            nn.ReLU()
        )
        
        # Combine latent representations
        combined_dim = 8 + 4 + 4
        
        self.bottleneck = nn.Sequential(
            nn.Linear(combined_dim, 8),
            nn.ReLU(),
            nn.Linear(8, bottleneck_dim)
        )
        
        # --- Decoder ---
        
        self.decoder_pre = nn.Sequential(
            nn.Linear(bottleneck_dim, 8),
            nn.ReLU(),
            nn.Linear(8, combined_dim),
            nn.ReLU()
        )
        
        # Decoders for each feature type
        self.cont_decoder = nn.Linear(8, n_continuous)          # linear
        self.binary_decoder = nn.Sequential(                    # sigmoid
            nn.Linear(4, n_binary),
            nn.Sigmoid()
        )
        self.event_decoder = nn.Sequential(                     # sigmoid
            nn.Linear(4, n_event_binary),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Split input
        cont = x[:, :self.n_cont]
        binary = x[:, self.n_cont:self.n_cont+self.n_bin]
        events = x[:, -self.n_evt:]
        
        # Encode
        z_cont = self.cont_encoder(cont)
        z_bin = self.binary_encoder(binary)
        z_evt = self.event_encoder(events)
        
        z_cat = torch.cat([z_cont, z_bin, z_evt], dim=1)
        z = self.bottleneck(z_cat)
        
        # Decode
        d = self.decoder_pre(z)
        
        # Split latent parts
        d_cont = d[:, :8]
        d_bin = d[:, 8:12]
        d_evt = d[:, 12:16]
        
        recon_cont = self.cont_decoder(d_cont)
        recon_bin = self.binary_decoder(d_bin)
        recon_evt = self.event_decoder(d_evt)
        
        # Reassemble output
        recon = torch.cat([recon_cont, recon_bin, recon_evt], dim=1)
        
        return recon, z



class AutoencoderTrainer:
    def __init__(
        self,
        model,
        cont_idx,
        bin_idx,
        evt_idx,
        lr=1e-3,
        weight_decay=1e-4
    ):
        self.model = model
        
        # Store index groups
        self.cont_idx = cont_idx
        self.bin_idx = bin_idx
        self.evt_idx = evt_idx
        
        # Losses for different feature types
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()

        # Optimizer: create only if a model is provided to allow using this
        # class for utility methods (e.g. data preparation) before instantiating
        # a trainer with a real model.
        if self.model is not None:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            self.optimizer = None

    # ---------------------------------------------------------
    # Placeholder: Determine feature type for a dataframe column
    # ---------------------------------------------------------
    @staticmethod
    def detect_column_type(series):
        """
        Placeholder logic — replace with your own rules.
        
        Returns: "continuous", "binary", or "event"
        """
        unique_vals = series.dropna().unique()

        # Example heuristic:
        if len(unique_vals) == 2 and set(unique_vals) <= {0, 1}:
            return "binary"
        elif series.name.lower().startswith("event"):
            return "event"
        else:
            return "continuous"

    # ---------------------------------------------------------
    # Compute mixed loss
    # ---------------------------------------------------------
    def compute_loss(self, recon, batch):
        loss = 0.0
        
        # Continuous → MSE
        if self.cont_idx:
            loss += self.mse(recon[:, self.cont_idx], batch[:, self.cont_idx])

        # Binary → BCE
        if self.bin_idx:
            loss += self.bce(recon[:, self.bin_idx], batch[:, self.bin_idx])

        # Event → BCE
        if self.evt_idx:
            loss += self.bce(recon[:, self.evt_idx], batch[:, self.evt_idx])

        return loss

    # ---------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------
    def fit(self, train_loader, val_loader=None, epochs=200, patience=15, device="cuda"):
        self.model.to(device)
        best_val = float("inf")
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                batch = batch.to(device)
                recon, _ = self.model(batch)
                
                loss = self.compute_loss(recon, batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * len(batch)

            train_loss /= len(train_loader.dataset)
            
            # -----------------------------
            # Validation
            # -----------------------------
            if val_loader is not None:
                val_loss = self.evaluate(val_loader, device)
                
                if val_loss < best_val:
                    best_val = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                print(
                    f"Epoch {epoch}: "
                    f"train={train_loss:.6f}, val={val_loss:.6f}"
                )

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            else:
                print(f"Epoch {epoch}: train={train_loss:.6f}")


    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------
    def evaluate(self, loader, device="cuda"):
        self.model.eval()
        loss_sum = 0.0
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                recon, _ = self.model(batch)
                loss = self.compute_loss(recon, batch)
                loss_sum += loss.item() * len(batch)
        
        return loss_sum / len(loader.dataset)

    # ---------------------------------------------------------
    # Anomaly scoring (per-sample reconstruction error)
    # ---------------------------------------------------------
    def anomaly_scores(self, loader, device="cuda"):
        self.model.eval()
        scores = []
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                recon, _ = self.model(batch)

                # Use mixed loss per row
                # Compute per-sample error vector
                err = torch.zeros(len(batch), device=device)

                # Continuous MSE
                if self.cont_idx:
                    err += ((recon[:, self.cont_idx] - batch[:, self.cont_idx]) ** 2).mean(dim=1)

                # Binary BCE
                if self.bin_idx:
                    b = recon[:, self.bin_idx]
                    t = batch[:, self.bin_idx]
                    err += -(t * b.log() + (1 - t) * (1 - b).log()).mean(dim=1)

                # Event BCE
                if self.evt_idx:
                    b = recon[:, self.evt_idx]
                    t = batch[:, self.evt_idx]
                    err += -(t * b.log() + (1 - t) * (1 - b).log()).mean(dim=1)

                scores.extend(err.cpu().numpy())
        
        return scores


    def prepare_ov_1_data(self, file_location):
        # Load your data from the specified file location
        # DuckDB connection
        con = duckdb.connect()

        # Create a df with all the values of temperature sensors sorted by their timestamp
        #! Fix column names in query
        df = con.execute(f"""
            SELECT 
            "stream:timestamp",
            "concept:name",
            "stream:system" || '_' || "stream:observation" AS "sensor:id",
            "stream:value"
            FROM read_parquet('{file_location}')
            WHERE "org:resource" = 'ov_1'
            AND "stream:observation" NOT LIKE '%NFC%'
            AND "stream:observation" NOT LIKE '%Current_State%'
            GROUP BY "stream:observation", "stream:system", "stream:value", "stream:timestamp", "concept:name"
            ORDER BY "stream:timestamp" ASC
        """).df()
        # ensure timestamp column is actual datetime objects for safe comparison with datetime.datetime
        df['stream:timestamp'] = pd.to_datetime(df['stream:timestamp'])
        
        print(" Data loaded. Reshaping...")

        # Change the data to include same timestamps in a single row
        # Create a new dataframe

        """
        Now we assign sensor values to timestamps

        How:
        Since a large portion of the data is already synchronized, we can use the time intervals given by 
        these synchronized values and from other sensors we just add the closest value
        """

        # One of the synchronized sensors, which will be used to get the timestamps
        reference_sensor_id = 'http://iot.uni-trier.de/FTOnto#CompressorPowerLevel_http://iot.uni-trier.de/FTOnto#OV_1_WT_1_Compressor_8'

        #? Is the ref dataframe directly accessible as id? So can we just copy values to the other df like here?
        # Split reference and others
        ref = df[df["sensor:id"] == reference_sensor_id].sort_values("stream:timestamp")
        others = df[df["sensor:id"] != reference_sensor_id].sort_values("stream:timestamp")

        # Start wide dataframe with reference sensor
        wide = ref[["stream:timestamp"]].drop_duplicates().reset_index(drop=True)
        wide[reference_sensor_id] = ref["stream:value"].values
        

        events = df["concept:name"].unique().tolist() # Binary value per event to indicate which event is happening

        # Create binary columns for each event
        for event in events:
            wide[f"event:{event}"] = (ref["concept:name"] == event).astype(int).values

        print(" Initial sensor added with timestamps and event data. Merging other sensors...")
        
        # Process each other sensor
        for sid, sdf in others.groupby("sensor:id"):

            if sid not in wide.columns:
                continue  # Skip sensors not in the learning dataframe

            # asof merge to find nearest timestamp
            merged = pd.merge_asof(
                left=wide[["stream:timestamp"]],
                right=sdf.sort_values("stream:timestamp"),
                left_on="stream:timestamp",
                right_on="stream:timestamp",
                direction="nearest"
            )
            wide[sid] = merged["stream:value"].values

        return wide

    def detect_column_types(df):
        cont_idx = []
        bin_idx = []
        evt_idx = []

        for i, col in enumerate(df.columns):
            series = df[col]
            unique_vals = series.dropna().unique()

            # Event columns start with 'event:'
            if col.startswith("event:"):
                evt_idx.append(i)
            
            # Binary detection (generic)
            elif len(unique_vals) <= 2 and set(unique_vals) <= {0, 1}:
                bin_idx.append(i)

            # Timestamps are NOT model features → ignore here
            elif "timestamp" in col:
                continue

            # Everything else = continuous
            else:
                cont_idx.append(i)

        return cont_idx, bin_idx, evt_idx


if __name__ == "__main__":

    from torch.utils.data import Dataset, DataLoader
    
    print("Starting autoencoder test...")

    class TensorDatasetWrapper(Dataset):
        def __init__(self, tensor):
            self.tensor = tensor
        def __len__(self):
            return self.tensor.size(0)
        def __getitem__(self, idx):
            return self.tensor[idx]

    print("Preparing data...")

    # Get file location
    parquet_directory = os.path.join(os.getcwd(), "Data", "IoT enriched event log paper", "20130794", "Cleaned Event Log", "parquet")
    file_location = os.path.join(parquet_directory, "all_combined_new.parquet")

    # 1) Load data
    trainer_for_methods = AutoencoderTrainer(
        model=None, cont_idx=[], bin_idx=[], evt_idx=[]
    )
    df = trainer_for_methods.prepare_ov_1_data(file_location)

    # 2) Remove timestamp column for training
    df_model = df.drop(columns=["stream:timestamp"])

    print("Detecting column types...")

    # 3) Detect column types
    cont_idx, bin_idx, evt_idx = AutoencoderTrainer.detect_column_types(df_model)
    print("Continuous:", cont_idx)
    print("Binary:", bin_idx)
    print("Event:", evt_idx)

    print("Preparing dataset and dataloaders...")

    # 4) Convert to tensor
    X = torch.tensor(df_model.values, dtype=torch.float32)

    # 5) Create dataset + loader
    dataset = TensorDatasetWrapper(X)
    n = len(dataset)
    n_train = int(n * 0.8)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n - n_train])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False)

    print("Initializing autoencoder...")

    # 6) Autoencoder
    model = MixedTabularAutoencoder(
        n_continuous=len(cont_idx),
        n_binary=len(bin_idx),
        n_event_binary=len(evt_idx),
        bottleneck_dim=3
    )

    print("Starting training...")
    
    # 7) Trainer
    trainer = AutoencoderTrainer(
        model=model,
        cont_idx=cont_idx,
        bin_idx=bin_idx,
        evt_idx=evt_idx
    )

    print("Training autoencoder...")

    # 8) Train
    trainer.fit(train_loader, val_loader, epochs=200, patience=15, device="cuda")

    print("Computing anomaly scores...")

    # 9) Compute anomaly scores
    scores = trainer.anomaly_scores(val_loader)


    print("Anomaly scores for validation set:")
    print(scores)