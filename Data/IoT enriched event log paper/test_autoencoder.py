import torch
import torch.nn as nn
import torch.optim as optim
import duckdb
import pandas as pd
import os

import torch
import torch.nn as nn

class MixedTabularAutoencoder(nn.Module):
    def __init__(self, n_continuous, n_binary, n_event_binary, bottleneck_dim=5):
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

                # Binary BCE (use stable BCELoss instead of manual log)
                if self.bin_idx:
                    b = recon[:, self.bin_idx]
                    t = batch[:, self.bin_idx]
                    # Clamp to avoid log(0) and log(-x)
                    b_safe = torch.clamp(b, min=1e-7, max=1-1e-7)
                    err += -(t * b_safe.log() + (1 - t) * (1 - b_safe).log()).mean(dim=1)

                # Event BCE (use stable BCELoss instead of manual log)
                if self.evt_idx:
                    b = recon[:, self.evt_idx]
                    t = batch[:, self.evt_idx]
                    # Clamp to avoid log(0) and log(-x)
                    b_safe = torch.clamp(b, min=1e-7, max=1-1e-7)
                    err += -(t * b_safe.log() + (1 - t) * (1 - b_safe).log()).mean(dim=1)

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
        
        print(f"    Loaded {len(df)} rows from {file_location}\n")
        
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
        reference_sensor_id = "http://iot.uni-trier.de/FTOnto#OV_1_WT_1_Compressor_8_http://iot.uni-trier.de/FTOnto#CompressorPowerLevel"

        #? Is the ref dataframe directly accessible as id? So can we just copy values to the other df like here?
        # Split reference and others
        ref = df[df["sensor:id"] == reference_sensor_id].sort_values("stream:timestamp")
        others = df[df["sensor:id"] != reference_sensor_id].sort_values("stream:timestamp")

        # Reset index for safe alignment and display the reference dataframe
        ref = ref.reset_index(drop=True)
        print("Reference dataframe preview (first 10 rows):")
        print(ref.head(10))
        print(f"Reference dataframe shape: {ref.shape}\n")

        # Initialize 'wide' with the reference timestamps and the reference sensor values
        wide = pd.DataFrame({
            "stream:timestamp": ref["stream:timestamp"].values,
            reference_sensor_id: ref["stream:value"].values
        })

        # Show the initialized wide head so you can confirm alignment before merging others
        print("Initialized 'wide' preview:")
        print(wide.head(10))

        events = df["concept:name"].unique().tolist() # Binary value per event to indicate which event is happening

        # Create binary columns for each event
        for event in events:
            wide[f"event:{event}"] = (ref["concept:name"] == event).astype(int).values

        print(" Initial sensor added with timestamps and event data. Merging other sensors...")
        
        for sensor_id in others["sensor:id"].unique():
            sensor_data = others[others["sensor:id"] == sensor_id][["stream:timestamp", "stream:value"]].reset_index(drop=True)
            
            print(f"  Merging sensor: {sensor_id} with {len(sensor_data)} entries")
            
            # For each timestamp in wide, find the closest timestamp in sensor_data
            sensor_values = []
            sensor_timestamps = sensor_data["stream:timestamp"].values
            sensor_values_list = sensor_data["stream:value"].values
            
            print(f"\n   Searching closest timestamps for {len(wide)} entries...")
            
            for ts in wide["stream:timestamp"]:
                # Find index of closest timestamp
                pos = sensor_data["stream:timestamp"].searchsorted(ts)

                if pos == 0:
                    sensor_values.append(sensor_values_list[0])
                elif pos >= len(sensor_timestamps):
                    sensor_values.append(sensor_values_list[-1])
                else:
                    left_ts = pd.Timestamp(sensor_timestamps[pos - 1])
                    right_ts = pd.Timestamp(sensor_timestamps[pos])
                    # choose the closer timestamp
                    if abs(ts - left_ts) <= abs(right_ts - ts):
                        sensor_values.append(sensor_values_list[pos - 1])
                    else:
                        sensor_values.append(sensor_values_list[pos])
                        
            print(f"   Merged {len(sensor_values)} values for sensor {sensor_id}")
            
            wide[sensor_id] = sensor_values

        return wide

    @staticmethod
    def detect_column_types(df):
        cont_idx = []
        bin_idx = []
        evt_idx = []

        for i, col in enumerate(df.columns):
            
            # Timestamps are NOT model features → ignore as early as possible
            if "timestamp" in col:
                continue

            # Event columns start with 'event:'
            if col.startswith("event:"):
                evt_idx.append(i)
                continue

            series = df[col]
            print(f" Sample values: {series.unique()[:5]}")
            unique_vals = series.dropna().unique()

            # Robust binary detection:
            #  - Try to coerce unique values to numeric and check subset of {0.0,1.0}
            #  - Treat boolean dtype as binary
            num_vals = pd.to_numeric(pd.Series(unique_vals), errors="coerce").dropna().unique()
            print(f" Unique values: {unique_vals}, Numeric values: {num_vals}")
            if len(num_vals) > 0 and set(num_vals).issubset({0.0, 1.0}):
                bin_idx.append(i)
            elif pd.api.types.is_bool_dtype(series):
                bin_idx.append(i)
            else:
                # Everything else = continuous
                cont_idx.append(i)

        return cont_idx, bin_idx, evt_idx
    



if __name__ == "__main__":

    from torch.utils.data import Dataset, DataLoader
    import numpy as np
    
    print("Starting autoencoder test...\n")

    class TensorDatasetWrapper(Dataset):
        def __init__(self, tensor):
            self.tensor = tensor
        def __len__(self):
            return self.tensor.size(0)
        def __getitem__(self, idx):
            return self.tensor[idx]

    print("Preparing data...\n")

    # Get file location
    parquet_directory = os.path.join(os.getcwd(), "Data", "IoT enriched event log paper", "20130794", "Cleaned Event Log", "parquet")
    file_location = os.path.join(parquet_directory, "all_combined_with_synthetic.parquet")

    # 1) Load data
    trainer_for_methods = AutoencoderTrainer(
        model=None, cont_idx=[], bin_idx=[], evt_idx=[]
    )
    df = trainer_for_methods.prepare_ov_1_data(file_location)

    # 2) Remove timestamp column for training
    df_model = df.drop(columns=["stream:timestamp"])
    
    # Print the first row of the df
    print("\nFirst row of the prepared data:")
    print(df_model)

    print("\nDetecting column types...\n")

    # 3) Detect column types
    cont_idx, bin_idx, evt_idx = AutoencoderTrainer.detect_column_types(df_model)
    print("Continuous:", cont_idx)
    print("Binary:", bin_idx)
    print("Event:", evt_idx)
    
    
    # Map detected index lists (which refer to original df_model column positions)
    # to column names so we can coerce/convert by name and reorder the dataframe
    cont_cols = [df_model.columns[i] for i in cont_idx] if cont_idx else []
    bin_cols = [df_model.columns[i] for i in bin_idx] if bin_idx else []
    evt_cols = [df_model.columns[i] for i in evt_idx] if evt_idx else []

    if bin_cols:
        for col in bin_cols:
            # Coerce to numeric, round to 0/1 and convert to float so NaNs become np.nan (compatible with numpy)
            df_model[col] = pd.to_numeric(df_model[col], errors="coerce").round().astype(float)
        print(f"Converted {len(bin_cols)} binary columns to numeric 0/1 floats (NaN preserved).")
    else:
        print("No binary columns to convert.")

    # Reorder columns to match model expectation: continuous, binary, event
    ordered_cols = cont_cols + bin_cols + evt_cols
    df_model_reordered = df_model[ordered_cols]

    # Ensure all remaining columns are numeric and use a concrete numpy-compatible dtype (float32)
    df_numeric = df_model_reordered.apply(pd.to_numeric, errors='coerce').astype('float32')

    print("\nPreparing dataset and dataloaders...\n")

    # 4) Convert to tensor
    X = torch.tensor(df_numeric.values, dtype=torch.float32)

    # 5) Create dataset + loader
    dataset = TensorDatasetWrapper(X)
    n = len(dataset)
    n_train = int(n * 0.8)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n - n_train])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False)

    print("Initializing autoencoder...\n")

    # 6) Autoencoder: sizes derived from counts of each column group
    model = MixedTabularAutoencoder(
        n_continuous=len(cont_cols),
        n_binary=len(bin_cols),
        n_event_binary=len(evt_cols),
        bottleneck_dim=3
    )

    print("Starting training...")
    
    # 7) Trainer: pass indices relative to the reordered tensor layout
    cont_positions = list(range(0, len(cont_cols)))
    bin_positions = list(range(len(cont_cols), len(cont_cols) + len(bin_cols)))
    evt_positions = list(range(len(cont_cols) + len(bin_cols), len(cont_cols) + len(bin_cols) + len(evt_cols)))

    trainer = AutoencoderTrainer(
        model=model,
        cont_idx=cont_positions,
        bin_idx=bin_positions,
        evt_idx=evt_positions
    )

    print("Training autoencoder...")

    # 8) Select device (GPU if available, otherwise CPU) and train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    trainer.fit(train_loader, val_loader, epochs=250, patience=15, device=device)

    print("Computing anomaly scores...")

    # 9) Compute anomaly scores (use same device)
    scores = trainer.anomaly_scores(val_loader, device=device)


    arr = np.array(scores)
    n = len(arr)
    if n == 0:
        print("No anomaly scores to display.")
    else:
        mean = arr.mean()
        median = np.median(arr)
        std = arr.std()
        p90 = np.percentile(arr, 90)
        p95 = np.percentile(arr, 95)
        p99 = np.percentile(arr, 99)

        print("Anomaly score summary:")
        print(f" count = {n}")
        print(f" mean  = {mean:.6f}, median = {median:.6f}, std = {std:.6f}")
        print(f" min   = {arr.min():.6f}, max    = {arr.max():.6f}")
        print(f" percentiles: 90% = {p90:.6f}, 95% = {p95:.6f}, 99% = {p99:.6f}")

        topk = min(10, n)
        top_idx = np.argsort(-arr)[:topk]

        print(f"\nTop {topk} anomalies (index in validation set, score):")
        for rank, idx in enumerate(top_idx, start=1):
            print(f" {rank:2d}. idx = {int(idx):4d}, score = {arr[idx]:.6f}")

        # Suggested threshold and counts
        suggested_threshold = p95
        n_above = int((arr > suggested_threshold).sum())
        print(f"\nSuggested threshold = 95th percentile ({suggested_threshold:.6f})")
        print(f" Samples above threshold: {n_above} ({n_above / n * 100:.1f}%)")

    # =====================================================================
    # Inspect top anomalies and random samples with their reconstructions
    # =====================================================================
    print("\n" + "="*80)
    print("TOP 3 MOST ANOMALOUS SAMPLES")
    print("="*80)

    # Get indices of top 3 anomalies
    top_3_indices = np.argsort(-arr)[:3]

    trainer.model.eval()
    with torch.no_grad():
        for rank, val_idx in enumerate(top_3_indices, start=1):
            # Get the actual sample from validation set
            sample_tensor = X[val_ds.indices[val_idx]].unsqueeze(0).to(device)
            recon_tensor, _ = trainer.model(sample_tensor)

            original = sample_tensor.squeeze().cpu().numpy()
            reconstruction = recon_tensor.squeeze().cpu().numpy()
            error = arr[val_idx]

            print(f"\n--- Anomaly #{rank} (Val Index: {val_idx}, Score: {error:.6f}) ---")
            print(f"Original:       {original}")
            print(f"Reconstruction: {reconstruction}")
            print(f"Difference:     {np.abs(original - reconstruction)}")

    # =====================================================================
    # Random 5 samples with their reconstructions
    # =====================================================================
    print("\n" + "="*80)
    print("5 RANDOM NORMAL SAMPLES")
    print("="*80)

    random_indices = np.random.choice(len(val_ds), size=5, replace=False)

    with torch.no_grad():
        for rank, val_idx in enumerate(random_indices, start=1):
            sample_tensor = X[val_ds.indices[val_idx]].unsqueeze(0).to(device)
            recon_tensor, _ = trainer.model(sample_tensor)

            original = sample_tensor.squeeze().cpu().numpy()
            reconstruction = recon_tensor.squeeze().cpu().numpy()
            error = arr[val_idx]

            print(f"\n--- Random Sample #{rank} (Val Index: {val_idx}, Score: {error:.6f}) ---")
            print(f"Original:       {original}")
            print(f"Reconstruction: {reconstruction}")
            print(f"Difference:     {np.abs(original - reconstruction)}")

