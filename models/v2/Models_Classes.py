import torch.nn as nn
import torch

### MLPBaseline ###
class MLPBaseline(nn.Module):
    """
    Simple MLP baseline model. Flattens the observed sequence and predicts the future trajectory.
    """
    def __init__(self, obs_len, pred_len, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(obs_len * input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pred_len * output_dim),
            nn.ReLU()
        )

    def forward(self, obs_traj):
        batch = obs_traj.size(0)
        x = obs_traj.view(batch, -1)  # Flatten input
        out = self.mlp(x)
        fut = out.view(batch, self.pred_len, self.output_dim)
        return fut
    

### Enc-Dec with noise (PECNet inspired) ###
class EncDec_Noise(nn.Module):
    """ 
    One latent space for all features. use this for xy or xy+zones+ttc 
    """
    def __init__(self, obs_len, pred_len, input_dim, output_dim, latent_dim, hidden_dim):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        # Encoder: encodes all
        self.encoder = nn.Sequential(
            nn.Linear(obs_len * input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # Latent variable
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        # Decoder: decodes latent + encoded obs to future trajectory
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            # nn.Linear(hidden_dim, pred_len * input_dim) # this was when input dim was only position
            nn.Linear(hidden_dim, pred_len * output_dim)
        )
    def forward(self, obs_traj):
        batch = obs_traj.size(0)               # obs_traj: (batch, obs_len, input_dim)
        obs_flat = obs_traj.view(batch, -1)    # obs_flat: (batch, obs_len * input_dim)
        enc = self.encoder(obs_flat)
        z = torch.randn(batch, self.latent_dim, device=obs_traj.device)
        enc_z = torch.cat([enc, z], dim=1)              # concatenate encoded obs with latent variable
        fut_flat = self.decoder(enc_z)
        fut = fut_flat.view(batch, self.pred_len, self.output_dim)
        return fut


class EncDec_Noise_1(nn.Module):
    """
    Seperate latent space for zones and ttc
    """
    def __init__(self, obs_len, pred_len, input_dim, output_dim, hidden_dim,
                 zone_dim, ttc_dim, zone_latent_dim, ttc_latent_dim):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.zone_dim = zone_dim
        self.ttc_dim = ttc_dim
        self.zone_latent_dim = zone_latent_dim
        self.ttc_latent_dim = ttc_latent_dim
        
        # Encoder for all features
        self.encoder_all = nn.Sequential(
            nn.Linear(obs_len * input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # Latent for zones only
        self.to_latent_zone = nn.Sequential(
            nn.Linear(obs_len * zone_dim, zone_latent_dim),
            nn.ReLU()
        )
        # Latent for ttc only
        self.to_latent_ttc = nn.Sequential(
            nn.Linear(obs_len * ttc_dim, ttc_latent_dim),
            nn.ReLU()
        )
        # Decoder: decodes concatenated encodings to future trajectory
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + zone_latent_dim + ttc_latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pred_len * output_dim)
        )

    def forward(self, obs_traj):
        batch = obs_traj.size(0)                # obs_traj: (batch, obs_len, input_dim)
        obs_flat = obs_traj.view(batch, -1)     # (batch, obs_len * input_dim)
        enc_all = self.encoder_all(obs_flat)
        # Extract zones and ttc from obs_traj
        zone_indx = [2,3,4,5]
        obs_zone = obs_traj[:, :, zone_indx]      # (batch, obs_len , zone_dim)
        obs_flat_zone = obs_zone.view(batch, -1)  # (batch, obs_len * zone_dim)
        
        ttc_indx = [6,7,8,9, 10,11,12,13,14]
        obs_ttc = obs_traj[:, :, ttc_indx]      # (batch, obs_len , ttc_dim)
        obs_flat_ttc = obs_ttc.view(batch, -1)  # (batch, obs_len * ttc_dim)
        
        enc_all = self.encoder_all(obs_flat)
        enc_zone = self.to_latent_zone(obs_flat_zone)
        enc_ttc = self.to_latent_ttc(obs_flat_ttc)

        z_zone = torch.randn(batch, self.zone_latent_dim, device=obs_zone.device)
        z_ttc  = torch.randn(batch, self.ttc_latent_dim, device=obs_ttc.device)
        # Concatenate all encodings
        enc_cat = torch.cat([enc_all,  z_zone,  z_ttc], dim=1)
        fut_flat = self.decoder(enc_cat)
        fut = fut_flat.view(batch, self.pred_len, self.output_dim)
        return fut
    

### LSTM ###
class LSTM(nn.Module):
    """
    For all features or a part of them (xy / xy+zone / xy+zone+ttc).
    all data (all features) is inserted directly into LSTM layers
    """
    def __init__(self, obs_len, pred_len, input_dim, output_dim, hidden_dim, lstm_layers):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, lstm_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, obs_traj):
        # obs_traj: (batch, obs_len, input_dim)
        batch_size = obs_traj.size(0)
        device = obs_traj.device
        h0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=device)
        c0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=device)
        out, (hn, cn) = self.lstm(obs_traj, (h0, c0))  # out: (batch, obs_len, hidden_dim)
        last_output = out[:, -1, :].unsqueeze(1)  # (batch, 1, hidden_dim)
        hn = hn
        cn = cn
        # Start with the last observed position as input
        last_pos = obs_traj[:, -1:, :]  # (batch, 1, input_dim)
        preds = []
        for _ in range(self.pred_len):
            out, (hn, cn) = self.lstm(last_pos, (hn, cn))  # out: (batch, 1, hidden_dim)
            pred_xy = self.fc(out)  # (batch, 1, output_dim)
            preds.append(pred_xy)
            # Prepare next input: replace x, y in last_pos with prediction, keep other features
            if self.input_dim == self.output_dim:
                last_pos = pred_xy  # Only x, y as input
            else:
                last_pos = last_pos.clone()
                last_pos[:, :, :self.output_dim] = pred_xy  # Replace x, y, keep other features

        fut = torch.cat(preds, dim=1)  # (batch, pred_len, output_dim)
        return fut


### LSTM_Pooling ###
class LSTM_Pooling(nn.Module):
    """
    Zones and TTC are pooled (encoded) seperatly or combined.
    Then pooled scene and social latent spaces are concatenated with xy
    and then inserted to LSTM.
    """
    def __init__(self, obs_len, pred_len, input_dim, output_dim, 
                 lstm_layers,
                 zone_dim, ttc_dim, scene_pool_dim, social_pool_dim, 
                 lstm_hidden_dim):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.zone_dim = zone_dim
        self.ttc_dim = ttc_dim
        self.scene_pool_dim = scene_pool_dim
        self.social_pool_dim = social_pool_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers
        # Scene pooling - zones
        self.scene_pool = nn.Sequential(
            nn.Linear(obs_len * zone_dim,       (obs_len * zone_dim)*2  ),
            nn.ReLU(),
            nn.Linear((obs_len * zone_dim)*2,    scene_pool_dim),
            nn.ReLU()
        )
        # Social encoder for ttc
        self.social_pool = nn.Sequential(
            nn.Linear(obs_len * ttc_dim,        (obs_len * ttc_dim)*2    ),
            nn.ReLU(),
            nn.Linear((obs_len * ttc_dim)*2,     social_pool_dim),
            nn.ReLU()
        )
        # combined encoder for zones and ttc
        self.combined_encoder = nn.Sequential(
            nn.Linear(obs_len * (zone_dim + ttc_dim), (obs_len * (zone_dim + ttc_dim))*2),
            nn.ReLU(),
            nn.Linear((obs_len * (zone_dim + ttc_dim))*2, scene_pool_dim + social_pool_dim ),
            nn.ReLU()
        )

        # Position xy encoder
        self.xy_encoder = nn.Sequential(
            nn.Linear(obs_len * 2,            (obs_len * 2)*2),
            nn.ReLU(),
            nn.Linear((obs_len * 2)*2,         lstm_hidden_dim),
            nn.ReLU()
        )
        # LSTM layers
        self.lstm = nn.LSTM(2 + scene_pool_dim + social_pool_dim, lstm_hidden_dim, lstm_layers, batch_first=True)
        self.fc =   nn.Linear(lstm_hidden_dim, output_dim)

    def forward(self, obs_traj):
        batch = obs_traj.size(0)   # obs_traj: (batch, obs_len, input_dim)
        device = obs_traj.device
        # define features data from batch 
        xy =    obs_traj[:,  :, :2]  # (batch, obs_len, 2)
        zones = obs_traj[:,  :,  2: (2 + self.zone_dim) ]  
        ttc =   obs_traj[:,  :,     (2 + self.zone_dim) : 2 + self.zone_dim + self.ttc_dim]  
        # Scene pooling and Social pooling
        zones_flat = zones.reshape(batch, -1)
        scene_latent = self.scene_pool(zones_flat)  
        ttc_flat = ttc.reshape(batch, -1)
        social_latent = self.social_pool(ttc_flat)

        scene_latent = scene_latent.unsqueeze(1).repeat(1, xy.size(1), 1)
        social_latent = social_latent.unsqueeze(1).repeat(1, xy.size(1), 1)
        # Concatenate xy, scene, and social features
        lstm_input = torch.cat([xy, scene_latent, social_latent], dim=2)  # dim=0 concatenates along the batch dimension
                                                                          # dim=1 concatenates along the sequence length
                                                                          # dim=2 concatenates along the feature dimension

        scene_last = scene_latent[:, -1:, :]      # (batch, 1, scene_pool_dim)
        social_last = social_latent[:, -1:, :]    # (batch, 1, social_pool_dim)
        # LSTM
        h0 = torch.zeros(self.lstm.num_layers, batch, self.lstm.hidden_size, device=device)
        c0 = torch.zeros(self.lstm.num_layers, batch, self.lstm.hidden_size, device=device)
        out, (hn, cn) = self.lstm(lstm_input, (h0, c0))
        # Predict future trajectory autoregressively
        preds = []
        last_xy = xy[:, -1:, :]  # (batch, 1, 2)
        for _ in range(self.pred_len):
            step_input = torch.cat([last_xy, scene_last, social_last], dim=2)
            out, (hn, cn) = self.lstm(step_input, (hn, cn))
            pred_xy = self.fc(out)  # (batch, 1, output_dim)
            preds.append(pred_xy)
            last_xy = pred_xy  # Feed prediction as next input
        fut = torch.cat(preds, dim=1)  # (batch, pred_len, output_dim)
        return fut


### Transformer ###
class TrajTransformer(nn.Module):
    """
    Basic Transformer made with copilot
    """
    def __init__(self, obs_len, pred_len, input_dim, output_dim, 
                 d_model, nhead, num_layers, 
                 dim_feedforward, dropout):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        # Input embedding for encoder and decoder
        self.input_embed = nn.Linear(input_dim, d_model)
        self.output_embed = nn.Linear(output_dim, d_model)
        # Positional encodings (learnable)
        self.pos_encoder = nn.Parameter(torch.zeros(1, obs_len, d_model))
        self.pos_decoder = nn.Parameter(torch.zeros(1, pred_len, d_model))
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        # Output projection
        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, obs_traj):
        # obs_traj: (batch, obs_len, input_dim)
        batch = obs_traj.size(0)
        device = obs_traj.device
        # Encoder input
        enc_in = self.input_embed(obs_traj) + self.pos_encoder  # (batch, obs_len, d_model)
        # Decoder input: start with zeros, then feed previous outputs (teacher forcing not used here)
        dec_in = torch.zeros(batch, self.pred_len, self.output_dim, device=device)
        dec_in = self.output_embed(dec_in) + self.pos_decoder  # (batch, pred_len, d_model)
        # Transformer expects (batch, seq, d_model)
        memory = self.transformer.encoder(enc_in)
        out = self.transformer.decoder(dec_in, memory)
        # Project to output
        fut = self.output_proj(out)  # (batch, pred_len, output_dim)
        return fut
    

### Data Loader ###
import glob
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader, random_split

class LoadCSV_to_Dataset(Dataset):
    """Load data from csv and create sequences and batches that fits pytorch
    Cahnges zones and ttc values of df for better training.
    """
    def __init__(self, folder_path, obs_input_len, fut_output_len, input_features, output_features):
        self.samples = []
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

        for csv_file in csv_files:
            df = pd.read_csv(csv_file) 
            df = df[df['class'] == 'pedestrian'] # Only keep pedestrian rows
            
            # change zone and ttc -1 values to positive values
            # see step 12 to check the zones values if it normalized dist or just 0/1
            # for col in ['zone_zebra', 'zone_1', 'zone_2', 'zone_3']:
            #     df[col] = df[col].replace(-1, 2) # .replace(old_value, new_value)
            
            for col in ['ttc_1', 'ttc_2', 'ttc_3', 'ttc_4', 'ttc_5', 'ttc_6', 'ttc_7', 'ttc_8', 'ttc_9']:
                df[col] = df[col].replace(-1, 0) # .replace(old_value, new_value)

            for track_id, group in df.groupby('trackId'):
                group = group.sort_values('frame').reset_index(drop=True)
                # xy =    group[['xCenter', 'yCenter']].values                  # target feature → position
                xy =    group[output_features].values                           # target feature → position
                features = group[input_features].values                         # input features → position + zones + ttc         
                seq_len = obs_input_len + fut_output_len
                for i in range(len(xy) - seq_len + 1): # sliding window: technique to generate 
                                                       # multiple overlapping sequences from a  
                                                       # longer sequence of data
                    # obs =     xy[i:i+obs_input_len]
                    obs = features[i:i+obs_input_len]
                    fut = xy[i+obs_input_len:i+seq_len]
                    self.samples.append((obs, fut))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obs, fut = self.samples[idx]
        return torch.tensor(obs, dtype=torch.float32), torch.tensor(fut, dtype=torch.float32)
    

# for visualize evaluation with origin info
class LoadCSV_to_Dataset_with_Origin(Dataset):
    """ Same as LoadCSV_to_Dataset with Origin information.
        Contains origin info in addition to obs and fut: recordingId and trackId
    """
    def __init__(self, folder_path, obs_input_len, fut_output_len, input_features, output_features):
        self.samples = []
        self.origins = []  # Store origin info for each sample
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

        for csv_file in csv_files:
            df = pd.read_csv(csv_file) 
            df = df[df['class'] == 'pedestrian'] # Only keep pedestrian rows

            for col in ['ttc_1', 'ttc_2', 'ttc_3', 'ttc_4', 'ttc_5', 'ttc_6', 'ttc_7', 'ttc_8', 'ttc_9']:
                df[col] = df[col].replace(-1, 0)

            # Get recordingId from the dataframe (assumes all rows in file have the same recordingId)
            if 'recordingId' in df.columns:
                recording_id = df['recordingId'].iloc[0]
            else:
                recording_id = None  # or raise an error if you want

            for track_id, group in df.groupby('trackId'):
                group = group.sort_values('frame').reset_index(drop=True)
                xy = group[output_features].values
                features = group[input_features].values
                seq_len = obs_input_len + fut_output_len
                for i in range(len(xy) - seq_len + 1):
                    obs = features[i:i+obs_input_len]
                    fut = xy[i+obs_input_len:i+seq_len]
                    self.samples.append((obs, fut))
                    self.origins.append({
                        'recordingId': recording_id,
                        'trackId': track_id,
                        'start_idx': i,                      # i is window index
                        'first_frame': group.loc[i, 'frame']      # frame number of the first observation
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obs, fut = self.samples[idx]
        return torch.tensor(obs, dtype=torch.float32), torch.tensor(fut, dtype=torch.float32)

    def get_origin(self, idx):
        """Return the origin info for a given index (for analysis only)."""
        return self.origins[idx]