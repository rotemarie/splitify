# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import AudioSegmentDataset
from model import UNetMulti  # Multi-output model
from stft_utils import stft_waveform, mag_and_phase_from_complex, istft_from_mag_phase


class MultiStemTrainer:
    """
    Multi-output U-Net trainer for source separation.

    Pipeline:
        waveform (mixture) →
        STFT → magnitude →
        U-Net (multi-mask) →
        per-stem predicted magnitude →
        ISTFT(pred_mag × mixture_phase) →
        per-stem waveform →
        time-domain L1 loss averaged across stems
    """

    def __init__(
        self,
        workspace,
        index_pkl,
        stems=("vocals", "bass", "drums", "other"),
        num_epochs=5,
        batch_size=8,
        learning_rate=1e-3,
        device="cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_every=1,
        n_fft=1024,
        hop_length=512,
    ):
        self.workspace = workspace
        self.index_pkl = index_pkl
        self.stems = stems
        self.num_stems = len(stems)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.checkpoint_every = checkpoint_every
        self.n_fft = n_fft
        self.hop_length = hop_length

        os.makedirs(workspace, exist_ok=True)

        # --------------------
        # 1. Dataset
        # --------------------
        print(f"Loading dataset from {index_pkl}")
        self.dataset = AudioSegmentDataset(index_pkl)
        self.dataloader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True
        )

        # --------------------
        # 2. Model, loss, optimizer
        # --------------------
        print(f"Initializing multi-stem model ({self.num_stems} outputs) on {device}")
        self.model = UNetMulti(in_channels=1, n_outputs=self.num_stems).to(device)
        self.criterion = nn.L1Loss()  # Time-domain MAE loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self):
        print(f"Starting training for {self.num_epochs} epochs on {self.device}.")
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0

            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
            for batch in pbar:
                # Dataset returns per sample: (C, L) where C=1 for mono
                # DataLoader batches to: (B, C, L) = (B, 1, L) 
                mixture = batch["mixture"].to(self.device)  # (B, 1, T)
                targets = {k: v.to(self.device) for k, v in batch["targets"].items()}

                mixture = mixture.squeeze(1)  # (B, T)

                # --------------------
                # 1. STFT (convert to complex)
                # --------------------
                mix_complex = stft_waveform(mixture, n_fft=self.n_fft, hop_length=self.hop_length)
                mix_mag, mix_phase = mag_and_phase_from_complex(mix_complex)

                # mix_mag already has shape (B, C, F, T) where C=1, so no need to unsqueeze
                mix_mag_input = mix_mag  # (B, 1, F, T)

                # --------------------
                # 2. Forward pass
                # --------------------
                pred_masks = self.model(mix_mag_input)  # (B, num_stems, F, T)
                pred_mags = pred_masks * mix_mag_input  # broadcast over stems

                # --------------------
                # 3. Reconstruct waveforms per stem
                # --------------------
                total_loss = 0.0
                for i, stem_name in enumerate(self.stems):
                    if stem_name not in targets:
                        continue  # skip missing stems

                    tgt_wave = targets[stem_name].squeeze(1)  # (B, T)
                    pred_mag = pred_mags[:, i, :, :]          # (B, F, T)

                    # Reconstruct using magnitude and phase
                    pred_wave = istft_from_mag_phase(
                        pred_mag.unsqueeze(1),  # Add channel dimension (B, 1, F, T)
                        mix_phase,  # mix_phase already has shape (B, 1, F, T)
                        n_fft=self.n_fft,
                        hop_length=self.hop_length,
                        length=tgt_wave.shape[-1],
                    ).squeeze(1)  # Remove channel dimension -> (B, T)

                    # --------------------
                    # 4. Compute per-stem L1 loss
                    # --------------------
                    stem_loss = self.criterion(pred_wave, tgt_wave)
                    total_loss += stem_loss

                total_loss /= self.num_stems  # average over all stems

                # --------------------
                # 5. Backpropagation
                # --------------------
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                running_loss += total_loss.item()
                pbar.set_postfix({"loss": f"{total_loss.item():.4f}"})

            epoch_loss = running_loss / len(self.dataloader)
            print(f"Epoch {epoch+1}/{self.num_epochs} - Loss: {epoch_loss:.4f}")

            # --------------------
            # 6. Save checkpoint
            # --------------------
            if (epoch + 1) % self.checkpoint_every == 0:
                ckpt_path = os.path.join(self.workspace, f"checkpoint_epoch{epoch+1}.pt")
                torch.save(self.model.state_dict(), ckpt_path)
                print(f"✅ Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train multi-output U-Net for source separation")
    parser.add_argument("--workspace", type=str, required=True, help="Path to workspace for checkpoints")
    parser.add_argument("--index_pkl", type=str, required=True, help="Path to indexed .pkl file")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint_every", type=int, default=1)
    args = parser.parse_args()

    trainer = MultiStemTrainer(
        workspace=args.workspace,
        index_pkl=args.index_pkl,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        checkpoint_every=args.checkpoint_every,
    )
    trainer.train()
