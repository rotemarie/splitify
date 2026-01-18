import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import AudioSegmentDataset
from learning.singlestem.model import UNet
from stft_utils import stft_waveform, mag_and_phase_from_complex, istft_from_mag_phase


class SeparatorTrainer:
    """
    Trainer for U-Net audio source separation in the time-frequency domain.

    Workflow:
    waveform → STFT → magnitude → U-Net (mask) → predicted magnitude →
    ISTFT(predicted_mag × mixture_phase) → predicted waveform
    → time-domain L1 loss(target, prediction)
    """

    def __init__(
        self,
        workspace,
        index_pkl,
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
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.checkpoint_every = checkpoint_every
        self.n_fft = n_fft
        self.hop_length = hop_length

        os.makedirs(workspace, exist_ok=True)

        # --------------------
        # 1. Load dataset
        # --------------------
        print(f"Loading dataset from {index_pkl}")
        self.dataset = AudioSegmentDataset(index_pkl)
        self.dataloader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True
        )

        # --------------------
        # 2. Initialize model, loss, optimizer
        # --------------------
        print(f"Initializing model on {device}")
        self.model = UNet(in_channels=1).to(device)
        self.criterion = nn.L1Loss()  # Time-domain MAE
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self):
        print(f"Starting training for {self.num_epochs} epochs.")
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0

            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
            for batch in pbar:
                mixture = batch["mixture"].to(self.device)  # (B, 1, T)
                target = batch["target"].to(self.device)

                mixture = mixture.squeeze(1)
                target = target.squeeze(1)

                # --------------------
                # 1. STFT
                # --------------------
                mix_complex = stft_waveform(mixture, n_fft=self.n_fft, hop_length=self.hop_length)
                tgt_complex = stft_waveform(target, n_fft=self.n_fft, hop_length=self.hop_length)

                # Magnitude and phase
                mix_mag, mix_phase = mag_and_phase_from_complex(mix_complex)
                tgt_mag, _ = mag_and_phase_from_complex(tgt_complex)

                # --------------------
                # 2. Forward pass
                # --------------------
                mix_mag = mix_mag.unsqueeze(1)  # (B, 1, F, T)
                tgt_mag = tgt_mag.unsqueeze(1)

                pred_mask = self.model(mix_mag)
                pred_mag = pred_mask * mix_mag

                # --------------------
                # 3. Reconstruct waveform
                # --------------------
                pred_waveform = istft_from_mag_phase(
                    pred_mag,  # Already has channel dimension (B, 1, F, T)
                    mix_phase.unsqueeze(1),  # Add channel dimension (B, 1, F, T)
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    length=mixture.shape[-1],
                ).squeeze(1)  # Remove channel dimension -> (B, T)

                # --------------------
                # 4. Compute loss & update
                # --------------------
                target = target[:, :pred_waveform.shape[-1]]
                loss = self.criterion(pred_waveform, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            epoch_loss = running_loss / len(self.dataloader)
            print(f"Epoch {epoch+1}/{self.num_epochs} - Loss: {epoch_loss:.4f}")

            # --------------------
            # 5. Save checkpoint
            # --------------------
            if (epoch + 1) % self.checkpoint_every == 0:
                ckpt_path = os.path.join(self.workspace, f"checkpoint_epoch{epoch+1}.pt")
                torch.save(self.model.state_dict(), ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train U-Net for source separation")
    parser.add_argument("--workspace", type=str, required=True, help="Path to workspace dir (for checkpoints)")
    parser.add_argument("--index_pkl", type=str, required=True, help="Path to indexed .pkl file")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint_every", type=int, default=1)
    args = parser.parse_args()

    trainer = SeparatorTrainer(
        workspace=args.workspace,
        index_pkl=args.index_pkl,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        checkpoint_every=args.checkpoint_every,
    )
    trainer.train()


'''
model operates in the time-frequency (TF) domain, i.e. on spectrogram magnitudes.
dataset provides waveform segments (mixture and target stems).

So the pipeline is:
waveform → STFT → magnitude spectrogram → model → mask
→ predicted magnitude = mask × mixture magnitude
→ ISTFT(predicted_mag, mixture_phase) → predicted waveform
→ loss(predicted_waveform, target_waveform)

Loss computed directly on reconstructed waveform

Checkpoints: Saved every checkpoint_every epochs to your workspace directory

python train.py --workspace ./checkpoints --index_pkl ./index.pkl --epochs 10
'''

