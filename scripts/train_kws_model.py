"""
Train an ultra-compact DS-CNN (Depthwise Separable CNN) KWS Model
and export INT8 quantized weights to include/model_data.h
"""
import os
import glob
import wave
import numpy as np

def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)

def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

def build_mel_filterbank(num_bins=16, fft_size=512, sr=16000, lower_f=300.0, upper_f=7500.0):
    num_fft_bins = fft_size // 2 + 1
    lower_mel = hz_to_mel(lower_f)
    upper_mel = hz_to_mel(upper_f)
    mel_points = np.linspace(lower_mel, upper_mel, num_bins + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((fft_size + 1) * hz_points / sr).astype(int)

    fbank = np.zeros((num_bins, num_fft_bins), dtype=np.float32)
    for m in range(1, num_bins + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus + 1e-8)
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m + 1e-8)
    return fbank

def extract_log_mel_spectrogram(audio, sr=16000, frame_len=512, frame_step=256, num_mel=16):
    # Hamming window
    window = np.hamming(frame_len).astype(np.float32)
    num_frames = (len(audio) - frame_len) // frame_step + 1
    fbank = build_mel_filterbank(num_bins=num_mel, fft_size=frame_len, sr=sr)

    features = np.zeros((num_frames, num_mel), dtype=np.float32)
    for i in range(num_frames):
        start = i * frame_step
        frame = audio[start:start + frame_len] * window
        fft_res = np.fft.rfft(frame, n=frame_len)
        power_spec = (np.abs(fft_res) ** 2) / float(frame_len)
        mel_energies = np.dot(fbank, power_spec)
        features[i] = np.log(mel_energies + 1e-6)
    return features

def load_dataset(data_dir):
    classes = ['silence', 'unknown', 'aura', 'exit']
    X = []
    y = []

    for label_idx, cls_name in enumerate(classes):
        cls_folder = os.path.join(data_dir, cls_name)
        wav_files = glob.glob(os.path.join(cls_folder, "*.wav"))
        print(f"Loading {len(wav_files)} files for class '{cls_name}' (label {label_idx})...")
        for wav_path in wav_files:
            try:
                with wave.open(wav_path, 'rb') as wf:
                    raw = wf.readframes(wf.getnframes())
                    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                if len(audio) < 16000:
                    audio = np.pad(audio, (0, 16000 - len(audio)))
                elif len(audio) > 16000:
                    audio = audio[:16000]

                feats = extract_log_mel_spectrogram(audio)
                if feats.shape == (61, 16):
                    X.append(feats)
                    y.append(label_idx)
            except Exception as e:
                continue

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    # Normalize features
    mean = np.mean(X)
    std = np.std(X) + 1e-6
    X = (X - mean) / std
    return X, y, mean, std

def train_and_export():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader

    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    X, y, norm_mean, norm_std = load_dataset(data_dir)
    print(f"Total dataset shape: X={X.shape}, y={y.shape}")

    # Shuffle and split
    indices = np.random.permutation(len(X))
    split = int(0.85 * len(X))
    train_idx, val_idx = indices[:split], indices[split:]

    X_train, y_train = torch.tensor(X[train_idx]).unsqueeze(1), torch.tensor(y[train_idx])
    X_val, y_val = torch.tensor(X[val_idx]).unsqueeze(1), torch.tensor(y[val_idx])

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32, shuffle=False)

    class TinyDSCNN(nn.Module):
        def __init__(self):
            super().__init__()
            # Layer 1: Standard Conv (1 -> 16)
            self.conv1 = nn.Conv2d(1, 16, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1), bias=True)
            self.relu1 = nn.ReLU()
            # Layer 2: Depthwise Conv (16 -> 16)
            self.dw1 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=1, padding=1, groups=16, bias=True)
            self.relu2 = nn.ReLU()
            # Layer 3: Pointwise Conv (16 -> 24)
            self.pw1 = nn.Conv2d(16, 24, kernel_size=1, bias=True)
            self.relu3 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            # Layer 4: Depthwise Conv (24 -> 24)
            self.dw2 = nn.Conv2d(24, 24, kernel_size=(3, 3), stride=1, padding=1, groups=24, bias=True)
            self.relu4 = nn.ReLU()
            # Layer 5: Pointwise Conv (24 -> 32)
            self.pw2 = nn.Conv2d(24, 32, kernel_size=1, bias=True)
            self.relu5 = nn.ReLU()
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(32, 4, bias=True)

        def forward(self, x):
            x = self.relu1(self.conv1(x))
            x = self.relu2(self.dw1(x))
            x = self.relu3(self.pw1(x))
            x = self.pool1(x)
            x = self.relu4(self.dw2(x))
            x = self.relu5(self.pw2(x))
            x = self.gap(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    model = TinyDSCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)

    import copy
    best_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    print("Training Tiny DS-CNN model...")
    for epoch in range(30):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                preds = model(batch_x).argmax(dim=1)
                correct += (preds == batch_y).sum().item()
                total += len(batch_y)
        acc = (correct / total) * 100.0 if total > 0 else 0.0
        print(f"Epoch {epoch+1:02d}/30 | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {acc:.2f}%")
        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(model.state_dict())

    print(f"Training finished! Best Validation Accuracy: {best_acc:.2f}%")
    model.load_state_dict(best_weights)

    # Quantize to INT8 and export C-header
    export_c_header(model, norm_mean, norm_std)

def quantize_tensor(tensor):
    w = tensor.detach().cpu().numpy()
    max_val = np.max(np.abs(w))
    scale = max_val / 127.0 if max_val > 0 else 1.0
    q = np.clip(np.round(w / scale), -128, 127).astype(np.int8)
    return q, scale

def export_c_header(model, norm_mean, norm_std):
    header_path = os.path.join(os.path.dirname(__file__), '..', 'include', 'model_data.h')
    os.makedirs(os.path.dirname(header_path), exist_ok=True)

    c1_w, c1_s = quantize_tensor(model.conv1.weight)
    c1_b = model.conv1.bias.detach().cpu().numpy()

    dw1_w, dw1_s = quantize_tensor(model.dw1.weight)
    dw1_b = model.dw1.bias.detach().cpu().numpy()

    pw1_w, pw1_s = quantize_tensor(model.pw1.weight)
    pw1_b = model.pw1.bias.detach().cpu().numpy()

    dw2_w, dw2_s = quantize_tensor(model.dw2.weight)
    dw2_b = model.dw2.bias.detach().cpu().numpy()

    pw2_w, pw2_s = quantize_tensor(model.pw2.weight)
    pw2_b = model.pw2.bias.detach().cpu().numpy()

    fc_w, fc_s = quantize_tensor(model.fc.weight)
    fc_b = model.fc.bias.detach().cpu().numpy()

    def array_to_c_string(arr, var_name, type_name="int8_t"):
        flat = arr.flatten()
        lines = [f"static const {type_name} {var_name}[{len(flat)}] = {{"]
        for i in range(0, len(flat), 16):
            chunk = ", ".join(f"{int(x)}" if "int" in type_name else f"{x:.6f}f" for x in flat[i:i+16])
            lines.append(f"    {chunk},")
        lines.append("};\n")
        return "\n".join(lines)

    with open(header_path, "w") as f:
        f.write("// Auto-generated INT8 Quantized Model Header for Aura KWS\n")
        f.write("#ifndef MODEL_DATA_H\n#define MODEL_DATA_H\n\n#include <cstdint>\n\n")
        f.write(f"static const float NORM_MEAN = {norm_mean:.6f}f;\n")
        f.write(f"static const float NORM_STD = {norm_std:.6f}f;\n\n")

        f.write(f"static const float C1_SCALE = {c1_s:.8f}f;\n")
        f.write(array_to_c_string(c1_w, "C1_WEIGHTS", "int8_t"))
        f.write(array_to_c_string(c1_b, "C1_BIAS", "float"))

        f.write(f"static const float DW1_SCALE = {dw1_s:.8f}f;\n")
        f.write(array_to_c_string(dw1_w, "DW1_WEIGHTS", "int8_t"))
        f.write(array_to_c_string(dw1_b, "DW1_BIAS", "float"))

        f.write(f"static const float PW1_SCALE = {pw1_s:.8f}f;\n")
        f.write(array_to_c_string(pw1_w, "PW1_WEIGHTS", "int8_t"))
        f.write(array_to_c_string(pw1_b, "PW1_BIAS", "float"))

        f.write(f"static const float DW2_SCALE = {dw2_s:.8f}f;\n")
        f.write(array_to_c_string(dw2_w, "DW2_WEIGHTS", "int8_t"))
        f.write(array_to_c_string(dw2_b, "DW2_BIAS", "float"))

        f.write(f"static const float PW2_SCALE = {pw2_s:.8f}f;\n")
        f.write(array_to_c_string(pw2_w, "PW2_WEIGHTS", "int8_t"))
        f.write(array_to_c_string(pw2_b, "PW2_BIAS", "float"))

        f.write(f"static const float FC_SCALE = {fc_s:.8f}f;\n")
        f.write(array_to_c_string(fc_w, "FC_WEIGHTS", "int8_t"))
        f.write(array_to_c_string(fc_b, "FC_BIAS", "float"))

        f.write("#endif // MODEL_DATA_H\n")

    print(f"Saved INT8 model data to: {header_path}")

if __name__ == "__main__":
    train_and_export()
