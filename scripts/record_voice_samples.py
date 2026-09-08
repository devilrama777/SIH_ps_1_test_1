"""
Interactive Voice Recorder for Aura Keyword Spotting
Allows users to record custom voice samples for 'Aura', 'Exit', or custom background noise.
Audio files are automatically saved to the data/ directory for model training.
"""
import os
import sys
import time
import wave
import numpy as np
import miniaudio

SAMPLE_RATE = 16000
DURATION_SEC = 1.0
BUFFER_SAMPLES = int(SAMPLE_RATE * DURATION_SEC)

OUTPUT_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

def record_clip(duration_sec=DURATION_SEC) -> np.ndarray:
    recorded_frames = []

    def capture_callback(device, p_input, frame_count):
        raw_bytes = bytes(p_input)
        recorded_frames.append(raw_bytes)

    device = miniaudio.CaptureDevice(
        sample_rate=SAMPLE_RATE,
        nchannels=1,
        format=miniaudio.SampleFormat.SIGNED16,
        buffersize_msec=50
    )
    device.start(capture_callback)
    time.sleep(duration_sec)
    device.stop()
    device.close()

    full_bytes = b"".join(recorded_frames)
    audio_i16 = np.frombuffer(full_bytes, dtype=np.int16)
    if len(audio_i16) < BUFFER_SAMPLES:
        audio_i16 = np.pad(audio_i16, (0, BUFFER_SAMPLES - len(audio_i16)))
    else:
        audio_i16 = audio_i16[:BUFFER_SAMPLES]
    return audio_i16

def save_wav(audio_data: np.ndarray, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with wave.open(filepath, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data.tobytes())

def record_keyword_interactive(keyword: str, count: int = 5):
    target_dir = os.path.join(OUTPUT_BASE, keyword)
    os.makedirs(target_dir, exist_ok=True)
    
    existing = len([f for f in os.listdir(target_dir) if f.endswith('.wav')])
    print(f"\n=======================================================")
    print(f" Recording {count} samples for keyword: '{keyword.upper()}'")
    print(f"=======================================================")
    print(f"When prompted, speak '{keyword}' clearly into your microphone.\n")

    for i in range(count):
        idx = existing + i + 1
        input(f"[{i+1}/{count}] Press [ENTER] when ready to say '{keyword}'...")
        print("  -> RECORDING NOW (1 second)... speak!", flush=True)
        clip = record_clip(1.0)
        filename = f"user_{keyword}_{idx}.wav"
        save_path = os.path.join(target_dir, filename)
        save_wav(clip, save_path)
        print(f"  -> Saved: {save_path}\n")

    print(f"Done recording '{keyword}'! Total samples in data/{keyword}/: {existing + count}")

def main():
    print("==================================================")
    print(" Aura Keyword Spotting - Custom Voice Recorder")
    print("==================================================")
    print("Choose an option:")
    print(" 1) Record 'Aura' (Wake Word)")
    print(" 2) Record 'Exit' (Sleep Word)")
    print(" 3) Record Both 'Aura' and 'Exit' (5 samples each)")
    print(" 4) Record Custom Unknown words / Command phrases")
    print(" 5) Exit")
    
    choice = input("\nEnter choice [1-5]: ").strip()
    if choice == '1':
        record_keyword_interactive('aura', 5)
    elif choice == '2':
        record_keyword_interactive('exit', 5)
    elif choice == '3':
        record_keyword_interactive('aura', 5)
        record_keyword_interactive('exit', 5)
    elif choice == '4':
        phrase = input("Enter keyword/phrase name for label: ").strip().lower()
        record_keyword_interactive('unknown', 5)
    else:
        print("Exiting.")
        return

    retrain = input("\nWould you like to re-train the model now with these new samples? (y/n): ").strip().lower()
    if retrain == 'y':
        print("\nStarting model training...")
        import subprocess
        train_script = os.path.join(os.path.dirname(__file__), 'train_kws_model.py')
        subprocess.run([sys.executable, train_script], check=True)

if __name__ == "__main__":
    main()
