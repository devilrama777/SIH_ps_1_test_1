import os
import asyncio
import wave
import numpy as np
import miniaudio
import edge_tts

SAMPLE_RATE = 16000
CLIP_DURATION = 1.0
NUM_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION)

INDIAN_VOICES = [
    'en-IN-PrabhatNeural',          # Indian English Male
    'en-IN-NeerjaNeural',           # Indian English Female
    'en-IN-NeerjaExpressiveNeural', # Indian English Female Expressive
    'hi-IN-MadhurNeural',           # Indian Hindi-accent Male
    'hi-IN-SwaraNeural'             # Indian Hindi-accent Female
]

OUTPUT_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

def save_wav(samples: np.ndarray, filepath: str):
    pcm16 = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(filepath, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm16.tobytes())

def pad_or_trim(samples: np.ndarray, target_len: int) -> np.ndarray:
    n = len(samples)
    if n == target_len:
        return samples
    if n > target_len:
        start = np.random.randint(0, n - target_len)
        return samples[start:start + target_len]
    else:
        out = np.zeros(target_len, dtype=np.float32)
        start = np.random.randint(0, target_len - n)
        out[start:start + n] = samples
        return out

def create_augmentations(base_samples: np.ndarray, num_variations: int = 25) -> list:
    augmented_list = []
    for _ in range(num_variations):
        # 1. Random time stretching / speed resample (0.85x to 1.15x)
        speed = np.random.uniform(0.85, 1.15)
        old_indices = np.arange(len(base_samples))
        new_length = int(len(base_samples) / speed)
        new_indices = np.linspace(0, len(base_samples) - 1, new_length)
        resampled = np.interp(new_indices, old_indices, base_samples).astype(np.float32)

        # 2. Pad or trim with random jitter
        padded = pad_or_trim(resampled, NUM_SAMPLES)

        # 3. Random gain / volume (0.6x to 1.4x)
        gain = np.random.uniform(0.6, 1.4)
        out = padded * gain

        # 4. Realistic acoustic noise (SNR 15dB to 35dB)
        snr_db = np.random.uniform(15.0, 35.0)
        sig_power = np.mean(out ** 2) + 1e-9
        noise_power = sig_power / (10.0 ** (snr_db / 10.0))
        noise = np.random.normal(0, np.sqrt(noise_power), len(out))

        # 5. 50Hz hum harmonic occasionally
        if np.random.rand() > 0.5:
            t = np.linspace(0, CLIP_DURATION, len(out), endpoint=False)
            hum = np.random.uniform(0.001, 0.004) * np.sin(2 * np.pi * 50 * t)
            noise += hum

        out = out + noise
        max_val = np.max(np.abs(out)) + 1e-9
        if max_val > 1.0:
            out = out / max_val
        augmented_list.append(out.astype(np.float32))
    return augmented_list

async def fetch_one_clean(text: str, voice: str, rate: str = "+0%") -> np.ndarray:
    try:
        communicate = edge_tts.Communicate(text, voice, rate=rate)
        audio_data = bytearray()
        async for chunk in communicate.stream():
            if chunk['type'] == 'audio':
                audio_data.extend(chunk['data'])
        if len(audio_data) == 0:
            return np.zeros(0, dtype=np.float32)
        decoded = miniaudio.decode(bytes(audio_data), nchannels=1, sample_rate=SAMPLE_RATE, output_format=miniaudio.SampleFormat.FLOAT32)
        return np.array(decoded.samples, dtype=np.float32)
    except Exception as e:
        print(f"Fetch failed for '{text}' with {voice}: {e}", flush=True)
        return np.zeros(0, dtype=np.float32)

async def main():
    for label in ['silence', 'unknown', 'aura', 'exit']:
        d = os.path.join(OUTPUT_BASE, label)
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(d):
            if f.endswith('.wav'):
                try: os.remove(os.path.join(d, f))
                except: pass

    print("=== Generating Indian Audio Dataset (Fast Neural Batch) ===", flush=True)

    # 1. Base 'Aura' downloads across Indian voices & rates
    print("1. Synthesizing base Indian 'Aura' samples...", flush=True)
    base_auras = []
    for voice in INDIAN_VOICES:
        for rate in ["-10%", "+0%", "+10%"]:
            samples = await fetch_one_clean("Aura", voice, rate)
            if len(samples) > 1000:
                base_auras.append(samples)
            # Also alternative pronunciation "Oura" / "Hey Aura"
            samples2 = await fetch_one_clean("Hey Aura", voice, rate)
            if len(samples2) > 1000:
                base_auras.append(samples2)

    print(f"Captured {len(base_auras)} pristine Indian 'Aura' base recordings. Augmenting...", flush=True)
    count_aura = 0
    for base in base_auras:
        augs = create_augmentations(base, num_variations=15)
        for aug in augs:
            save_wav(aug, os.path.join(OUTPUT_BASE, 'aura', f"in_aura_{count_aura}.wav"))
            count_aura += 1
    print(f"Total 'Aura' dataset: {count_aura} Indian samples.", flush=True)

    # 2. Base 'Exit' downloads across Indian voices & rates (PURE 'Exit' only)
    print("2. Synthesizing base Indian 'Exit' samples...", flush=True)
    base_exits = []
    for voice in INDIAN_VOICES:
        for rate in ["-10%", "+0%", "+10%"]:
            samples = await fetch_one_clean("Exit", voice, rate)
            if len(samples) > 1000:
                base_exits.append(samples)
            samples2 = await fetch_one_clean("Exit.", voice, rate)
            if len(samples2) > 1000:
                base_exits.append(samples2)

    print(f"Captured {len(base_exits)} pristine Indian 'Exit' base recordings. Augmenting...", flush=True)
    count_exit = 0
    for base in base_exits:
        augs = create_augmentations(base, num_variations=15)
        for aug in augs:
            save_wav(aug, os.path.join(OUTPUT_BASE, 'exit', f"in_exit_{count_exit}.wav"))
            count_exit += 1
    print(f"Total 'Exit' dataset: {count_exit} Indian samples.", flush=True)

    # 3. Base 'Unknown' words/phrases across Indian voices
    # Include words that sound phonetically close to 'exit' so they NEVER trigger false exits!
    print("3. Synthesizing base Indian 'Unknown' phrases...", flush=True)
    unknown_texts = [
        # Phonetic lookalikes to Exit (contrastive negatives):
        "exact", "extra", "except", "exist", "exercise", "access", "accept",
        "next", "text", "six", "check", "expert", "packet", "select",
        # Common command phrases & words:
        "what is the time", "what is today date", "tell me time", "time kya hai",
        "what is the temperature", "what is oxygen level", "check cabin pressure",
        "how are you", "can you hear me", "who are you", "what are you doing",
        "turn on the light", "turn off display", "open telemetry", "start engine",
        "one", "two", "three", "four", "five", "seven", "eight", "nine", "zero",
        "namaste", "shukriya", "hello", "theek hai", "yes", "no", "okay",
        "satellite", "isro", "chandrayaan", "gaganyaan", "rocket", "orbit", "speed",
        "sleep", "so jao", "band karo", "stop", "terminate", "pause", "resume",
        "water level", "cabin temp", "battery status", "altitude reading", "speed of rocket"
    ]
    base_unknowns = []
    for text in unknown_texts:
        # Use Prabhat (Male) and Neerja (Female)
        v = 'en-IN-PrabhatNeural' if np.random.rand() > 0.5 else 'en-IN-NeerjaNeural'
        samples = await fetch_one_clean(text, v, "+0%")
        if len(samples) > 1000:
            base_unknowns.append(samples)

    print(f"Captured {len(base_unknowns)} Indian 'Unknown' base recordings. Augmenting...", flush=True)
    count_unk = 0
    for base in base_unknowns:
        augs = create_augmentations(base, num_variations=20)
        for aug in augs:
            save_wav(aug, os.path.join(OUTPUT_BASE, 'unknown', f"in_unk_{count_unk}.wav"))
            count_unk += 1
    print(f"Total 'Unknown' dataset: {count_unk} Indian samples.", flush=True)

    # 4. Silence & Room Noise
    print("4. Generating 'Silence' & room noise...", flush=True)
    for i in range(400):
        scale = np.random.uniform(0.0003, 0.012)
        t = np.linspace(0, CLIP_DURATION, NUM_SAMPLES, endpoint=False)
        if i % 3 == 0:
            hum = (scale * 0.4) * np.sin(2 * np.pi * 50 * t) + np.random.normal(0, scale * 0.3, NUM_SAMPLES)
        elif i % 3 == 1:
            hum = (scale * 0.4) * np.sin(2 * np.pi * 120 * t) + np.random.normal(0, scale * 0.3, NUM_SAMPLES)
        else:
            hum = np.random.normal(0, scale * 0.5, NUM_SAMPLES)
        save_wav(hum.astype(np.float32), os.path.join(OUTPUT_BASE, 'silence', f"silence_{i}.wav"))
    print("Total 'Silence' dataset: 400 samples.", flush=True)

    print("=== Complete Indian Audio Dataset Ready! ===", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
