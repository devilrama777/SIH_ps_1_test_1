"""
High-Quality Synthetic Dataset Generator for 'Aura' KWS Model
Classes:
0: silence (ambient quiet room noise at real mic levels)
1: unknown (large, diverse vocabulary of conversational phrases, questions, words, numbers)
2: aura (wake word)
3: exit (termination words: "exit", "sleep")
"""
import os
import wave
import numpy as np
import comtypes.client

SAMPLE_RATE = 16000
CLIP_DURATION = 1.0
NUM_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION)

def synthesize_to_wav(voice_engine, text, wav_path, rate=0, volume=100, voice_item=None):
    if voice_item:
        voice_engine.Voice = voice_item
    voice_engine.Rate = rate
    voice_engine.Volume = volume

    stream = comtypes.client.CreateObject('SAPI.SpFileStream')
    stream.Open(wav_path, 3, False)
    voice_engine.AudioOutputStream = stream
    voice_engine.Speak(text)
    stream.Close()

def load_wav_resample_normalize(wav_path, target_sr=16000):
    with wave.open(wav_path, 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_data = wf.readframes(n_frames)

    if sampwidth == 2:
        audio = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 1:
        audio = (np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        audio = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0

    if n_channels > 1:
        audio = audio[::n_channels]

    if framerate != target_sr and len(audio) > 0:
        indices = np.round(np.arange(0, len(audio), framerate / target_sr)).astype(int)
        indices = indices[indices < len(audio)]
        audio = audio[indices]

    return audio

def pad_or_trim(audio, target_length=NUM_SAMPLES, shift_jitter=True):
    if len(audio) == 0:
        return np.zeros(target_length, dtype=np.float32)
    
    if len(audio) > target_length:
        start = (len(audio) - target_length) // 2
        return audio[start:start + target_length]
    
    diff = target_length - len(audio)
    if shift_jitter:
        pad_left = np.random.randint(0, diff + 1)
    else:
        pad_left = diff // 2
    pad_right = diff - pad_left
    return np.pad(audio, (pad_left, pad_right), mode='constant')

def add_noise_and_augment(audio, snr_db_range=(12, 30)):
    # Random realistic volume scale matching mic levels
    gain = np.random.uniform(0.3, 0.9)
    audio = audio * gain

    snr_db = np.random.uniform(*snr_db_range)
    signal_power = np.mean(audio ** 2) + 1e-12
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), size=len(audio))
    
    augmented = audio + noise
    max_val = np.max(np.abs(augmented))
    if max_val > 1.0:
        augmented = augmented / max_val
    return augmented.astype(np.float32)

def save_wav(audio_data, file_path, sr=16000):
    audio_int16 = np.clip(audio_data * 32767.0, -32768.0, 32767.0).astype(np.int16)
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_int16.tobytes())

def generate_dataset():
    output_base = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(output_base, exist_ok=True)
    temp_dir = os.path.join(output_base, 'temp')
    os.makedirs(temp_dir, exist_ok=True)

    classes = ['silence', 'unknown', 'aura', 'exit']
    for c in classes:
        cls_dir = os.path.join(output_base, c)
        os.makedirs(cls_dir, exist_ok=True)
        # Clear old files
        for old_f in os.listdir(cls_dir):
            try:
                os.remove(os.path.join(cls_dir, old_f))
            except:
                pass

    voice_engine = comtypes.client.CreateObject('SAPI.SpVoice')
    all_voices = voice_engine.GetVoices()
    selected_voices = [all_voices.Item(i) for i in range(min(2, all_voices.Count))]

    print(f"Using {len(selected_voices)} English SAPI voice(s).")

    # 1. Aura (Wake word only)
    print("Synthesizing 'Aura' samples...")
    aura_words = ["Aura", "aura", "Hey Aura", "Aura.", "Aura!"]
    count_aura = 0
    for v_item in selected_voices:
        for rate in [-3, -2, -1, 0, 1, 2, 3]:
            for vol in [75, 85, 100]:
                for w in aura_words:
                    temp_wav = os.path.join(temp_dir, f"aura_tmp_{count_aura}.wav")
                    synthesize_to_wav(voice_engine, w, temp_wav, rate=rate, volume=vol, voice_item=v_item)
                    raw = load_wav_resample_normalize(temp_wav)
                    for aug_i in range(2):
                        padded = pad_or_trim(raw, NUM_SAMPLES, shift_jitter=True)
                        augmented = add_noise_and_augment(padded, snr_db_range=(15, 30))
                        dst = os.path.join(output_base, 'aura', f"aura_{count_aura}_{aug_i}.wav")
                        save_wav(augmented, dst)
                    count_aura += 1
    print(f"Generated {count_aura * 2} 'Aura' samples.")

    # 2. Exit (ONLY keyword for stopping/sleeping)
    print("Synthesizing 'Exit' samples...")
    exit_words = ["Exit", "exit", "Exit.", "Exit!", "Aura exit"]
    count_exit = 0
    for v_item in selected_voices:
        for rate in [-3, -2, -1, 0, 1, 2, 3]:
            for vol in [75, 85, 100]:
                for w in exit_words:
                    temp_wav = os.path.join(temp_dir, f"exit_tmp_{count_exit}.wav")
                    synthesize_to_wav(voice_engine, w, temp_wav, rate=rate, volume=vol, voice_item=v_item)
                    raw = load_wav_resample_normalize(temp_wav)
                    for aug_i in range(2):
                        padded = pad_or_trim(raw, NUM_SAMPLES, shift_jitter=True)
                        augmented = add_noise_and_augment(padded, snr_db_range=(15, 30))
                        dst = os.path.join(output_base, 'exit', f"exit_{count_exit}_{aug_i}.wav")
                        save_wav(augmented, dst)
                    count_exit += 1
    print(f"Generated {count_exit * 2} 'Exit' samples.")

    # 3. Unknown (Large, varied vocabulary: commands, questions, sleep, terminate, everyday words)
    print("Synthesizing diverse 'Unknown' conversational samples...")
    unknown_phrases = [
        # Questions users ask as commands:
        "what is the time", "what is today's date", "how are you", "can you hear me", 
        "what is the temperature", "what is the oxygen level", "what is your name",
        "tell me a joke", "who are you", "what are you doing", "what is the mission status",
        "where are we going", "how much fuel is left", "what is the pressure reading",
        "tell me the speed", "is the system ready", "check the connection", "turn on the light",
        "turn off the light", "show me telemetry", "open navigation", "calculate trajectory",
        # Sleep and terminate (now treated as standard speech):
        "sleep", "go to sleep", "shut down", "stop", "terminate", "termi", "ter",
        # Common single words:
        "hello", "computer", "system", "satellite", "rocket", "orbit", "space", 
        "station", "sensor", "telemetry", "oxygen", "pressure", "thruster", "speed", 
        "navigate", "camera", "power", "battery", "light", "display", "status",
        "confirm", "abort", "standby", "engage", "frequency", "antenna", "signal",
        "radio", "yes", "no", "okay", "start", "next", "previous", "cancel",
        # Numbers:
        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "zero",
        # Conversational words:
        "please", "thank you", "morning", "night", "test", "testing", "microphone",
        "audio", "sound", "volume", "music", "play", "pause", "resume", "forward"
    ]
    count_unknown = 0
    for v_item in selected_voices:
        for rate in [-2, 0, 2]:
            for phrase in unknown_phrases:
                temp_wav = os.path.join(temp_dir, f"unknown_tmp_{count_unknown}.wav")
                synthesize_to_wav(voice_engine, phrase, temp_wav, rate=rate, volume=90, voice_item=v_item)
                raw = load_wav_resample_normalize(temp_wav)
                for aug_i in range(2):
                    padded = pad_or_trim(raw, NUM_SAMPLES, shift_jitter=True)
                    augmented = add_noise_and_augment(padded, snr_db_range=(12, 28))
                    dst = os.path.join(output_base, 'unknown', f"unknown_{count_unknown}_{aug_i}.wav")
                    save_wav(augmented, dst)
                count_unknown += 1
    print(f"Generated {count_unknown * 2} 'Unknown' samples.")

    # 4. Silence (Diverse hums, fans, room acoustics, electric noise)
    print("Synthesizing 'Silence' & background hum samples...")
    for i in range(400):
        scale = np.random.uniform(0.0005, 0.02)
        noise_type = i % 4
        if noise_type == 0:
            # Pure white / gaussian noise
            noise = np.random.normal(0, scale, NUM_SAMPLES)
        elif noise_type == 1:
            # 50Hz / 60Hz powerline hum + harmonic
            t = np.linspace(0, CLIP_DURATION, NUM_SAMPLES, endpoint=False)
            f0 = 50.0 if (i % 2 == 0) else 60.0
            noise = (scale * 0.4) * np.sin(2 * np.pi * f0 * t) + (scale * 0.2) * np.sin(2 * np.pi * 2 * f0 * t) + np.random.normal(0, scale * 0.4, NUM_SAMPLES)
        elif noise_type == 2:
            # Low frequency acoustic rumble / fan hum (100Hz - 200Hz)
            t = np.linspace(0, CLIP_DURATION, NUM_SAMPLES, endpoint=False)
            f_fan = np.random.uniform(80, 180)
            noise = (scale * 0.5) * np.sin(2 * np.pi * f_fan * t) + np.random.normal(0, scale * 0.3, NUM_SAMPLES)
        else:
            # Ultra quiet room silence
            noise = np.random.normal(0, 0.0003, NUM_SAMPLES)

        dst = os.path.join(output_base, 'silence', f"silence_{i}.wav")
        save_wav(noise.astype(np.float32), dst)
    print(f"Generated 400 'Silence' samples.")

    # Clean up temp
    for f in os.listdir(temp_dir):
        try: os.remove(os.path.join(temp_dir, f))
        except: pass
    try: os.rmdir(temp_dir)
    except: pass

    print("=== Dataset Generation Complete! ===")

if __name__ == "__main__":
    generate_dataset()
