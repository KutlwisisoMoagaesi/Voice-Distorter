import numpy as np
import sounddevice as sd
import time
from scipy import signal

print("Started!")

class VoiceDistorter:
    def __init__(self, chunk_size=512, sample_rate=44100):
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate

        self.pitch_shift = 0.05
        self.bit_crush_bits = 10
        self.ring_mod_freq = 15
        self.distortion_gain = 1.5
        self.noise_threshold = 0.05

        sd.default.samplerate = self.sample_rate
        sd.default.channels = 1
        sd.default.blocksize = self.chunk_size

    def noise_gate(self, audio, threshold):
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < threshold:
            return np.zeros_like(audio)
        return audio

    def light_bit_crush(self, audio, bits):
        max_val = 2 ** (bits - 10)
        crushed = np.round(audio * max_val) / max_val
        return 3.5 * crushed + 3.5 * audio

    def subtle_ring_modulation(self, audio, freq):
        t = np.arange(len(audio)) / self.sample_rate
        mod = 0.8 + 0.2 * np.sin(2 * np.pi * freq * t)
        return audio * mod

    def apply_distortion(self, audio):
        audio = self.noise_gate(audio, self.noise_threshold)

        audio = np.tanh(audio * self.distortion_gain)

        audio = self.light_bit_crush(audio, self.bit_crush_bits)

        audio = self.subtle_ring_modulation(audio, self.ring_mod_freq)

        sos_high = signal.butter(5, 200, 'highpass', fs=self.sample_rate, output='sos')
        sos_low = signal.butter(5, 4000, 'lowpass', fs=self.sample_rate, output='sos')

        audio = signal.sosfilt(sos_high, audio)
        audio = signal.sosfilt(sos_low, audio)

        return np.clip(audio, -5.0, 5.0)

    def audio_callback(self, indata, outdata, frames, time_info, status):
        audio = indata[:, 0]
        processed = self.apply_distortion(audio)
        outdata[:] = processed.reshape(-1, 1)

    def start(self):
        with sd.Stream(callback=self.audio_callback):
            print("Voice distorter running... Press Ctrl+C to stop.")
            while True:
                time.sleep(0.1)


if __name__ == "__main__":
    VoiceDistorter().start()
