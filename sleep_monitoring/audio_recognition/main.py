import os
import matplotlib.pyplot as plt
from scipy.io import wavfile
from audio_recognition.functions import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow_hub as hub

# load yamnet model
print("Loading yamnet_1 model...")
model = hub.load('../../models/yamnet_1')
print("Model loaded successfully!")

class_map_path = model.class_map_path().numpy()
class_names = class_names_from_csv(class_map_path)

n = 101
# a = signal.firwin(n, cutoff=0.3, window="hanning", pass_zero=False)

wav_file_name = '../../data/input/audio/audio_8_spec.wav'
# wav_file_new_name = '../../data/input/audio/audio_6a.wav'

# sample_rate1, wav_data1 = sf.read(wav_file_name)
# sf.write(wav_file_new_name, data, samplerate, subtype='PCM_16')
# print(sf.default_subtype('WAV'))
# exit()
print("Audio '" + wav_file_name + "' uploaded.")
sample_rate, wav_data = wavfile.read(wav_file_name, 'rb')
sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)

# Basic information about the audio
duration = len(wav_data) / sample_rate
print("Audio informations:")
print(f'Sample rate: {sample_rate} Hz')
print(f'Total duration: {duration:.2f}s')
print(f'Size of the input: {len(wav_data)}')

# Listening to the wav file.
# Audio(wav_data, rate=sample_rate)
# winsound.PlaySound(wav_file_name, winsound.SND_FILENAME)

# waveform = wav_data / tf.int16.max
waveform = wav_data[:, 0] / tf.int16.max
# Run the model, check the output.
scores, embeddings, spectrogram = model(waveform)
scores_np = scores.numpy()
spectrogram_np = spectrogram.numpy()
infered_class = class_names[scores_np.mean(axis=0).argmax()]
print(f'The main sound is: {infered_class}')

plt.figure(figsize=(10, 6))
# Plot the waveform.
plt.subplot(3, 1, 1)
plt.plot(waveform)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.xlim([0, len(waveform)])
# Plot the log-mel spectrogram (returned by the model).
plt.subplot(3, 1, 2)
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.imshow(spectrogram_np.T, aspect='auto', interpolation='nearest', origin='lower')
# Plot and label the model output scores for the top-scoring classes.
mean_scores = np.mean(scores, axis=0)
top_n = 5
top_class_indices = np.argsort(mean_scores)[::-1][:top_n]
plt.subplot(3, 1, 3)
plt.imshow(scores_np[:, top_class_indices].T, aspect='auto', interpolation='nearest', cmap='gray_r')
# patch_padding = (PATCH_WINDOW_SECONDS / 2) / PATCH_HOP_SECONDS
# values from the model documentation
patch_padding = (0.025 / 2) / 0.01
plt.xlim([-patch_padding - 0.5, scores.shape[0] + patch_padding - 0.5])
# Label the top_N classes.
yticks = range(0, top_n, 1)
plt.yticks(yticks, [class_names[top_class_indices[x]] for x in yticks])
_ = plt.ylim(-0.5 + np.array([top_n, 0]))

plt.show()
