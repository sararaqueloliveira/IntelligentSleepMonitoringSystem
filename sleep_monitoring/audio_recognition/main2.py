import os
import matplotlib.pyplot as plt
from scipy.io import wavfile
from audio_recognition.functions import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow_hub as hub

# load yamnet model
saved_model_path = '../../models/yamnet_retrained'
reloaded_model = tf.saved_model.load(saved_model_path)

my_classes = ['snoring', 'breathing']

wav_file_name = '../../data/input/audio/audio_30.wav'
#sample_rate, wav_data = wavfile.read(wav_file_name, 'rb')
#sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)
waveform = load_wav_16k_mono(wav_file_name)

scores = reloaded_model(waveform)
your_top_class = tf.argmax(scores)
your_infered_class = my_classes[your_top_class]
class_probabilities = tf.nn.softmax(scores, axis=-1)
your_top_score = class_probabilities[your_top_class]
print(f'[Your model] The main sound is: {your_infered_class} ({your_top_score})')

probabilities = class_probabilities.numpy()

# Figure Size

plt.barh(my_classes, probabilities, color=['black'], height=0.2)
plt.title('Audio classification score')
plt.ylabel('Classes')
plt.xlabel('Score')
plt.show()

