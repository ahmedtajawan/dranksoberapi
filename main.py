import streamlit as st
import sounddevice as sd
import numpy as np
import io
import wavio

st.title('🎙️ Streamlit In-Browser Audio Recorder')

# Audio recording settings
fs = 44100  # Sample rate
seconds = st.slider('Select recording duration (seconds)', 1, 60, 5)  # Recording duration

# Initialize session state to store audio data
if 'audio_buffer' not in st.session_state:
    st.session_state['audio_buffer'] = None

# Record button
if st.button('Record'):
    st.write('Recording...')
    recording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished

    # Save recording to a buffer instead of a file
    audio_buffer = io.BytesIO()
    wavio.write(audio_buffer, recording, fs, sampwidth=2)
    audio_buffer.seek(0)

    # Store in session state
    st.session_state['audio_buffer'] = audio_buffer

    st.write('Recording complete!')

# Playback the recording if available
if st.session_state['audio_buffer'] is not None:
    st.audio(st.session_state['audio_buffer'].getvalue(), format='audio/wav')
