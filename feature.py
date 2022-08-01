import numpy as np
import scipy
from scipy.io import wavfile
from lib import  hann, fft, fft_v, dct, freq_to_mel, mel_to_freq

def normalize_audio(audio):
    audio = audio / np.max(np.abs(audio))
    return audio

def frame_audio(audio, Frame_size=1024, hop_len=400, sample_rate=44100):
    
    audio = np.pad(audio, int(Frame_size / 2), mode='reflect')
    frame_num = int((len(audio) - Frame_size) / hop_len) + 1
    frames = np.zeros((frame_num,Frame_size))
    
    for n in range(frame_num):
        frames[n] = audio[n*hop_len:n*hop_len+Frame_size]
    
    return frames

def time_to_freq(audio_win, Frame_size=1024):

    audio_winT = np.transpose(audio_win)
    audio_fft = np.empty((int(1 + Frame_size // 2), audio_winT.shape[1]), dtype=np.complex64, order='F') # Empty fft matrix for compex numbers

    for n in range(audio_fft.shape[1]):
        audio_fft[:, n] = fft_v(audio_winT[:, n])[:audio_fft.shape[0]] # Filling the FFT matrix for the first audio_fft.shape[0] elements

    return np.transpose(audio_fft) # Return frequency domain matrix

def get_filter_points(fmin, fmax, filter_num, Frame_size, sample_rate=44100):
    
    fmin_mel = freq_to_mel(fmin)
    fmax_mel = freq_to_mel(fmax) # Change frequency domain to mel domain
    
    mels = np.linspace(fmin_mel, fmax_mel, num=filter_num+2) # Dividing from 0 to a maximum of (filter_num+2) parts
    freqs = mel_to_freq(mels) # Change mel domain to frequency domain
    
    return np.floor((Frame_size + 1) / sample_rate * freqs).astype(int), freqs # Return filter points and frequencies of that points 

def get_filters(filter_points, Frame_size, mel_freqs, filter_num):
    
    filters = np.zeros((len(filter_points)-2,int(Frame_size/2+1))) # Make empty filters matrix
    
    for n in range(len(filter_points)-2): # Filling filters matrix
        filters[n, filter_points[n] : filter_points[n + 1]] = np.linspace(0, 1, filter_points[n + 1] - filter_points[n]) # First filter part: creates points from 0 to 1 (filter_points[n + 1] - filter_points[n]) times. direction up /
        filters[n, filter_points[n + 1] : filter_points[n + 2]] = np.linspace(1, 0, filter_points[n + 2] - filter_points[n + 1]) # First filter part: creates points from 0 to 1 (filter_points[n + 2] - filter_points[n + 1]) times. direction down \
    
    enorm = 2.0 / (mel_freqs[2:filter_num+2] - mel_freqs[:filter_num]) # Normalization of filters
    filters *= enorm[:, np.newaxis] # If we wont normalize the filters, we will see the noise increase with frequency because of the filter width.

    return filters # Filters /\/\/\...

def mfcc(y, sr=44100, n_fft=1024, hop_length=400, n_mfcc=20, n_filters = 40):
    
    norm_audio = normalize_audio(y) # Normalize the audio data
    frames = frame_audio(norm_audio, Frame_size=n_fft) # Make frames
    
    window = hann(n_fft) # Make Hann windowing function
    audio_win = frames * window # windowing frames

    audio_fft = time_to_freq(audio_win=audio_win, Frame_size=n_fft) # Change time domain to frequecy domain
    
    audio_power = np.square(np.abs(audio_fft)) # Power of audio data

    freq_min = 0
    freq_max = sr / 2 # Minimum and maximum frequencies for filter points
    filter_points, mel_freqs = get_filter_points(fmin = freq_min, fmax = freq_max, filter_num=n_filters, Frame_size=n_fft, sample_rate=sr) # Points of filters and them frequencies
    filters = get_filters(filter_points=filter_points, Frame_size=n_fft, mel_freqs=mel_freqs, filter_num=n_filters) # Get filters matix
    
    audio_filtered = np.dot(filters, np.transpose(audio_power)) # filtering audio power
    audio_log = 10.0 * np.log10(audio_filtered) # Final Audio power matrix
    
    dct_filters = dct(n_mfcc, n_filters) # Making Discrete Cosine Transform Filters
    cepstral_coefficents = np.dot(dct_filters, audio_log) # Filtering Audio power and get Cepstral Coeficents...

    return cepstral_coefficents, audio_log #return Cepstral coeficents and audio power for spectogram






    





