# Ideal, Natural, & Flat-top -Sampling
# Aim
Write a simple Python program for the construction and reconstruction of ideal, natural, and flattop sampling.
# Tools required
# Program
IDEAL SAMPLING:
import numpy as np;
import matplotlib.pyplot as plt;
from scipy.signal import resample;
fs=100;
t=np.arange(0,1,1/fs);
f=5;
signal=np.sin(2*np.pi*f*t);
t_sampled=np.arange(0,1,1/fs);
signal_sampled=np.sin(2*np.pi*f*t_sampled);
reconstructed_signal=resample(signal_sampled,len(t));
plt.figure(figsize=(12,8));
plt.subplot(3,1,1);
plt.plot(t,signal,label='Continuous Signal');
plt.title('Continuous Signal (fs=100Hz)');
plt.xlabel('Time[s]');
plt.ylabel('Amplitude');
plt.grid(True);
plt.legend();
plt.subplot(3,1,2);
plt.stem(t_sampled,signal_sampled,linefmt='r-',markerfmt='ro',basefmt='r-',label='Sampled Signal');
plt.title('Sampled Signal (fs=100Hz)');


NATURAL SAMPLING:
import numpy as np;
import matplotlib.pyplot as plt;
from scipy.signal import butter,lfilter;
fs=1000;
T=1;
t=np.arange(0,T,1/fs);
fm=5;
message_signal=np.sin(2*np.pi*fm*t);
pulse_rate=50;
pulse_train=np.zeros_like(t);
pulse_width=int(fs/pulse_rate/2);
for i in range(0,len(t),int(fs/pulse_rate)):
    pulse_train[i:i+pulse_width]=1;
nat_signal=message_signal*pulse_train;
sampled_signal=nat_signal[pulse_train==1];
sample_times=t[pulse_train==1];
reconstructed_signal=np.zeros_like(t);
for i,time in enumerate(sample_times):
    idx=np.argmin(np.abs(t-time));
    reconstructed_signal[idx:idx+pulse_width]=sampled_signal[i];
def lowpass_filter(sig,cutoff,fs,order=5):
    nyq=0.5*fs;
    wn=cutoff/nyq;
    b,a=butter(order,wn,btype='low');
    return lfilter(b,a,sig);

reconstructed_signal=lowpass_filter(reconstructed_signal,10,fs);
plt.figure(figsize=(14,10));
plt.subplot(4,1,1);
plt.plot(t,message_signal);
plt.title("Original Message Signal");
plt.grid(True);
plt.subplot(4,1,2);
plt.plot(t,pulse_train);
plt.title("Pulse Train");
plt.grid(True);
plt.subplot(4,1,3);
plt.plot(t,nat_signal);
plt.title("Natural Sampled Signal");
plt.grid(True);
plt.subplot(4,1,4);
plt.plot(t,reconstructed_signal);
plt.title("Reconstructed Signal");
plt.grid(True);
plt.tight_layout();
plt.show();

FLAT TOP SAMPLING:
import numpy as np;
import matplotlib.pyplot as plt;
from scipy.signal import butter,lfilter;
fs=1000;
T=1;
t=np.arange(0,T,1/fs);
fm=5;
message_signal=np.sin(2*np.pi*fm*t);
pulse_rate=50;
pulse_train_indices=np.arange(0,len(t),int(fs/pulse_rate));
pulse_train=np.zeros_like(t);
pulse_train[pulse_train_indices]=1;
flat_top_signal=np.zeros_like(t);
pulse_width_samples=int(fs/(2*pulse_rate));
for idx in pulse_train_indices:
    val=message_signal[idx];
    end_idx=min(idx+pulse_width_samples,len(t));
    flat_top_signal[idx:end_idx]=val;
def lowpass_filter(sig,cutoff,fs,order=5):
    nyq=0.5*fs;
    wn=cutoff/nyq;
    b,a=butter(order,wn,btype='low');
    return lfilter(b,a,sig);
cutoff_freq=2*fm;
reconstructed_signal=lowpass_filter(flat_top_signal,cutoff_freq,fs);
plt.figure(figsize=(14,10));
plt.subplot(4,1,1);
plt.plot(t,message_signal);
plt.title("Original Message Signal");
plt.grid(True);
plt.subplot(4,1,2);
plt.stem(t[pulse_train_indices],pulse_train[pulse_train_indices],basefmt=" ");
plt.title("Ideal Sampling Instances");
plt.grid(True);
plt.subplot(4,1,3);
plt.plot(t,flat_top_signal);
plt.title("Flat-Top Sampled Signal");
plt.grid(True);
plt.subplot(4,1,4);
plt.plot(t,reconstructed_signal);
plt.title("Reconstructed Signal");
plt.grid(True);
plt.tight_layout();
plt.show();


# Output Waveform
IDEALSAMPLING
<img width="1189" height="790" alt="545514946-50a83352-0467-4d45-97fb-3bc5218ae6f1" src="https://github.com/user-attachments/assets/b4e49303-6e37-456c-bba9-d79dd94c0c8b" />
NATURAL SAMPLING
<img width="1390" height="989" alt="545515483-18de145b-0877-4b6b-bf47-9fa50162254c" src="https://github.com/user-attachments/assets/51222288-a8c1-46c2-80d3-e04717e3f177" />
FLAT TOP SAMPLING
<img width="1398" height="990" alt="545515968-6984b98c-4483-422a-9446-a06f3c179a8c" src="https://github.com/user-attachments/assets/a073b319-b1fe-4034-a795-9a1ca42e470f" />
# Results 
Thus, the construction and reconstruction of Ideal, Natural, and Flat-top sampling were successfully implemented using Python, and the corresponding waveforms were obtained.


