import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz 
import sounddevice as sd
import soundfile as sf
from numba import jit


@jit(nopython=True, nogil=True)
def compute_anf(norm, alpha, r, mu, length, x, y, u, u_buf):
    beta = 0.05 # 時定数
    for k in range(length):
        # フィルタリング
        u[k] = x[k] - alpha * u_buf[1] - r * u_buf[0]
        y[k] = r * u[k] + alpha * u_buf[1] + u_buf[0]
        y[k] = 0.5 * (y[k] + x[k])  
        
        # 更新
        # norm = E[u^2(n-1)] は積分回路を利用
        norm = beta * norm + (1-beta) * u_buf[1] * u_buf[1]
        if norm > 1e-12: # normがある大きさ以上の場合、更新
            alpha -= mu * y[k] * u_buf[1] / norm

        # alphaの値域を制限
        if  alpha < -(1+r):
            alpha = -(1+r)+1e-12
        elif alpha > 1+r:
            alpha = 1+r-1e-12
            
        u_buf[0] = u_buf[1]
        u_buf[1] = u[k]       

    return norm, alpha

class AdaptiveNotchFilter():
    def __init__(self, sr, bandwidth=0.8, mu=0.01, alpha=-1.0):
        self.u_buf = np.zeros(2)
        self.u     = None
        self.norm  = 0.0
        
        self.mu    = mu
        self.sr    = sr
        self.r     = bandwidth
        self.alpha = alpha
        self.b     = np.zeros(3)
        self.a     = np.zeros(3)
        self.set_coefs()

    def set_coefs(self):
        if   self.alpha < -(1+self.r):
            self.alpha = -(1+self.r)+1e-12
        elif self.alpha > 1+self.r:
            self.alpha = 1+self.r-1e-12
            
        self.b[0] = self.r
        self.b[1] = self.alpha
        self.b[2] = 1.0
        self.a[0] = 1.0
        self.a[1] = self.alpha
        self.a[2] = self.r

        wc0 = np.arccos(-self.alpha/(self.r+1))
        self.center_freq = wc0 * self.sr/2/np.pi
        
        print(f"fc: {self.center_freq} Hz")
        
    def freqz(self, worN=4096, plot_on=True):
        h_w, h = freqz(self.b, a=self.a, worN=worN)
        freqs = h_w / np.pi * (self.sr//2)
        
        if plot_on:
            plt.clf()
            plt.subplot(2,1,1)
            plt.plot(freqs, 20*np.log10(np.abs(h)))
            plt.plot(freqs, 20*np.log10(np.abs(0.5*(h+1))))
            plt.axvline(x=self.center_freq, color="b", linestyle='--')
            plt.xscale("log")
            plt.grid()
            plt.xlim([10,self.sr//2])
            #plt.ylim([-40,0.0])

            plt.subplot(2,1,2)
            plt.plot(freqs, np.angle(h))
            plt.xscale("log")
            plt.grid()
            plt.xlim([10,self.sr//2])
            
            plt.tight_layout()
            plt.show()
        
        return h
        
    def online(self, x, y):
        if self.u is None:
            self.u = np.zeros(x.shape[0])
        self.u[:] *= 0.0
        
        self.norm, self.alpha = compute_anf(self.norm, self.alpha, self.r, self.mu, x.shape[0], x, y, self.u, self.u_buf)
        self.set_coefs()

       
if __name__ == "__main__":
    filepath = "./sin.wav"
    savefilepath = "./save.wav"
    sig, sr = sf.read(filepath, always_2d=True)
    
    anf_list = [AdaptiveNotchFilter(sr), AdaptiveNotchFilter(sr)]
    anf_list[0].freqz()

    current_frame = 0
    blocksize = 1024
    tmpdata = np.zeros((blocksize, sig.shape[1]))
    savedata = np.zeros(sig.shape)

    def callback(outdata, frames, time, status):
        global current_frame, tmpdata, anf, anf_list
        chunksize = min(sig.shape[0] - current_frame, frames)
        tmpdata[:] *= 0.0
        tmpdata[0:chunksize] = sig[current_frame:current_frame + chunksize]

        for k in range(outdata.shape[1]):
            anf_list[k].online(tmpdata[:,k], outdata[:,k])

        if chunksize < frames:
            raise sd.CallbackStop()
        else:
            savedata[current_frame:current_frame + chunksize] = outdata[:]
        current_frame += chunksize

    with sd.OutputStream(
        samplerate=sr, 
        blocksize=blocksize,
        channels=sig.shape[1],
        callback=callback
    ):
        sd.sleep(int((sig.shape[0]+blocksize*10)//sr * 1000))
    sf.write(savefilepath, savedata, sr)

