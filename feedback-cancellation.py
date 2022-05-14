import os
import scipy
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

np.random.seed(2) # 乱数シード
#===============================================
# パラメタ
sr = 16000           # サンプリング周波数
gain = 1.0           # フィードバックゲイン
latency   = sr//100  # [samples] システム遅延時間
n_ir_coef = 1024     # [samples] 推定するインパルス応答長
CANCELLING = True    # 適応フィルタ on/off
# パラメタ (NLMS用)
mu = 0.0025           # ステップサイズ
beta = 0.00001       # ゼロ割防止用
#===============================================

#  マイク入力信号
d, _ = librosa.load(librosa.ex('libri1', hq=True), sr=sr, mono=True)
# テスト信号
#d = np.zeros(sr//5)
#d[100] = 1.0
length = d.shape[0]

# インパルス応答生成
h = np.random.laplace(0.0, 0.004, n_ir_coef)
h[50]  = 0.4
h[173] = -0.2

# 入出力・中間信号
w  = np.random.laplace(0.0, 0.001, n_ir_coef) # 推定された音響経路のインパルス応答
x  = np.zeros(length)                         # スピーカ出力信号（システム遅延付加）
xh = np.zeros(length + n_ir_coef - 1)         # 音響経路によるフィードバック音
xw = np.zeros(length + n_ir_coef - 1)         # 適応フィルタ係数
y  = np.zeros(length)                         # d + xh
e  = np.zeros(length)                         # y - xw
tmp_x_rev  = np.zeros(n_ir_coef)              # x の信号バッファ。フィルタ係数算出時に利用

if not CANCELLING:
    w *= 0.0
else:
    import matplotlib.animation as animation
    fig, ax = plt.subplots()
    ims = []


for n in range(length):
    # フィードバック信号 x を生成
    if n >= latency:
        x[n] = gain * e[n-latency]
    else:
        x[n] = 0.0
    
    # 音響経路(スピーカ->マイク)の信号 xh と音源 d のマイク入力信号 y を生成
    xh[n:n+n_ir_coef] += x[n] * h # 実環境では h は未知
    y[n] = d[n] + xh[n] 
        
    # 疑似音響経路の信号 xw とマイク入力信号 y の混合信号 e を生成
    xw[n:n+n_ir_coef] += x[n] * w
    e[n]              = y[n] - xw[n]
    
    # 係数更新
    if n >= n_ir_coef:
        tmp_x_rev[:] = x[n-n_ir_coef:n][::-1]
    else:
        tmp_npad = n_ir_coef - n
        tmp_x_rev[:] = 0.0
        tmp_x_rev[0:tmp_npad] = x[0:tmp_npad][::-1]
        
    norm_x = np.sum(tmp_x_rev ** 2)
    
    if norm_x > 0.00001 and CANCELLING:
        w[:] = w + mu * e[n] * tmp_x_rev / (norm_x + beta) 
        
        # 動画用プロット作成
        if True and n % (sr//10) == 0:
            print(f"{n} / {length}\tplot w")
            ax.set_ylim([-0.5, 0.5])
            ax.grid(True)
            im1 = ax.plot(h, c="r", alpha=0.5)
            im2 = ax.plot(w, c="b", alpha=0.75)
            ims.append(im1+im2)

if CANCELLING:
    # フィルタ係数のプロット動画書き出し
    ani = animation.ArtistAnimation(fig, ims, interval = 100)
    #ani.save("test.mp4", writer="ffmpeg")
    ani.save("test.gif", writer="imagemagick")
    
    
sf.write(f"feedback_d.wav", d, sr)
sf.write(f"feedback_e.wav", e, sr)

plt.clf()
n_x_max = length
plt.subplot(4,2,1)
plt.plot(d)
plt.xlim([0, n_x_max])
plt.ylim([-1.0, 1.0])
plt.grid()
plt.title("d")
plt.subplot(4,2,3)
plt.plot(e)
plt.xlim([0, n_x_max])
plt.ylim([-1.0, 1.0])
plt.grid()
plt.title("e")
plt.subplot(4,2,5)
plt.plot(x)
plt.xlim([0, n_x_max])
plt.ylim([-1.0, 1.0])
plt.grid()
plt.title("x")
plt.subplot(4,2,7)
plt.plot(abs(d[0:e.shape[0]]-e)**2)
plt.xlim([0, n_x_max])
plt.grid()
plt.title("(e-d)**2")

plt.subplot(4,2,2)
plt.plot(xh)
plt.xlim([0, n_x_max])
plt.ylim([-1.0, 1.0])
plt.grid()
plt.title("xh")
plt.subplot(4,2,4)
plt.plot(xw)
plt.xlim([0, n_x_max])
plt.ylim([-1.0, 1.0])
plt.grid()
plt.title("xw")
plt.subplot(4,2,6)
plt.plot(y)
plt.xlim([0, n_x_max])
plt.ylim([-1.0, 1.0])
plt.grid()
plt.title("y")

plt.tight_layout()
plt.show()


import code
console = code.InteractiveConsole(locals=locals())
console.interact()


