import os
import scipy.signal
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt


np.random.seed(2) # 乱数シード
#===============================================
# パラメタ
sr = 16000            # サンプリング周波数
gain = 1.0            # フィードバックゲイン
blocksize = 2048      # [samples] 処理フレームのサンプル数
latency = blocksize   # [samples] システム遅延時間
n_ir_coef = blocksize # [samples] 推定するインパルス応答長
CANCELLING = True     # 適応フィルタ on/off
W_PLOT = True         # 適応フィルタ係数プロット on/offs
# パラメタ (NLMS用)
mu = 0.0025    # ステップサイズ
#===============================================

# マイク入力信号
d, _ = librosa.load(librosa.ex('libri1', hq=True), sr=sr, mono=True)
length = (d.shape[0] // blocksize) * blocksize
d = d[0:length]
# テスト信号
length = blocksize*7
d = np.zeros(length)
d[100] = 1.0

# インパルス応答生成
h = np.random.normal(0.0, 0.01, n_ir_coef)
h *= 0.0
h[152]  = -1.0


# 入出力・中間信号
n_blocks = len(d) // blocksize  # 処理ブロック数
n_tap = n_ir_coef - 1           # フィルタタップ数
w  = np.random.normal(0.0, 0.01, n_ir_coef) # 推定された音響経路のインパルス応答
x  = np.zeros(length)                         # スピーカ出力信号（システム遅延付加）
xh = np.zeros(length + n_ir_coef - 1)         # 音響経路によるフィードバック音
xw = np.zeros(length + n_ir_coef - 1)         # 適応フィルタ係数
y  = np.zeros(length)                         # d + xh
e  = np.zeros(length)                         # y - xw
error = np.zeros(n_blocks)

tmp_x       = np.zeros(blocksize)
tmp_x_buf   = np.zeros(blocksize * 2)
tmp_y       = np.zeros(blocksize)
tmp_e       = np.zeros(blocksize)
tmp_d       = np.zeros(blocksize)
tmp_xh      = np.zeros(blocksize + n_tap)
tmp_xw      = np.zeros(blocksize + n_tap)
tmp_xh_next = np.zeros(n_tap)
tmp_xw_next = np.zeros(n_tap)
tmp_x_rev   = np.zeros(n_ir_coef)         # x の信号バッファ。フィルタ係数算出時に利用
tmp_dlt_w   = np.zeros(n_ir_coef)


if not CANCELLING:
    w *= 0.0
else:
    import matplotlib.animation as animation
    fig, ax = plt.subplots()
    ims = []
    
# Main processing
for k in range(n_blocks):
    indices         = slice(k*blocksize,(k+1)*blocksize)
    indices_latency = slice(k*blocksize-latency,(k+1)*blocksize-latency)
    
    # フィードバック信号 x を生成
    if k >= latency // blocksize:
        tmp_x[:] = gain * e[indices_latency]
    else:
        tmp_x[:] = 0.0
    tmp_x_buf[0:blocksize]           = tmp_x_buf[blocksize:2*blocksize]
    tmp_x_buf[blocksize:2*blocksize] = tmp_x
    x[indices] = tmp_x
    
    # マイク入力信号を生成
    tmp_d[:] = d[indices]
    
    # 音響経路 (スピーカ->マイク) の信号 xh と音源 d のマイク入力信号 y を生成
    tmp_xh[:]        = scipy.signal.oaconvolve(tmp_x, h, mode="full")
    tmp_xh[0:n_tap] += tmp_xh_next
    tmp_xh_next[:]   = tmp_xh[blocksize:blocksize+n_tap] 
    tmp_y[:]         = tmp_d + tmp_xh[0:blocksize]

    xh[indices] = tmp_xh[0:blocksize]
    y[indices]  = tmp_y

    # 疑似音響経路の信号 xw とマイク入力信号 y の混合信号 e を生成
    tmp_xw[:]        = scipy.signal.oaconvolve(tmp_x, w, mode="full")
    tmp_xw[0:n_tap] += tmp_xw_next
    tmp_xw_next[:]   = tmp_xw[blocksize:blocksize+n_tap] 
    tmp_e[:]         = tmp_y - tmp_xw[0:blocksize]        
    
    xw[indices] = tmp_xw[0:blocksize]
    e[indices]  = tmp_e

    # 係数更新 
    tmp_dlt_w[:] = 0.0
    count_n = 0
    if k >= latency // blocksize:
        for n in range(blocksize):
            tmp_x_rev[:] = tmp_x_buf[blocksize+n-n_ir_coef:blocksize+n][::-1]
            tmp_norm_x   = np.sum(tmp_x_rev ** 2)
            """
            if CANCELLING and tmp_norm_x > 0.00001 :
                tmp_dlt_w[:] = tmp_e[n] * tmp_x_rev / tmp_norm_x
                ro = 0.0001
                mu = 0.001
                w[:] = (1.0-ro*mu)*w[:] + mu * tmp_dlt_w[:]

            if W_PLOT and (n == 0 or n == blocksize -1):
                print(f"{k+1} / {n_blocks}\tplot w\t{np.mean(w**2):0.6f}\t{np.max(np.abs(w)):0.6f}")
                ax.set_ylim([-0.5, 0.5])
                ax.grid(True)
                im1 = ax.plot(h, c="r", alpha=0.5)
                im2 = ax.plot(w, c="b", alpha=0.75)
                ims.append(im1+im2)
            """           
            if tmp_norm_x > 0.00001 :
                count_n += 1
                tmp_dlt_w[:] += tmp_e[n] * tmp_x_rev / tmp_norm_x
                
        if CANCELLING:
            tmp_error = np.mean((h-w)**2)
            if count_n > 0:
                mu = 0.1
                w[:] = w + mu * tmp_dlt_w[:] / count_n
            error[k] = np.mean((h-w)**2)

            print(f"{k+1} / {n_blocks} ({count_n})\t{tmp_error:0.6f} {error[k]:0.6f}\t{np.mean((w)**2):0.6f}\t{np.max(np.abs(w)):0.6f}")
            if W_PLOT:
                ax.set_ylim([-0.2, 0.2])
                ax.grid(True)
                im1 = ax.plot(h, c="r", alpha=0.5)
                im2 = ax.plot(w, c="b", alpha=0.75)
                ims.append(im1+im2)
            
# フィルタ係数の更新アニメーション書き出し
if CANCELLING and W_PLOT:
    ani = animation.ArtistAnimation(fig, ims, interval=100)
    ani.save("test.mp4", writer="ffmpeg")
    #ani.save("test.gif", writer="imagemagick")

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

plt.subplot(4,2,8)
plt.plot(h)
plt.grid()
plt.title("h")

plt.tight_layout()
plt.savefig("_plot.png")

import code
console = code.InteractiveConsole(locals=locals())
console.interact()
