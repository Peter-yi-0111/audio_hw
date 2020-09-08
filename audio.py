import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

audio_path = '/Users/peterchen/Documents/lab/audio_hw/語音訊號處理.wav'
y , sr = librosa.load(audio_path,sr=None)

#hamming window
window = np.array([0.54 - 0.46 * np.cos(2 * np.pi * n / (512-1))for n in range(512)])

#frameshift = 16000 * 0.016 = 256
#framsize = 16000 * 0.032 = 512

#分偵的function
y_frame2 = librosa.util.frame(y,frame_length = 512,hop_length = 256)

#將音訊矩陣轉置，音訊矩陣變為(200,512)，才能和hamming window 相乘
y_frame2 = np.transpose(y_frame2)

#ans為加窗矩陣
ans = y_frame2 * window

#分窗的合成圖
plt.figure()
a1 = np.arange(512)
for i in range(5): ##range(5)，5個迭代，可以更改到200
    a2 = a1 + 512*i
    plt.plot(a2/16000,ans[i])

plt.title('wave window')
plt.legend()

#能量：加絕對值
energy = abs(ans)

#energy的圖
plt.figure()
for i in range(5): #range(5)，5個迭代，可以更改到200
    a2 = a1 + 512*i

    plt.plot(a2/16000,energy[i])
plt.title('energy')
plt.legend()

#過零率的公式
def sgn(a):
    if a > 0:
        return 1
    else:
        return -1

def ZCR( wavedata , framesize):
    frame_num = len(wavedata)
    zcr = np.zeros((frame_num,1))
    for i in range(frame_num):
        for j in range(1,512):
            if abs(sgn(wavedata[i][j])-sgn(wavedata[i][j-1])) > 0:
                zcr[i] += 1
        zcr[i] /= 512
    return zcr

zcr = ZCR(ans , 512 )
a1 = np.arange(200)
plt.figure()
plt.title('zero crossing rate')
plt.plot(a1*256/16000,zcr)
plt.legend()

#actocorrelation 公式
def actocorrelation(frame_number):
    actocorrelation = []
    for i in range(200):
        actocorrelation.append(np.sum(ans[frame_number]*ans[(frame_number+i)%200]))
    return actocorrelation

plt.figure()
plt.title('actocorrelation')
plt.plot(actocorrelation(199)) #199為framenumber 可以更改0-199
plt.legend()

#pitch的計算
def pitch(frame_number):
    temp = []
    time = []
    pitch = 0
    temp = actocorrelation(frame_number)
    for i in range(1,199):
        if temp[i] > temp[i-1] and temp[i] > temp[i+1]:
            time.append((temp[i],i))
    if time == []:
        return pitch
    time.sort(reverse = True)
    for i in range(1,5):
        pitch = abs(time[i][1]-time[i-1][1])+pitch
    pitch = pitch /4
    pitch = pitch *256 /16000
    return 1/pitch

pitch2 = []
for frame_number in range(200):
    pitch2.append(pitch(frame_number))

plt.figure()
plt.title('pitch')
plt.plot(pitch2)
plt.legend()

#end-point detection的攻勢
n = 10
average = []
for i in range(n):
    average.append(np.sum(abs(ans[i]) )/512)
max = np.max(average)
min = np.min(average)
I1 = 0.03 * (max - min) + min
I2 = 4 * min
ITL = np.min((I1,I2))
ITU = 5 * ITL

IZC1 = []
for i in range(n):
    IZC1.append(zcr[i])
IZC = np.sum(IZC1) / n
std = np.std(IZC1)

IZCT = np.min((25,IZC + 2*std))

#找N1
N1 = None
for i in range(n,200):
    start = sum(abs(ans[i]))/512
    if N1 != None:
        if start < ITL:
            N1 = None
        if start > ITU:
            break
    if N1 == None and start > ITL:
        N1 = i
s = 0
for i in range(N1,np.maximum(N1-25,0),-1):
    if zcr[i] >= IZCT:
        s+=1
        a = i
if s>=3:
    N1 = a
N1 = N1 * 256 / 16000
#找N2
N2 = None
for i in range(199,n-1,-1):
    start = sum(abs(ans[i]))/512
    if N2 != None:
        if start < ITL:
            N2 = None
        if start > ITU:
            break
    if N2 == None and start > ITL:
        N2 = i
N2 = N2 * 256 / 16000

plt.figure()
plt.title('end-point detction')
librosa.display.waveplot(y, sr)
plt.axvline(N1,color = 'r')
plt.axvline(N2,color = 'g')
plt.show()