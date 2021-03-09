import cv2
import pyaudio
import wave
import numpy as np
import simpleaudio as sa
import matplotlib.pyplot as plt

def rekam_audio():
    filename = "recorded.wav"
    chunk = 1024
    FORMAT = pyaudio.paInt16
    channels = 1 # 1=mono 2=stereo
    sample_rate = 44100
    record_seconds = 5

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    output=True,
                    frames_per_buffer=chunk)
    frames = []
    print("Recording...")
    for i in range(int(44100 / chunk * record_seconds)):
        data = stream.read(chunk)
        #stream.write(data) #saat merekam akan terdengar juga suara kita
        frames.append(data)
    print("Finished recording.")
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(filename, "wb")
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(sample_rate)
    wf.writeframes(b"".join(frames))
    wf.close()

def play_audio():
    filename = 'recorded.wav'
    wave_obj = sa.WaveObject.from_wave_file(filename)
    play_obj = wave_obj.play()
    play_obj.wait_done()

def plot_audio():
    audio1 = wave.open("recorded.wav")
    signal = audio1.readframes(-1)
    signal = np.frombuffer(signal, dtype="int16")
    fs = audio1.getframerate()

    time = np.linspace(0, len(signal) / fs, num=len(signal))

    plt.figure(1)
    plt.title("Sound Signal Wave")
    plt.xlabel("Time")
    plt.plot(time, signal)
    plt.show()

def image():
    video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        check, frame = video.read()
        cv2.imshow("Capturing",frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    cv2.imwrite("filename.jpg",frame)
    video.release()
    cv2.destroyAllWindows()

def show_image():
    gambar = cv2.imread("filename.jpg", cv2.IMREAD_COLOR)
    cv2.imshow("Hasil Foto", gambar)
    cv2.waitKey(0)

while True:
    print("1. Audio (Record & Save)\n2. Play Audio\n3. Plot Signal Audio\n4. Image (OpenCam, Capture Image & Save Image)\n5. Show Image\n6. Exit")
    print("Pilihan : ")
    pilihan = int(input())
    if pilihan == 1:
        rekam_audio()
    elif pilihan == 2:
        play_audio()
    elif pilihan == 3:
        plot_audio()
    elif pilihan == 4:
        print("Tekan 'q' untuk snapshot")
        image()
    elif pilihan == 5:
        show_image()
    elif pilihan == 6:
        exit()
    else:
        print("Pilihan Tidak Ada!")