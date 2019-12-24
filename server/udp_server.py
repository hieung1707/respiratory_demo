import socket
import time
from utils import *
import struct
import threading
import numpy as np
import librosa
import os
import resampy


# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
# Bind the socket to the port
server_address = (UDP_SERVER_IP, UDP_SERVER_PORT)
sock.bind(server_address)
client_address = (UDP_CLIENT_IP, UDP_CLIENT_PORT)
tcp_address = (UDP_CLIENT_IP, 4322)
is_logging = False
realtime = True
save_logging = False
is_handshaking = False
num_bytes = 0
percentages = 65535
range_acc = 2048
range_gyr = 256
freq = 50
mode = 0
last_logging_precentages = 0
last_logging_storage = 0
current_idx = 0
sampling_rate = 11025

TO_RAD = np.pi / 180
ACCEL_FACTOR = 2048
GYRO_FACTOR = 16.4


def generate_viz():
    global sampling_rate
    wav_path = '../data/RALE dataset/wheezing_a.wav'
    # wav_path = '//home/hieung1707/projects/respiratory_demo/data/audio_and_txt_files/101_1b1_Al_sc_Meditron.wav'
    if not os.path.exists(wav_path):
        print('File not found: {}'.format(wav_path))
    wav, sr = librosa.load(wav_path, sr=None)
    if sr != sampling_rate:
        wav = resampy.resample(wav, sr, sampling_rate)
    return wav


def rcv_data():
    old_server = True
    global mode, freq, is_logging, realtime, last_logging_precentages, last_logging_storage, percentages, num_bytes, save_logging, is_handshaking, current_idx, viz_logs
    print("SERVER IS ACTIVE")
    while True:
        (data, add) = sock.recvfrom(12)
        if data[:2] == header:
            if old_server:
                command_id = data[2]
                length = data[3]
                resp = data[4:4+length]
            else:
                command_id = data[6]
                length = data[7]
                resp = data[8:8 + length]
            if command_id == 1:
                mode = struct.unpack('B', resp)[0]
                print('Mode: {}'.format(mode))
            elif command_id == 2:
                freq = struct.unpack('H', resp)[0]
                print('Frequency: {}'.format(freq))
            elif command_id == 3:
                if resp != b'\x01':
                    continue
                print(mode)
                if mode == 0:
                    current_idx = 0
                    realtime = True
                    is_logging = False
                    save_logging = False
                    print('Total records: {}'.format(len(viz_logs)))
                elif mode == 1:
                    realtime = False
                    is_logging = True
                    save_logging = False
                    last_logging_precentages = time.time()
                    last_logging_storage = last_logging_precentages
                print('Start up', realtime, is_logging)
            elif command_id == 7:
                realtime = False
                is_logging = False
                save_logging = False
                percentages = 65535
                num_bytes = 0
                is_handshaking = True
                print("Exit mode {}".format(mode))
                time.sleep(1)
            elif command_id == 8:
                realtime = False
                is_logging = False
                save_logging = True
                current_idx = 0
                print('Saving log...')
            elif command_id == 9:
                save_logging = False
                print("Erasing log")
            elif command_id == 11:
                is_handshaking = False
                print("Handshake succeed with {}".format(add))
                time.sleep(5)


def send_data():
    global percentages, num_bytes, is_logging, realtime, last_logging_precentages, last_logging_storage, save_logging, logs, current_idx, is_handshaking, viz_logs
    device_id = struct.pack('I', 1234)
    while True:
        data = header + device_id
        if is_handshaking:
            print('is handshaking')
            command_id = b'\x0B'
            length = b'\x01'
            resp = b'\01'
            data += command_id + length + resp
            sock.sendto(data, client_address)
            time.sleep(1)
            continue
        elif is_logging:
            curr_time = time.time()
            is_sending = False
            if curr_time - last_logging_storage >= 2.0:
                print('logging storage')
                command_id = b'\x04'
                data += command_id + struct.pack('B', 4) + struct.pack('I', num_bytes)
                num_bytes += int(freq * 14)
                sock.sendto(data, client_address)
                is_sending = True
                last_logging_storage = curr_time
            if curr_time - last_logging_precentages >= 4.0:
                print('logging percentage')
                command_id = b'\x05'
                data += command_id + struct.pack('B', 2) + struct.pack('H', percentages)
                percentages = max(0, percentages - 1)
                sock.sendto(data, client_address)
                is_sending = True
                last_logging_precentages = curr_time
            if is_sending:
                command_id = b'\x06'
                data += command_id + struct.pack('B', 4) + struct.pack('I', int(curr_time))
                sock.sendto(data, client_address)
            continue
        elif realtime:
            wav_samples = viz_logs[current_idx:current_idx+hop]
            length = b'\x01'
            command_id = b'\x0c'
            total_length = wav_samples.shape[0]
            if current_idx + hop > len(viz_logs):
                remainings = current_idx + hop - len(viz_logs)
                wav_samples = np.append(wav_samples, viz_logs[:remainings])
            data = header + command_id + length + struct.pack('H', hop)
            for i in range(hop):
                data += struct.pack('f', wav_samples[i])
            sock.sendto(data, client_address)
            sleep_time = hop * 1. / sampling_rate
            # print(sleep_time)
            time.sleep(sleep_time)
            current_idx = (current_idx + hop + 1) % len(viz_logs)
            continue
        elif save_logging:
            tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            tcp_sock.connect(tcp_address)
            print('connected')
            if current_idx == len(logs):
                data += b'\x0a' + struct.pack('B', 1) + struct.pack('B', 1)
                tcp_sock.send(data)
                tcp_sock.close()
                save_logging = False
                print(len(data))
                print("DONE TRANSMITTING")
                tcp_sock.close()
                continue
            battery_info = struct.pack('H', 65535) + b'\x00' + b'\x00'
            acc_x = struct.pack('h', logs[current_idx][1])
            acc_y = struct.pack('h', logs[current_idx][2])
            acc_z = struct.pack('h', logs[current_idx][3])
            gyr_x = struct.pack('h', logs[current_idx][4])
            gyr_y = struct.pack('h', logs[current_idx][5])
            gyr_z = struct.pack('h', logs[current_idx][6])
            data += b'\x08' + struct.pack('B', 14) + battery_info + acc_x + acc_y + acc_z + gyr_x + gyr_y + gyr_z
            # sock.sendto(data, client_address)
            tcp_sock.send(data)
            tcp_sock.close()
            current_idx += 1
            time.sleep(2./freq)
            continue


if __name__ == '__main__':
    viz_logs = generate_viz()
    t = threading.Thread(target=send_data)
    t.start()
    print("Sending data")
    rcv_data()
