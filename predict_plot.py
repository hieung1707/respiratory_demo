import socket
import struct
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from collections import deque
import os
import datetime
from utils import *
import threading
from model.classification_model import Model

duration = int(sampling_rate*displayed_duration)
info = deque([[0]]*duration, maxlen=duration)

first_time = True
first_ts = time.time()
total_time = 0
total_count = 0
last_ts = 0
max_gyr = 0
diff = 0
is_updating = False


class DataPlot:
    def __init__(self, maxlen=3, max_entries=100):
        self.maxlen = maxlen
        self.max_entries = max_entries

        self.axis_x = deque(maxlen=max_entries)
        self.axis_ys = []
        for idx in range(maxlen):
            self.axis_ys.append(deque(maxlen=max_entries))

        for cnt in range(max_entries):
            self.axis_x.append(cnt)
            for idx in range(maxlen):
                self.axis_ys[idx].append(0.0)

    def add(self, x, ys):
        self.axis_x.append(x)
        for idx in range(self.maxlen):
            self.axis_ys[idx].append(ys[idx])

    def add_ys(self, ys):
        for idx in range(self.maxlen):
            self.axis_ys[idx].append(ys[idx])


class RealtimePlot:
    def __init__(self, axes, data_plot, colors, xlim=(0, 100), ylim=(-5, 5)):
     
        self.axes = axes
        self.data_plot = data_plot
        self.colors = colors
        self.lineplot = []

        self.axes.set_xlim(xlim)
        self.axes.set_ylim(ylim)
        self.axes.relim()

        for idx, color in enumerate(colors):
            self.lineplot.append(axes.plot([], [], color=color, animated=True)[0])

    def plot(self):
        for idx, color in enumerate(self.colors):
            self.lineplot[idx].set_data(self.data_plot.axis_x, self.data_plot.axis_ys[idx])
            # self.lineplot[idx].set_ydata(self.data_plot.axis_ys[idx])


def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP


class UDPServer:
    def __init__(self, sock=None, host='0.0.0.0', port=1234, target_ip= (UDP_SERVER_IP, UDP_SERVER_PORT)):
        self.target_ip = target_ip
        if sock is None:
            self.s = socket.socket
            self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            print("Listening on udp %s:%s" % (get_local_ip(), port))
            self.s.bind((host, port))
        else:
            self.s = sock

        self.fig = plt.figure(figsize=(15, 7))
        self.max_len = duration
        self.ax_label = self.fig.add_subplot(2, 1, 1)
        self.ax_label.axis('off')
        left, width = .25, .5
        bottom, height = .25, .5
        right = left + width
        top = bottom + height
        self.text = self.ax_label.text(0.5 * (left + right), 0.5 * (bottom + top), 'A text',
                                       horizontalalignment='center',
                                       verticalalignment='center',
                                       fontsize=40, color='red',
                                       transform=self.ax_label.transAxes)
        self.text.set_text('')
        self.text.set_color('black')
        self.ax_wav = self.fig.add_subplot(2, 1, 2)
        self.ax_wav.set_title('Wave samples')
        self.data_wav = DataPlot(maxlen=1, max_entries=self.max_len)
        # self.data_gyroscope = DataPlot(max_entries=self.max_len)

        self.colors = ['blue']
        self.plot_wav = RealtimePlot(self.ax_wav, self.data_wav, self.colors, xlim=(0, self.max_len), ylim=(-1, 1))
        log_folder = 'log'
        if not os.path.exists(log_folder):
            os.mkdir(log_folder)

        self.last_count = 0
        self.first_time = True
        self.is_updating = False
        self.first_ts = time.time()
        self.total_time = 0
        self.hop = int(0.335) * sampling_rate
        self.total_count = 0
        self.last_ts = 0
        self.max_gyr = 0
        self.diff = 0
        self.nsamples_since_last_predict = 0
        self.rcv_bytes = 6 + hop * 4
        self.last_prediction = ''
        self.info = deque([[0]*6]*self.max_len, maxlen=self.max_len)
        self.is_running = True
        self.is_predicting = False
        self.has_header = True
        self.has_timestamp = False
        self.start_bytes, self.end_bytes = self.get_package_struct()
        # model
        self.model = Model()

    def get_package_struct(self):
        start_bytes = []
        end_bytes = []
        current_idx = 0

        if self.has_header:
            start_bytes.append(current_idx)
            end_bytes.append(current_idx+2)
            current_idx += 2

        if self.has_timestamp:
            start_bytes.append(current_idx)
            end_bytes.append(current_idx+4)
            current_idx += 4

        start_bytes.append(current_idx)
        end_bytes.append(current_idx+2)
        current_idx += 2

        for i in range(hop):
            start_bytes.append(current_idx)
            end_bytes.append(current_idx+4)
            current_idx += 4

        return start_bytes, end_bytes
    
    def update_predict(self, info):
        temp_pred = self.model.predict_one_label(info)
        if temp_pred == 3:
            temp_pred = ''
        # print(temp_pred)
        self.last_prediction = temp_pred
        # if temp_pred != self.last_prediction:
        #     self.last_prediction = temp_pred
        #     self.text.set_text(self.last_prediction)
        #     self.fig.canvas.draw()
        self.is_predicting = False

    def read_data(self):
        timestamp = 0
        while self.is_running:
            curr_time = time.time()
            (data, addr) = self.s.recvfrom(self.rcv_bytes)
            if data:
                if self.first_time:
                    self.first_ts = curr_time
                    self.last_ts = self.first_ts
                    self.first_time = False
                    self.diff = 0
                self.total_time = curr_time - self.first_ts
                # self.total_count = self.total_count + stride_size
                self.diff += 1
                # self.is_updating = True
                # This is for custom hardware
                data_header = data[:2]
                if data_header != header:
                    print('got something else')
                    continue
                command_id = data[2:3]
                length = data[3:4]
                if command_id != b'\x0c':
                    continue
                total_length = struct.unpack('H', data[4:6])[0]
                curr_idx = 6
                for i in range(total_length):
                    wav_sample = struct.unpack('f', data[curr_idx:curr_idx+4])[0]
                    self.info.append([wav_sample])
                    curr_idx += 4

                self.total_count += total_length
                self.nsamples_since_last_predict += total_length

                if curr_time - self.last_ts >= 1.:
                    # print(self.diff, self.total_count * 1. / (self.total_time + 1e-16), curr_time - self.last_ts)
                    self.last_ts = curr_time
                    self.diff = 0
                # self.is_updating = False

    def receive_and_animate(self, frame):
        if self.last_count == self.total_count:
            return self.plot_wav.lineplot + [self.text]
        if self.nsamples_since_last_predict >= self.hop:
            predict_thread = threading.Thread(target=self.update_predict, args=self.info)
            predict_thread.start()
        self.last_count = self.total_count
        self.text.set_text(self.last_prediction)
        # bytes received from watch, 16 for custom hardware and 56 for apple watch series 3
        for i in range(-1 - hop, 0):
            self.data_wav.add_ys(np.array(self.info[i], dtype=np.float32))
        self.plot_wav.plot()
        return self.plot_wav.lineplot + [self.text]

    def send_exit_command(self):
        data = header + b'\x07' + struct.pack('B', 0)
        self.s.sendto(data, self.target_ip)

    def start_plotting(self):
        ani = animation.FuncAnimation(self.fig, self.receive_and_animate, interval=10, blit=True)
        plt.show()
        self.is_running = False
        # self.send_exit_command()


if __name__ == "__main__":
    server = UDPServer(sock=None, host=UDP_CLIENT_IP, port=UDP_CLIENT_PORT)
    t = threading.Thread(target=server.read_data)
    t.start()
    server.start_plotting()
