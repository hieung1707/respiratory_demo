# UDP_SERVER_IP = '192.168.11.127' # IP for demo watch
# UDP_SERVER_IP = '192.168.11.121'
UDP_SERVER_IP = '0.0.0.0'
UDP_SERVER_PORT = 8888
UDP_CLIENT_IP = '0.0.0.0'
UDP_CLIENT_PORT = 1235
TCP_SERVER_PORT = 4322

# server config
header = b'\xB5' + b'\x62'
sampling_rate = 11025
displayed_duration = 1.295
hop = 110

# feature properties
feature_types = ['gfcc', 'mfcc', 'melspec', 'cochleagram']
win_length = 0.025
stride_length = 0.01
win_size = int(win_length * sampling_rate)
stride_size = int(stride_length * sampling_rate)
n_filters = 128

# class mapping
class_mapping = {
    0: 'crackles',
    1: 'wheeze',
    2: 'both',
    3: 'normal'
}

existing_labels = list(class_mapping.values())

# file paths
log_path = '/home/hieung1707/projects/vin_HAR/test_log/config/system_log.txt'
weights_path = '/home/hieung1707/projects/respiratory_demo/model/weights/convlstm_seq_cochleagram_final_2lstm_bn.hdf5'