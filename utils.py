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
    0: 'Rì rào',
    1: 'Khò khè',
    2: 'Rì rào lẫn khò khè',
    3: 'Bình thường'
}

existing_labels = list(class_mapping.values())

# file paths
log_path = '/home/hieung1707/projects/vin_HAR/test_log/config/system_log.txt'
weights_path = '/home/hieung1707/projects/respiratory_demo/model/weights/convlstm_seq_cochleagram_final_2lstm_bn.hdf5'


def read_labels(file_path):
    labels = []
    with open(file_path, 'r+') as label_file:
        lines = label_file.readlines()
        for line in lines:
            line = line.replace('\n', '')
            data = line.split('\t')
            start_idx = int(float(data[0]) * sampling_rate)
            end_idx = int(float(data[1]) * sampling_rate)
            has_crackles = int(data[2])
            has_wheeze = int(data[3])
            if has_crackles and not has_wheeze:
                label = 0
            elif not has_crackles and has_wheeze:
                label = 1
            elif has_crackles and has_wheeze:
                label = 2
            else:
                label = 3
            labels.append({
                'start': start_idx,
                'end': end_idx,
                'label': label
            })
    return labels