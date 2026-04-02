import torch
from torch.utils.data import DataLoader
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='zte4g/',
                 target='OT', scale=True, timeenc=0, freq='h', percent=100, bs=None):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        self.bs = bs
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent
        # self.adj_k = adj_k

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        # print('# of base stations: ', self.enc_in)
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        # self.scaler = Chronos2MultiVariateScaling(n_vars=len(self.bs))
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path + '/traffic.csv'), index_col=0)
        df_raw.fillna(0, inplace=True)

        # df_raw = df_raw.iloc[:, :20]

        cluster_bs = [str(cell) for cell in self.bs]
        df_raw = df_raw.filter(cluster_bs)
        # noised = True
        # if noised:
        #     df_raw.iloc[-200:-170, :] = 0
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            df_data = df_raw
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
            # data = self.scaler(df_data.values, mode='norm')
        else:
            data = df_data.values

        df_stamp = pd.DataFrame({'date': df_raw.index.values[border1:border2]})
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2].astype(np.float32)
        self.data_y = data[border1:border2].astype(np.float32)
        self.data_stamp = data_stamp.astype(np.float32)

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        # seq_x = self.data_x[s_begin:s_end, self.adj_k[feat_id]]
        # seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        # seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        seq_y_mark = self.data_stamp[r_begin:r_end]

        seq_x_mark = self.data_stamp[s_begin:s_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        # return self.scaler.inverse_transform(data)
        return self.scaler(data, mode='denorm')


def data_provider(args, flag):
    Data = Dataset_Custom
    timeenc = 0 if args.embed != 'timeF' else 1
    percent = args.percent

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        percent=percent,
        bs=args.station_ids
    )

    # Add these explicit worker seeding parameters
    import torch
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=0,
        worker_init_fn=lambda worker_id: np.random.seed(args.seed + worker_id),
        persistent_workers=False,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last
    )

    return data_set, data_loader
