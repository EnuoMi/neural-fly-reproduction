import os, re
from typing import List, Dict
from ast import literal_eval
from collections import namedtuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

folder = './data/experiment'
filename_fields = ['vehicle', 'trajectory', 'method', 'condition']


def smooth_labels(y, win=30, axis=0):
    """
    对标签做滑动平均平滑（反射边界版）
    - 支持 y 形状: (T,), (T, D), (D, T)
    - 对每个分量沿时间轴独立平滑
    win: 窗口长度（点数）
    axis: 时间轴（默认 0）。如果你的 y 是 (3, T)，传 axis=1。
    """
    if win <= 1:
        return y

    y = np.asarray(y)
    kernel = np.ones(win, dtype=float) / win
    pad = win // 2

    # 统一把时间轴挪到第0维： (T, ...) 方便处理
    y0 = np.moveaxis(y, axis, 0)  # time-first
    T = y0.shape[0]
    rest_shape = y0.shape[1:]     # e.g. (3,) or (D,)

    # reshape 成 (T, K)，逐列平滑
    y2 = y0.reshape(T, -1)
    y2_s = np.empty_like(y2)

    for k in range(y2.shape[1]):
        sig = y2[:, k]
        sig_pad = np.pad(sig, pad_width=pad, mode="reflect")
        sig_s = np.convolve(sig_pad, kernel, mode="valid")
        y2_s[:, k] = sig_s[:T]

    # reshape 回原状，再把时间轴移回去
    y0_s = y2_s.reshape((T,) + rest_shape)
    y_s = np.moveaxis(y0_s, 0, axis)

    return y_s

def save_data(Data: List[dict], folder: str,
              fields=['t', 'p', 'p_d', 'v', 'v_d', 'q', 'R', 'w', 'T_sp', 'q_sp', 'hover_throttle', 'fa', 'pwm']):
    ''' Save {Data} to individual csv files in {folder}, serializing (2+)d ndarrays as lists '''
    if not os.path.isdir(folder):
        os.makedirs(folder)
        print('Created data folder ' + folder)
    for data in Data:
        if 'fa' in fields and 'fa' not in data:
            data['fa'] = data['fa_num_Tsp']

        df = pd.DataFrame()

        missing_fields = []
        for field in fields:
            try:
                df[field] = data[field].tolist()
            except KeyError as err:
                missing_fields.append(field)
        if len(missing_fields) > 0:
            print('missing fields ', ', '.join(missing_fields))

        filename = '_'.join(data[field] for field in filename_fields)
        df.to_csv(f"{folder}/{filename}.csv")


def load_data(folder: str, expnames=None) -> List[dict]:
    ''' Loads csv files from {folder} and return as list of dictionaries of ndarrays '''
    Data = []

    if expnames is None:
        filenames = os.listdir(folder)
    elif isinstance(expnames, str):  # if expnames is a string treat it as a regex expression
        filenames = []
        for filename in os.listdir(folder):
            if re.search(expnames, filename) is not None:
                filenames.append(filename)
    elif isinstance(expnames, list):
        filenames = (expname + '.csv' for expname in expnames)
    else:
        raise NotImplementedError()
    for filename in filenames:
        # Ingore not csv files, assume csv files are in the right format
        if not filename.endswith('.csv'):
            continue

        # Load the csv using a pandas.DataFrame
        df = pd.read_csv(folder + '/' + filename)

        # Lists are loaded as strings by default, convert them back to lists
        for field in df.columns[1:]:
            if isinstance(df[field][0], str):
                df[field] = df[field].apply(literal_eval)

        # Copy all the data to a dictionary, and make things np.ndarrays
        Data.append({})
        for field in df.columns[1:]:
            Data[-1][field] = np.array(df[field].tolist(), dtype=float)

        # Add in some metadata from the filename
        namesplit = filename.split('.')[0].split('_')
        for i, field in enumerate(filename_fields):
            Data[-1][field] = namesplit[i]
        # Data[-1]['method'] = namesplit[0]
        # Data[-1]['condition'] = namesplit[1]

    return Data


SubDataset = namedtuple('SubDataset', 'X Y C meta')
feature_len = {}


# def format_data(RawData: List[Dict['str', np.ndarray]], features: 'list[str]' = ['v', 'q', 'pwm'], output: str = 'fa',
#                 hover_pwm_ratio=1.):
#     ''' Returns a list of SubDataset's collated from RawData.
#
#         RawData: list of dictionaries with keys of type str. For keys corresponding to data fields, the value should be type np.ndarray.
#         features: fields to collate into the SubDataset.X element
#         output: field to copy into the SubDataset.Y element
#         hover_pwm_ratio: (average pwm at hover for testing data drone) / (average pwm at hover for training data drone)
#          '''
#     Data = []
#     for i, data in enumerate(RawData):
#         # Create input array
#         X = []
#         for feature in features:
#             if feature == 'pwm':
#                 X.append(data[feature] / 1000 * hover_pwm_ratio)
#             else:
#                 X.append(data[feature])
#             feature_len[feature] = len(data[feature][0])
#         X = np.hstack(X)
#
#         # Create label array
#         Y = data[output]
#
#         # Pseudo-label for cross-entropy
#         C = i
#
#         # Save to dataset
#         Data.append(SubDataset(X, Y, C, {'method': data['method'], 'condition': data['condition'], 't': data['t']}))
#
#     return Data

def format_data(RawData, features=['v', 'q', 'pwm'], output='fa',
                hover_pwm_ratio=1., smooth_win=1, pwm_delay_steps=3):
    """
    Returns a list of SubDataset's collated from RawData.

    RawData: list of dictionaries with keys of type str. For keys corresponding to data fields,
             the value should be type np.ndarray.
    features: fields to collate into the SubDataset.X element
    output: field to copy into the SubDataset.Y element
    hover_pwm_ratio: (average pwm at hover for testing data drone) / (average pwm at hover for training data drone)
    smooth_win: optional label smoothing window (default 1 = no smoothing)
    pwm_delay_steps: number of delay steps for pwm. If 3, stack [pwm(t), pwm(t-1), pwm(t-2), pwm(t-3)]
    """
    Data = []
    k = int(pwm_delay_steps)

    # 小工具：保证二维 (T, d)
    def as_2d(a):
        a = np.asarray(a)
        if a.ndim == 1:
            a = a[:, None]
        return a

    for i, data in enumerate(RawData):
        # -------------------------
        # 1) label & time
        # -------------------------
        Y_full = as_2d(data[output])  # fa 通常是 (T, 3)
        T = Y_full.shape[0]

        t_full = np.asarray(data.get('t', np.arange(T)))

        # -------------------------
        # 2) 裁剪：为了 pwm(t-k) 存在，起点从 t=k
        # -------------------------
        if k > 0:
            sl = slice(k, T)   # 仅保留 [k, ..., T-1]
        else:
            sl = slice(0, T)

        Y = Y_full[sl]
        t_used = t_full[sl]

        # -------------------------
        # 3) input X
        # -------------------------
        X_parts = []

        # 注意：feature_len 是 utils.py 里的全局 dict，plot_subdataset 依赖它
        # 这里每个 episode 都会覆盖写入同样的 key（维度应当一致）
        for feature in features:
            arr_full = as_2d(data[feature])   # (T, d_feature)
            base_dim = arr_full.shape[1]

            if feature == 'pwm':
                # 单位缩放
                arr_full = arr_full / 1000.0 * hover_pwm_ratio

                # 延迟堆叠：[pwm(t), pwm(t-1), ..., pwm(t-k)]
                blocks = []
                for d in range(0, k + 1):
                    blocks.append(arr_full[k - d : T - d])  # (T-k, d_pwm)
                arr_used = np.hstack(blocks)               # (T-k, d_pwm*(k+1))

                # 更新 feature_len：pwm 维度扩大 (k+1) 倍
                feature_len[feature] = base_dim * (k + 1)

            else:
                # 其它特征只取对齐段
                arr_used = arr_full[sl]

                # 更新 feature_len：保持原维度
                feature_len[feature] = base_dim

            X_parts.append(arr_used)

        X = np.hstack(X_parts)  # (T-k, total_dim)

        # -------------------------
        # 4) optional label smoothing (一般你现在不需要，保持默认即可)
        # -------------------------
        if smooth_win and smooth_win > 1:
            Y = smooth_labels(Y, win=smooth_win)

        # -------------------------
        # 5) Pseudo-label
        # -------------------------
        C = i

        # -------------------------
        # 6) Save to dataset
        # -------------------------
        Data.append(SubDataset(
            X, Y, C,
            {'method': data['method'], 'condition': data['condition'], 't': t_used}
        ))

    return Data



def plot_subdataset(data, features, title_prefix=''):
    fig, axs = plt.subplots(1, len(features) + 1, figsize=(10, 4))
    idx = 0
    for feature, ax in zip(features, axs):
        for j in range(feature_len[feature]):
            ax.plot(data.meta['t'], data.X[:, idx], label=f"{feature}_{j}")
            idx += 1
        ax.legend()
        ax.set_xlabel('time [s]')
    ax = axs[-1]
    ax.plot(data.meta['t'], data.Y)
    ax.legend(('fa_x', 'fa_y', 'fa_z'))
    ax.set_xlabel('time [s]')
    fig.suptitle(f"{title_prefix} {data.meta['condition']}: c={data.C}")
    fig.tight_layout()
