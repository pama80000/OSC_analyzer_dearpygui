#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
データ処理ユーティリティモジュール。

波形データの処理に関する共通機能を提供します。
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
from typing import Dict, List, Tuple, Any, Optional, Union


def apply_window(data: np.ndarray, window_type: str = 'hann') -> np.ndarray:
    """
    窓関数を適用する。

    Args:
        data: 入力データ
        window_type: 窓関数の種類 ('hann', 'hamming', 'blackman', 'rectangular')

    Returns:
        窓関数が適用されたデータ
    """
    if window_type == 'hann':
        window = np.hanning(len(data))
    elif window_type == 'hamming':
        window = np.hamming(len(data))
    elif window_type == 'blackman':
        window = np.blackman(len(data))
    elif window_type == 'rectangular':
        window = np.ones(len(data))
    else:
        raise ValueError(f"未対応の窓関数タイプ: {window_type}")
    
    return data * window


def compute_fft(data: np.ndarray, sample_rate: float = 1.0, window_type: str = 'hann') -> Tuple[np.ndarray, np.ndarray]:
    """
    FFTを計算する。

    Args:
        data: 入力データ
        sample_rate: サンプリングレート（Hz）
        window_type: 窓関数の種類

    Returns:
        (周波数軸, FFT振幅)のタプル
    """
    # 窓関数の適用
    windowed_data = apply_window(data, window_type)
    
    # FFTの計算
    n = len(data)
    fft_data = fft(windowed_data)
    
    # 周波数軸の計算
    freq = fftfreq(n, 1/sample_rate)
    
    # 正の周波数のみを取得
    positive_freq_idx = np.arange(1, n//2)
    freq = freq[positive_freq_idx]
    fft_amplitude = 2.0/n * np.abs(fft_data[positive_freq_idx])
    
    return freq, fft_amplitude


def apply_filter(data: np.ndarray, filter_type: str, cutoff_freq: Union[float, Tuple[float, float]],
                 order: int = 4, sample_rate: float = 1.0) -> np.ndarray:
    """
    フィルタを適用する。

    Args:
        data: 入力データ
        filter_type: フィルタの種類 ('lowpass', 'highpass', 'bandpass', 'bandstop')
        cutoff_freq: カットオフ周波数（Hz）、bandpassとbandstopの場合は(低域, 高域)のタプル
        order: フィルタの次数
        sample_rate: サンプリングレート（Hz）

    Returns:
        フィルタリングされたデータ
    """
    nyq = 0.5 * sample_rate
    
    if filter_type in ['bandpass', 'bandstop']:
        if not isinstance(cutoff_freq, tuple) or len(cutoff_freq) != 2:
            raise ValueError("bandpassとbandstopフィルタには(低域, 高域)のタプルが必要です")
        
        low, high = cutoff_freq
        low_norm = low / nyq
        high_norm = high / nyq
        
        if low_norm >= high_norm:
            raise ValueError("低域カットオフは高域カットオフより小さくなければなりません")
        
        normal_cutoff = [low_norm, high_norm]
    else:
        if isinstance(cutoff_freq, tuple):
            raise ValueError("lowpassとhighpassフィルタには単一のカットオフ周波数が必要です")
        
        normal_cutoff = cutoff_freq / nyq
    
    # フィルタの設計
    b, a = signal.butter(order, normal_cutoff, btype=filter_type, analog=False)
    
    # フィルタの適用
    filtered_data = signal.filtfilt(b, a, data)
    
    return filtered_data


def extract_envelope(data: np.ndarray, smooth: bool = False, window_size: int = 10) -> np.ndarray:
    """
    ヒルベルト変換を用いて包絡線を抽出する。

    Args:
        data: 入力データ
        smooth: 平滑化するかどうか
        window_size: 平滑化の窓サイズ

    Returns:
        包絡線データ
    """
    # ヒルベルト変換による解析信号の計算
    analytic_signal = signal.hilbert(data)
    
    # 包絡線（振幅）の計算
    envelope = np.abs(analytic_signal)
    
    # 平滑化（オプション）
    if smooth and window_size > 1:
        envelope = np.convolve(envelope, np.ones(window_size)/window_size, mode='same')
    
    return envelope


def extract_features(data: np.ndarray, time_data: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    波形から特徴量を抽出する。

    Args:
        data: 入力データ
        time_data: 時間軸データ（オプション）

    Returns:
        特徴量の辞書
    """
    features = {}
    
    # 基本統計量
    features['max'] = float(np.max(data))
    features['min'] = float(np.min(data))
    features['peak_to_peak'] = float(features['max'] - features['min'])
    features['mean'] = float(np.mean(data))
    features['median'] = float(np.median(data))
    features['std'] = float(np.std(data))
    features['rms'] = float(np.sqrt(np.mean(np.square(data))))
    
    # ゼロクロス数
    zero_crossings = np.where(np.diff(np.signbit(data)))[0]
    features['zero_crossing_count'] = int(len(zero_crossings))
    
    # ピーク検出
    peaks, _ = signal.find_peaks(data, height=0.5*features['peak_to_peak'])
    features['peak_count'] = int(len(peaks))
    
    # 時間軸データがある場合の特徴量
    if time_data is not None and len(time_data) == len(data):
        # 信号の継続時間
        features['duration'] = float(time_data[-1] - time_data[0])
        
        # ピーク間の平均時間
        if len(peaks) > 1:
            peak_times = time_data[peaks]
            peak_intervals = np.diff(peak_times)
            features['mean_peak_interval'] = float(np.mean(peak_intervals))
        else:
            features['mean_peak_interval'] = 0.0
    
    return features


def slice_data(data: np.ndarray, time_data: np.ndarray, start_time: float, end_time: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    指定した時間範囲でデータを切り出す。

    Args:
        data: 入力データ
        time_data: 時間軸データ
        start_time: 開始時間
        end_time: 終了時間

    Returns:
        (切り出された時間軸データ, 切り出されたデータ)のタプル
    """
    # 時間範囲内のインデックスを取得
    mask = (time_data >= start_time) & (time_data <= end_time)
    
    # データの切り出し
    sliced_time = time_data[mask]
    sliced_data = data[mask]
    
    return sliced_time, sliced_data


def execute_custom_code(data: np.ndarray, code: str, input_var_name: str = 'data', output_var_name: str = 'result') -> np.ndarray:
    """
    カスタムPythonコードを実行する。

    Args:
        data: 入力データ
        code: 実行するPythonコード
        input_var_name: 入力データの変数名
        output_var_name: 出力データの変数名

    Returns:
        処理結果のデータ
    """
    # 実行環境の準備
    locals_dict = {
        input_var_name: data,
        'np': np,
        'signal': signal,
        'fft': fft,
        'ifft': ifft,
        'fftfreq': fftfreq
    }
    
    # コードの実行
    try:
        exec(code, globals(), locals_dict)
    except Exception as e:
        raise RuntimeError(f"カスタムコードの実行エラー: {str(e)}")
    
    # 結果の取得
    if output_var_name not in locals_dict:
        raise ValueError(f"出力変数 '{output_var_name}' が見つかりません")
    
    result = locals_dict[output_var_name]
    
    # 結果の検証
    if not isinstance(result, np.ndarray):
        raise ValueError(f"出力は numpy.ndarray である必要があります（現在の型: {type(result).__name__}）")
    
    return result