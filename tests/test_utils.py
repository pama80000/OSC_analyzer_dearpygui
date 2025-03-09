#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ユーティリティ関数のテストモジュール。

データ処理ユーティリティ関数をテストします。
"""

import os
import numpy as np
import pytest
from typing import Dict, Tuple, Any

from src.utils.data_utils import (
    apply_window,
    compute_fft,
    apply_filter,
    extract_envelope,
    extract_features,
    slice_data,
    execute_custom_code
)


# テスト用のダミーデータ
@pytest.fixture
def dummy_data():
    """テスト用のダミーデータを生成する。"""
    # 時間軸データ
    time = np.linspace(0, 1, 1000)
    
    # 正弦波 + ノイズ
    data = np.sin(2 * np.pi * 10 * time) + 0.5 * np.sin(2 * np.pi * 20 * time) + 0.1 * np.random.randn(len(time))
    
    return time, data


# 窓関数適用のテスト
def test_apply_window(dummy_data):
    """窓関数適用機能をテストする。"""
    _, data = dummy_data
    
    # 各種窓関数の適用
    hann_data = apply_window(data, "hann")
    hamming_data = apply_window(data, "hamming")
    blackman_data = apply_window(data, "blackman")
    rectangular_data = apply_window(data, "rectangular")
    
    # 結果の検証
    assert len(hann_data) == len(data)
    assert len(hamming_data) == len(data)
    assert len(blackman_data) == len(data)
    assert len(rectangular_data) == len(data)
    
    # 窓関数の効果を検証
    assert np.array_equal(rectangular_data, data)  # 矩形窓はデータを変更しない
    assert not np.array_equal(hann_data, data)  # その他の窓関数はデータを変更する
    
    # 未対応の窓関数タイプ
    with pytest.raises(ValueError):
        apply_window(data, "unknown_window")


# FFT計算のテスト
def test_compute_fft(dummy_data):
    """FFT計算機能をテストする。"""
    time, data = dummy_data
    
    # FFTの計算
    freq, amplitude = compute_fft(data, 1000.0, "hann")
    
    # 結果の検証
    assert len(freq) > 0
    assert len(amplitude) > 0
    assert len(freq) == len(amplitude)
    
    # 周波数範囲の検証
    assert freq[0] > 0  # 正の周波数のみ
    assert freq[-1] < 500.0  # ナイキスト周波数以下
    
    # 異なる窓関数での計算
    freq2, amplitude2 = compute_fft(data, 1000.0, "rectangular")
    
    # 窓関数の違いによる結果の違いを検証
    assert not np.array_equal(amplitude, amplitude2)


# フィルタ適用のテスト
def test_apply_filter(dummy_data):
    """フィルタ適用機能をテストする。"""
    _, data = dummy_data
    
    # 各種フィルタの適用
    lowpass_data = apply_filter(data, "lowpass", 20.0, 4, 1000.0)
    highpass_data = apply_filter(data, "highpass", 5.0, 4, 1000.0)
    bandpass_data = apply_filter(data, "bandpass", (5.0, 20.0), 4, 1000.0)
    bandstop_data = apply_filter(data, "bandstop", (5.0, 20.0), 4, 1000.0)
    
    # 結果の検証
    assert len(lowpass_data) == len(data)
    assert len(highpass_data) == len(data)
    assert len(bandpass_data) == len(data)
    assert len(bandstop_data) == len(data)
    
    # フィルタの効果を検証（詳細な検証は難しいので、元のデータと異なることを確認）
    assert not np.array_equal(lowpass_data, data)
    assert not np.array_equal(highpass_data, data)
    assert not np.array_equal(bandpass_data, data)
    assert not np.array_equal(bandstop_data, data)
    
    # エラーケース
    with pytest.raises(ValueError):
        # bandpassフィルタにタプルでないカットオフ周波数を指定
        apply_filter(data, "bandpass", 10.0, 4, 1000.0)
    
    with pytest.raises(ValueError):
        # lowpassフィルタにタプルのカットオフ周波数を指定
        apply_filter(data, "lowpass", (5.0, 20.0), 4, 1000.0)
    
    with pytest.raises(ValueError):
        # 低域カットオフが高域カットオフより大きい
        apply_filter(data, "bandpass", (20.0, 5.0), 4, 1000.0)


# 包絡線抽出のテスト
def test_extract_envelope(dummy_data):
    """包絡線抽出機能をテストする。"""
    _, data = dummy_data
    
    # 包絡線の抽出（平滑化なし）
    envelope1 = extract_envelope(data, False)
    
    # 包絡線の抽出（平滑化あり）
    envelope2 = extract_envelope(data, True, 10)
    
    # 結果の検証
    assert len(envelope1) == len(data)
    assert len(envelope2) == len(data)
    
    # 包絡線の性質を検証
    assert np.all(envelope1 >= 0)  # 包絡線は非負
    assert np.all(envelope2 >= 0)
    
    # 平滑化の効果を検証
    assert not np.array_equal(envelope1, envelope2)
    
    # 包絡線は元の信号の振幅以上にならないはず
    max_data_abs = np.max(np.abs(data))
    assert np.all(envelope1 <= max_data_abs * 1.1)  # 少し余裕を持たせる


# 特徴量抽出のテスト
def test_extract_features(dummy_data):
    """特徴量抽出機能をテストする。"""
    time, data = dummy_data
    
    # 特徴量の抽出（時間データなし）
    features1 = extract_features(data)
    
    # 特徴量の抽出（時間データあり）
    features2 = extract_features(data, time)
    
    # 結果の検証
    assert isinstance(features1, dict)
    assert isinstance(features2, dict)
    
    # 基本統計量の検証
    assert "max" in features1
    assert "min" in features1
    assert "mean" in features1
    assert "std" in features1
    assert "rms" in features1
    
    # 時間関連の特徴量の検証
    assert "duration" not in features1  # 時間データなしの場合
    assert "duration" in features2  # 時間データありの場合
    
    # 特徴量の値の検証
    assert features1["max"] == np.max(data)
    assert features1["min"] == np.min(data)
    assert np.isclose(features1["mean"], np.mean(data))
    assert np.isclose(features1["std"], np.std(data))
    assert np.isclose(features1["rms"], np.sqrt(np.mean(np.square(data))))
    
    # 時間関連の特徴量の値の検証
    if "duration" in features2:
        assert np.isclose(features2["duration"], time[-1] - time[0])


# データ切り出しのテスト
def test_slice_data(dummy_data):
    """データ切り出し機能をテストする。"""
    time, data = dummy_data
    
    # データの切り出し
    start_time = 0.2
    end_time = 0.8
    sliced_time, sliced_data = slice_data(data, time, start_time, end_time)
    
    # 結果の検証
    assert len(sliced_time) == len(sliced_data)
    assert len(sliced_time) < len(time)
    
    # 切り出し範囲の検証
    assert sliced_time[0] >= start_time
    assert sliced_time[-1] <= end_time
    
    # 切り出されたデータの検証
    start_idx = np.argmax(time >= start_time)
    end_idx = np.argmax(time > end_time) if np.any(time > end_time) else len(time)
    expected_length = end_idx - start_idx
    assert len(sliced_time) == expected_length


# カスタムコード実行のテスト
def test_execute_custom_code(dummy_data):
    """カスタムコード実行機能をテストする。"""
    _, data = dummy_data
    
    # カスタムコードの実行
    code = """
# 移動平均フィルタ
window_size = 5
result = np.convolve(data, np.ones(window_size)/window_size, mode='same')
"""
    result = execute_custom_code(data, code, "data", "result")
    
    # 結果の検証
    assert isinstance(result, np.ndarray)
    assert len(result) == len(data)
    
    # 期待される結果との比較
    window_size = 5
    expected_result = np.convolve(data, np.ones(window_size)/window_size, mode='same')
    assert np.allclose(result, expected_result)
    
    # エラーケース
    with pytest.raises(RuntimeError):
        # 構文エラーのあるコード
        execute_custom_code(data, "result = data + ", "data", "result")
    
    with pytest.raises(ValueError):
        # 出力変数が存在しないコード
        execute_custom_code(data, "x = data + 1", "data", "result")
    
    with pytest.raises(ValueError):
        # 出力が配列でないコード
        execute_custom_code(data, "result = 42", "data", "result")