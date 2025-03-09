#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ノードのテストモジュール。

各ノードの基本機能をテストします。
"""

import os
import numpy as np
import pytest
from typing import Dict, Tuple, Any

from src.nodes.base_node import Node
from src.nodes.data_load_node import DataLoadNode
from src.nodes.time_slice_node import TimeSliceNode
from src.nodes.fft_node import FFTNode
from src.nodes.filter_node import FrequencyFilterNode
from src.nodes.feature_node import FeatureExtractionNode
from src.nodes.envelope_node import EnvelopeNode
from src.nodes.custom_python_node import CustomPythonNode
from src.nodes.display_node import WaveformDisplayNode
from src.utils.data_utils import compute_fft, apply_filter, extract_envelope, extract_features


# テスト用のダミーデータ
@pytest.fixture
def dummy_data():
    """テスト用のダミーデータを生成する。"""
    # 時間軸データ
    time = np.linspace(0, 1, 1000)
    
    # 正弦波 + ノイズ
    data = np.sin(2 * np.pi * 10 * time) + 0.5 * np.sin(2 * np.pi * 20 * time) + 0.1 * np.random.randn(len(time))
    
    return time, data


# 基本ノードのテスト
def test_base_node():
    """基本ノードの機能をテストする。"""
    # ノードの作成
    node = Node("TestNode", (100, 100))
    
    # 入出力ピンの追加
    in_pin = node.add_input_pin("input", "float")
    out_pin = node.add_output_pin("output", "float")
    
    # パラメータの追加と取得
    node.add_parameter("param1", 10.0, "float")
    node.add_parameter("param2", "test", "text")
    
    assert node.get_parameter("param1") == 10.0
    assert node.get_parameter("param2") == "test"
    
    # パラメータの設定
    node.set_parameter("param1", 20.0)
    assert node.get_parameter("param1") == 20.0
    
    # シリアライズとデシリアライズ
    data = node.serialize()
    new_node = Node.deserialize(data)
    
    assert new_node.title == "TestNode"
    assert new_node.get_parameter("param1") == 20.0
    assert new_node.get_parameter("param2") == "test"
    assert "input" in new_node.input_pins
    assert "output" in new_node.output_pins


# データ読み込みノードのテスト
def test_data_load_node(tmp_path):
    """データ読み込みノードの機能をテストする。"""
    # テスト用のCSVファイルを作成
    csv_path = tmp_path / "test_data.csv"
    time = np.linspace(0, 1, 100)
    data = np.sin(2 * np.pi * 10 * time)
    
    # CSVファイルに書き込み
    import pandas as pd
    df = pd.DataFrame({"time": time, "data": data})
    df.to_csv(csv_path, index=False)
    
    # ノードの作成
    node = DataLoadNode("DataLoadNode", (100, 100))
    
    # パラメータの設定
    node.set_parameter("file_path", str(csv_path))
    node.set_parameter("file_type", (["Auto", "TRC", "CSV"], 0))  # Auto
    
    # 処理の実行
    result = node.process()
    
    # 結果の検証
    assert "time" in result
    assert "data" in result
    assert len(result["time"]) == 100
    assert len(result["data"]) == 100
    assert np.allclose(result["time"], time)
    assert np.allclose(result["data"], data)


# 時間指定切り出しノードのテスト
def test_time_slice_node(dummy_data):
    """時間指定切り出しノードの機能をテストする。"""
    time, data = dummy_data
    
    # ノードの作成
    node = TimeSliceNode("TimeSliceNode", (100, 100))
    
    # 入力データの設定（モック）
    node.get_input_data = lambda pin_name: time if pin_name == "time" else data
    
    # パラメータの設定
    node.set_parameter("start_time", 0.2)
    node.set_parameter("end_time", 0.8)
    node.set_parameter("auto_range", False)
    
    # 処理の実行
    result = node.process()
    
    # 結果の検証
    assert "time" in result
    assert "data" in result
    
    # 切り出し範囲の確認
    assert result["time"][0] >= 0.2
    assert result["time"][-1] <= 0.8
    assert len(result["time"]) < len(time)


# FFTノードのテスト
def test_fft_node(dummy_data):
    """FFTノードの機能をテストする。"""
    time, data = dummy_data
    
    # ノードの作成
    node = FFTNode("FFTNode", (100, 100))
    
    # 入力データの設定（モック）
    node.get_input_data = lambda pin_name: time if pin_name == "time" else data
    
    # パラメータの設定
    node.set_parameter("window_type", (["hann", "hamming", "blackman", "rectangular"], 0))
    node.set_parameter("sample_rate", 1000.0)
    node.set_parameter("auto_sample_rate", False)
    
    # 処理の実行
    result = node.process()
    
    # 結果の検証
    assert "freq" in result
    assert "amplitude" in result
    assert len(result["freq"]) > 0
    assert len(result["amplitude"]) > 0
    
    # ユーティリティ関数との比較
    freq, amplitude = compute_fft(data, 1000.0, "hann")
    assert np.allclose(result["freq"], freq)
    assert np.allclose(result["amplitude"], amplitude)


# 周波数フィルターノードのテスト
def test_filter_node(dummy_data):
    """周波数フィルターノードの機能をテストする。"""
    time, data = dummy_data
    
    # ノードの作成
    node = FrequencyFilterNode("FilterNode", (100, 100))
    
    # 入力データの設定（モック）
    node.get_input_data = lambda pin_name: time if pin_name == "time" else data
    
    # パラメータの設定（ローパスフィルタ）
    node.set_parameter("filter_type", (["lowpass", "highpass", "bandpass", "bandstop"], 0))
    node.set_parameter("cutoff_freq", 15.0)
    node.set_parameter("order", 4)
    node.set_parameter("sample_rate", 1000.0)
    node.set_parameter("auto_sample_rate", False)
    
    # 処理の実行
    result = node.process()
    
    # 結果の検証
    assert "time" in result
    assert "data" in result
    assert len(result["time"]) == len(time)
    assert len(result["data"]) == len(data)
    
    # ユーティリティ関数との比較
    filtered_data = apply_filter(data, "lowpass", 15.0, 4, 1000.0)
    assert np.allclose(result["data"], filtered_data)


# 特徴量抽出ノードのテスト
def test_feature_extraction_node(dummy_data):
    """特徴量抽出ノードの機能をテストする。"""
    time, data = dummy_data
    
    # ノードの作成
    node = FeatureExtractionNode("FeatureNode", (100, 100))
    
    # 入力データの設定（モック）
    node.get_input_data = lambda pin_name: time if pin_name == "time" else data
    
    # パラメータの設定
    node.set_parameter("extract_basic", True)
    node.set_parameter("extract_peaks", True)
    node.set_parameter("extract_time", True)
    
    # 処理の実行
    result = node.process()
    
    # 結果の検証
    assert "features" in result
    features = result["features"]
    
    # 基本統計量の確認
    assert "max" in features
    assert "min" in features
    assert "mean" in features
    assert "std" in features
    assert "rms" in features
    
    # ユーティリティ関数との比較
    expected_features = extract_features(data, time)
    for key in ["max", "min", "mean", "std", "rms"]:
        assert np.isclose(features[key], expected_features[key])


# 包絡線抽出ノードのテスト
def test_envelope_node(dummy_data):
    """包絡線抽出ノードの機能をテストする。"""
    time, data = dummy_data
    
    # ノードの作成
    node = EnvelopeNode("EnvelopeNode", (100, 100))
    
    # 入力データの設定（モック）
    node.get_input_data = lambda pin_name: time if pin_name == "time" else data
    
    # パラメータの設定
    node.set_parameter("smooth", True)
    node.set_parameter("window_size", 10)
    
    # 処理の実行
    result = node.process()
    
    # 結果の検証
    assert "time" in result
    assert "envelope" in result
    assert len(result["time"]) == len(time)
    assert len(result["envelope"]) == len(data)
    
    # ユーティリティ関数との比較
    expected_envelope = extract_envelope(data, True, 10)
    assert np.allclose(result["envelope"], expected_envelope)


# カスタムPythonノードのテスト
def test_custom_python_node(dummy_data):
    """カスタムPythonノードの機能をテストする。"""
    time, data = dummy_data
    
    # ノードの作成
    node = CustomPythonNode("CustomPythonNode", (100, 100))
    
    # 入力データの設定（モック）
    node.get_input_data = lambda pin_name: time if pin_name == "time" else data
    
    # パラメータの設定
    test_code = """
# 移動平均フィルタ
window_size = 5
result = np.convolve(data, np.ones(window_size)/window_size, mode='same')
"""
    node.set_parameter("code", test_code)
    node.set_parameter("input_var", "data")
    node.set_parameter("output_var", "result")
    node.set_parameter("process_time", False)
    
    # 処理の実行
    result = node.process()
    
    # 結果の検証
    assert "time" in result
    assert "result" in result
    assert len(result["time"]) == len(time)
    assert len(result["result"]) == len(data)
    
    # 期待される結果との比較
    window_size = 5
    expected_result = np.convolve(data, np.ones(window_size)/window_size, mode='same')
    assert np.allclose(result["result"], expected_result)


# 波形表示ノードのテスト
def test_display_node(dummy_data):
    """波形表示ノードの機能をテストする。"""
    time, data = dummy_data
    
    # ノードの作成
    node = WaveformDisplayNode("DisplayNode", (100, 100))
    
    # 入力データの設定（モック）
    node.get_input_data = lambda pin_name: time if pin_name == "time" else data
    
    # パラメータの設定
    node.set_parameter("line_color", (0, 255, 0, 255))
    node.set_parameter("line_thickness", 1.0)
    node.set_parameter("marker_type", (["None", "Circle", "Square", "Diamond", "Up", "Down", "Left", "Right"], 0))
    node.set_parameter("show_grid", True)
    node.set_parameter("auto_fit", True)
    
    # 処理の実行
    result = node.process()
    
    # 結果の検証（入力をそのまま出力するはず）
    assert "time" in result
    assert "data" in result
    assert np.array_equal(result["time"], time)
    assert np.array_equal(result["data"], data)