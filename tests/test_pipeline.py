#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
パイプラインのテストモジュール。

パイプラインの機能をテストします。
"""

import os
import json
import tempfile
import numpy as np
import pytest
from typing import Dict, Tuple, Any

from src.pipeline.pipeline import Pipeline
from src.nodes.base_node import Node
from src.nodes.data_load_node import DataLoadNode
from src.nodes.time_slice_node import TimeSliceNode
from src.nodes.fft_node import FFTNode
from src.nodes.filter_node import FrequencyFilterNode


# テスト用のダミーデータ
@pytest.fixture
def dummy_data():
    """テスト用のダミーデータを生成する。"""
    # 時間軸データ
    time = np.linspace(0, 1, 1000)
    
    # 正弦波 + ノイズ
    data = np.sin(2 * np.pi * 10 * time) + 0.5 * np.sin(2 * np.pi * 20 * time) + 0.1 * np.random.randn(len(time))
    
    return time, data


# パイプラインのセットアップ
@pytest.fixture
def setup_pipeline():
    """テスト用のパイプラインをセットアップする。"""
    # パイプラインのインスタンスを取得
    pipeline = Pipeline.get_instance()
    
    # 既存のノードをクリア
    pipeline.clear()
    
    # ノードタイプの登録
    pipeline.register_node_type(DataLoadNode)
    pipeline.register_node_type(TimeSliceNode)
    pipeline.register_node_type(FFTNode)
    pipeline.register_node_type(FrequencyFilterNode)
    
    return pipeline


# パイプラインの基本機能テスト
def test_pipeline_basic(setup_pipeline):
    """パイプラインの基本機能をテストする。"""
    pipeline = setup_pipeline
    
    # ノードの追加
    node1 = DataLoadNode("DataLoadNode", (100, 100))
    node2 = TimeSliceNode("TimeSliceNode", (300, 100))
    
    pipeline.add_node(node1)
    pipeline.add_node(node2)
    
    assert len(pipeline.nodes) == 2
    
    # ノードの削除
    pipeline.remove_node(node1)
    
    assert len(pipeline.nodes) == 1
    assert pipeline.nodes[0] == node2
    
    # パイプラインのクリア
    pipeline.clear()
    
    assert len(pipeline.nodes) == 0


# ノード接続のテスト
def test_pipeline_connections(setup_pipeline):
    """パイプラインのノード接続機能をテストする。"""
    pipeline = setup_pipeline
    
    # ノードの追加
    node1 = DataLoadNode("DataLoadNode", (100, 100))
    node2 = TimeSliceNode("TimeSliceNode", (300, 100))
    
    pipeline.add_node(node1)
    pipeline.add_node(node2)
    
    # 出力ピンと入力ピンのIDを取得
    from_pin_id = node1.output_pins["data"].id
    to_pin_id = node2.input_pins["data"].id
    
    # ピンの接続
    result = pipeline.connect_pins(from_pin_id, to_pin_id)
    
    assert result is True
    assert node1.output_pins["data"].connected_to == to_pin_id
    assert node2.input_pins["data"].connected_to == from_pin_id
    
    # ピンの切断
    result = pipeline.disconnect_pins(from_pin_id, to_pin_id)
    
    assert result is True
    assert node1.output_pins["data"].connected_to is None
    assert node2.input_pins["data"].connected_to is None


# 実行順序のテスト
def test_pipeline_execution_order(setup_pipeline):
    """パイプラインの実行順序をテストする。"""
    pipeline = setup_pipeline
    
    # ノードの追加
    node1 = DataLoadNode("DataLoadNode", (100, 100))
    node2 = TimeSliceNode("TimeSliceNode", (300, 100))
    node3 = FFTNode("FFTNode", (500, 100))
    
    pipeline.add_node(node1)
    pipeline.add_node(node2)
    pipeline.add_node(node3)
    
    # ノードの接続
    pipeline.connect_pins(node1.output_pins["data"].id, node2.input_pins["data"].id)
    pipeline.connect_pins(node1.output_pins["time"].id, node2.input_pins["time"].id)
    pipeline.connect_pins(node2.output_pins["data"].id, node3.input_pins["data"].id)
    pipeline.connect_pins(node2.output_pins["time"].id, node3.input_pins["time"].id)
    
    # 実行順序の取得
    execution_order = pipeline.get_execution_order()
    
    # 実行順序の検証
    assert len(execution_order) == 3
    assert execution_order[0] == node1  # DataLoadNodeが最初
    assert execution_order[1] == node2  # TimeSliceNodeが2番目
    assert execution_order[2] == node3  # FFTNodeが最後


# パイプラインの保存と読み込みのテスト
def test_pipeline_save_load(setup_pipeline, tmp_path):
    """パイプラインの保存と読み込み機能をテストする。"""
    pipeline = setup_pipeline
    
    # ノードの追加
    node1 = DataLoadNode("DataLoadNode", (100, 100))
    node2 = TimeSliceNode("TimeSliceNode", (300, 100))
    
    pipeline.add_node(node1)
    pipeline.add_node(node2)
    
    # パラメータの設定
    node1.set_parameter("file_path", "test.csv")
    node2.set_parameter("start_time", 0.1)
    node2.set_parameter("end_time", 0.9)
    
    # ノードの接続
    pipeline.connect_pins(node1.output_pins["data"].id, node2.input_pins["data"].id)
    pipeline.connect_pins(node1.output_pins["time"].id, node2.input_pins["time"].id)
    
    # 一時ファイルパスの作成
    file_path = tmp_path / "test_pipeline.json"
    
    # パイプラインの保存
    pipeline.save(str(file_path))
    
    # パイプラインのクリア
    pipeline.clear()
    assert len(pipeline.nodes) == 0
    
    # パイプラインの読み込み
    pipeline.load(str(file_path))
    
    # 読み込み結果の検証
    assert len(pipeline.nodes) == 2
    
    # ノードの種類の検証
    assert isinstance(pipeline.nodes[0], DataLoadNode) or isinstance(pipeline.nodes[1], DataLoadNode)
    assert isinstance(pipeline.nodes[0], TimeSliceNode) or isinstance(pipeline.nodes[1], TimeSliceNode)
    
    # パラメータの検証
    data_load_node = next(node for node in pipeline.nodes if isinstance(node, DataLoadNode))
    time_slice_node = next(node for node in pipeline.nodes if isinstance(node, TimeSliceNode))
    
    assert data_load_node.get_parameter("file_path") == "test.csv"
    assert time_slice_node.get_parameter("start_time") == 0.1
    assert time_slice_node.get_parameter("end_time") == 0.9
    
    # 接続の検証
    data_output_pin = data_load_node.output_pins["data"]
    time_output_pin = data_load_node.output_pins["time"]
    data_input_pin = time_slice_node.input_pins["data"]
    time_input_pin = time_slice_node.input_pins["time"]
    
    assert data_output_pin.connected_to is not None
    assert time_output_pin.connected_to is not None


# パイプラインのモジュールエクスポートのテスト
def test_pipeline_export_module(setup_pipeline, tmp_path):
    """パイプラインのモジュールエクスポート機能をテストする。"""
    pipeline = setup_pipeline
    
    # ノードの追加
    node1 = DataLoadNode("DataLoadNode", (100, 100))
    node2 = TimeSliceNode("TimeSliceNode", (300, 100))
    node3 = FFTNode("FFTNode", (500, 100))
    
    pipeline.add_node(node1)
    pipeline.add_node(node2)
    pipeline.add_node(node3)
    
    # ノードの接続
    pipeline.connect_pins(node1.output_pins["data"].id, node2.input_pins["data"].id)
    pipeline.connect_pins(node1.output_pins["time"].id, node2.input_pins["time"].id)
    pipeline.connect_pins(node2.output_pins["data"].id, node3.input_pins["data"].id)
    pipeline.connect_pins(node2.output_pins["time"].id, node3.input_pins["time"].id)
    
    # 一時ファイルパスの作成
    file_path = tmp_path / "test_pipeline_module.py"
    
    # モジュールのエクスポート
    pipeline.export_as_module(str(file_path))
    
    # ファイルの存在確認
    assert os.path.exists(file_path)
    
    # ファイルの内容確認
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 必要な関数が含まれているか確認
    assert "def process_pipeline(" in content
    assert "def process_0_dataloadnode(" in content or "def process_1_dataloadnode(" in content or "def process_2_dataloadnode(" in content
    assert "def process_0_timeslicenode(" in content or "def process_1_timeslicenode(" in content or "def process_2_timeslicenode(" in content
    assert "def process_0_fftnode(" in content or "def process_1_fftnode(" in content or "def process_2_fftnode(" in content


# 循環依存関係の検出テスト
def test_pipeline_circular_dependency(setup_pipeline):
    """パイプラインの循環依存関係検出機能をテストする。"""
    pipeline = setup_pipeline
    
    # ノードの追加
    node1 = DataLoadNode("DataLoadNode", (100, 100))
    node2 = TimeSliceNode("TimeSliceNode", (300, 100))
    
    pipeline.add_node(node1)
    pipeline.add_node(node2)
    
    # 正常な接続
    pipeline.connect_pins(node1.output_pins["data"].id, node2.input_pins["data"].id)
    pipeline.connect_pins(node1.output_pins["time"].id, node2.input_pins["time"].id)
    
    # 実行順序の取得（正常）
    execution_order = pipeline.get_execution_order()
    assert len(execution_order) == 2
    
    # 循環依存関係を作成
    pipeline.connect_pins(node2.output_pins["data"].id, node1.input_pins["data"].id)
    
    # 実行順序の取得（循環依存関係があるため例外が発生するはず）
    with pytest.raises(ValueError) as excinfo:
        pipeline.get_execution_order()
    
    assert "循環依存関係" in str(excinfo.value)