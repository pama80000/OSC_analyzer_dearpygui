#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
オシロスコープデータ処理フローのメインモジュール。

アプリケーションのエントリポイントとなるモジュールです。
"""

import os
import sys
from typing import Dict, List, Any, Optional, Type

import dearpygui.dearpygui as dpg

from src.node_editor.node_editor import NodeEditor
from src.pipeline.pipeline import Pipeline
from src.nodes.base_node import Node
from src.nodes.data_load_node import DataLoadNode
from src.nodes.time_slice_node import TimeSliceNode
from src.nodes.fft_node import FFTNode
from src.nodes.filter_node import FrequencyFilterNode
from src.nodes.feature_node import FeatureExtractionNode
from src.nodes.envelope_node import EnvelopeNode
from src.nodes.custom_python_node import CustomPythonNode
from src.nodes.display_node import WaveformDisplayNode


def register_node_types(node_editor: NodeEditor, pipeline: Pipeline) -> None:
    """
    ノードタイプを登録する。

    Args:
        node_editor: ノードエディタ
        pipeline: パイプライン
    """
    # 各ノードタイプの登録
    node_types = [
        ("データ読み込み", DataLoadNode),
        ("時間指定切り出し", TimeSliceNode),
        ("FFT", FFTNode),
        ("周波数フィルター", FrequencyFilterNode),
        ("特徴量抽出", FeatureExtractionNode),
        ("包絡線抽出", EnvelopeNode),
        ("カスタムPython", CustomPythonNode),
        ("波形表示", WaveformDisplayNode)
    ]
    
    for name, node_class in node_types:
        node_editor.register_node_type(node_class, name)
        pipeline.register_node_type(node_class)


def main():
    """アプリケーションのメインエントリポイント。"""
    # パイプラインの取得
    pipeline = Pipeline.get_instance()
    
    # ノードエディタの作成
    node_editor = NodeEditor(width=1280, height=800)
    
    # ノードタイプの登録
    register_node_types(node_editor, pipeline)
    
    # ノードエディタのUI作成
    node_editor.create_ui()
    
    # アプリケーションの実行
    node_editor.run()


if __name__ == "__main__":
    main()
