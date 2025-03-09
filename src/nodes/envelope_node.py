#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
包絡線抽出ノードモジュール。

ヒルベルト変換を用いて波形から包絡線を抽出するノードを実装します。
"""

import numpy as np
from typing import Dict, Tuple, Any, Optional

import dearpygui.dearpygui as dpg

from src.nodes.base_node import Node
from src.utils.data_utils import extract_envelope


class EnvelopeNode(Node):
    """包絡線抽出ノード。"""

    def __init__(self, title: str, pos: Tuple[int, int] = (0, 0)):
        """
        EnvelopeNodeクラスのコンストラクタ。

        Args:
            title: ノードのタイトル
            pos: ノードの初期位置 (x, y)
        """
        super().__init__(title, pos)
        
        # 入力ピンの追加
        self.add_input_pin("time", "array")
        self.add_input_pin("data", "array")
        
        # 出力ピンの追加
        self.add_output_pin("time", "array")
        self.add_output_pin("envelope", "array")
        
        # パラメータの追加
        self.add_parameter("smooth", True, "bool")
        self.add_parameter("window_size", 10, "int")

    def process(self) -> Dict[str, np.ndarray]:
        """
        包絡線抽出処理を実行する。

        Returns:
            出力ピン名をキーとする出力データの辞書
        """
        # キャッシュがあり、かつ再計算が不要な場合はキャッシュを返す
        if self.result_cache is not None and not self.is_dirty:
            return self.result_cache
        
        # 入力データの取得
        time_array = self.get_input_data("time")
        data_array = self.get_input_data("data")
        
        # 入力データがない場合は空の配列を返す
        if time_array is None or data_array is None or len(time_array) == 0 or len(data_array) == 0:
            self.result_cache = {
                "time": np.array([]),
                "envelope": np.array([])
            }
            self.is_dirty = False
            return self.result_cache
        
        # パラメータの取得
        smooth = self.get_parameter("smooth")
        window_size = self.get_parameter("window_size")
        
        try:
            # 包絡線の抽出
            envelope = extract_envelope(data_array, smooth, window_size)
            
            # 結果をキャッシュし、ダーティフラグをクリア
            self.result_cache = {
                "time": time_array,
                "envelope": envelope
            }
            self.is_dirty = False
            
            return self.result_cache
        
        except Exception as e:
            # エラーが発生した場合は空の配列を返す
            print(f"包絡線抽出エラー: {str(e)}")
            self.result_cache = {
                "time": time_array,
                "envelope": np.array([])
            }
            self.is_dirty = False
            return self.result_cache

    def update_ui(self) -> None:
        """ノードのUI（特にグラフ）を更新する。"""
        # 基本UIの更新
        super().update_ui()
        
        # グラフが表示されていない場合は更新しない
        if not self.show_graph or self.ui_graph_id is None:
            return
        
        # 処理結果を取得
        outputs = self.process()
        if not outputs:
            return
        
        # 時間と包絡線データの取得
        time_array = outputs.get("time")
        envelope = outputs.get("envelope")
        
        if time_array is None or envelope is None or len(time_array) == 0 or len(envelope) == 0:
            return
        
        # 入力データの取得（比較表示用）
        input_data = self.get_input_data("data")
        
        # 既存のグラフをクリア
        dpg.delete_item(self.ui_graph_id, children_only=True)
        
        # 包絡線グラフの作成
        with dpg.plot(label="包絡線", height=200, width=-1, parent=self.ui_graph_id):
            # 軸の設定
            dpg.add_plot_legend()
            dpg.add_plot_axis(dpg.mvXAxis, label="時間")
            
            with dpg.plot_axis(dpg.mvYAxis, label="振幅"):
                # 元の波形データの表示（存在する場合）
                if input_data is not None and len(input_data) > 0:
                    # データ点数が多い場合は間引く
                    if len(time_array) > 1000:
                        step = len(time_array) // 1000
                        plot_time = time_array[::step]
                        plot_data = input_data[::step]
                    else:
                        plot_time = time_array
                        plot_data = input_data
                    
                    dpg.add_line_series(
                        plot_time.tolist(),
                        plot_data.tolist(),
                        label="元の波形"
                    )
                
                # 包絡線データの表示
                if len(time_array) > 1000:
                    step = len(time_array) // 1000
                    plot_time = time_array[::step]
                    plot_envelope = envelope[::step]
                else:
                    plot_time = time_array
                    plot_envelope = envelope
                
                dpg.add_line_series(
                    plot_time.tolist(),
                    plot_envelope.tolist(),
                    label="包絡線"
                )