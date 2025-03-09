#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
時間指定切り出しノードモジュール。

指定した時間範囲のデータを切り出すノードを実装します。
"""

import numpy as np
from typing import Dict, Tuple, Any, Optional

import dearpygui.dearpygui as dpg

from src.nodes.base_node import Node
from src.utils.data_utils import slice_data


class TimeSliceNode(Node):
    """時間指定切り出しノード。"""

    def __init__(self, title: str, pos: Tuple[int, int] = (0, 0)):
        """
        TimeSliceNodeクラスのコンストラクタ。

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
        self.add_output_pin("data", "array")
        
        # パラメータの追加
        self.add_parameter("start_time", 0.0, "float")
        self.add_parameter("end_time", 1.0, "float")
        self.add_parameter("auto_range", True, "bool")
        
        # 時間範囲の自動設定フラグ
        self.time_range_set = False

    def process(self) -> Dict[str, np.ndarray]:
        """
        時間指定切り出し処理を実行する。

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
                "data": np.array([])
            }
            self.is_dirty = False
            return self.result_cache
        
        # 時間範囲の自動設定
        auto_range = self.get_parameter("auto_range")
        if auto_range and not self.time_range_set:
            self.set_parameter("start_time", float(time_array[0]))
            self.set_parameter("end_time", float(time_array[-1]))
            self.time_range_set = True
        
        # 時間範囲の取得
        start_time = self.get_parameter("start_time")
        end_time = self.get_parameter("end_time")
        
        # 範囲の調整
        start_time = max(start_time, time_array[0])
        end_time = min(end_time, time_array[-1])
        
        try:
            # データの切り出し
            sliced_time, sliced_data = slice_data(data_array, time_array, start_time, end_time)
            
            # 結果をキャッシュし、ダーティフラグをクリア
            self.result_cache = {
                "time": sliced_time,
                "data": sliced_data
            }
            self.is_dirty = False
            
            return self.result_cache
        
        except Exception as e:
            # エラーが発生した場合は空の配列を返す
            print(f"時間指定切り出しエラー: {str(e)}")
            self.result_cache = {
                "time": np.array([]),
                "data": np.array([])
            }
            self.is_dirty = False
            return self.result_cache

    def update_ui(self) -> None:
        """ノードのUI（特にグラフ）を更新する。"""
        # 基本UIの更新
        super().update_ui()
        
        # 入力データの取得
        time_array = self.get_input_data("time")
        
        # 入力データがある場合、時間範囲の最小値と最大値を更新
        if time_array is not None and len(time_array) > 0:
            # パラメータUIの取得
            start_time_ui_id = self.ui_param_ids.get("start_time")
            end_time_ui_id = self.ui_param_ids.get("end_time")
            
            if start_time_ui_id is not None and end_time_ui_id is not None:
                # 時間範囲の最小値と最大値を設定
                min_time = float(time_array[0])
                max_time = float(time_array[-1])
                
                # UIの更新
                dpg.set_item_callback(
                    start_time_ui_id,
                    lambda sender, data: self.set_parameter("start_time", max(min_time, min(data, max_time)))
                )
                dpg.set_item_callback(
                    end_time_ui_id,
                    lambda sender, data: self.set_parameter("end_time", max(min_time, min(data, max_time)))
                )