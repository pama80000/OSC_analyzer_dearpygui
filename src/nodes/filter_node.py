#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
周波数フィルターノードモジュール。

特定の周波数帯域をフィルタリングするノードを実装します。
"""

import numpy as np
from typing import Dict, Tuple, Any, Optional, Union

import dearpygui.dearpygui as dpg

from src.nodes.base_node import Node
from src.utils.data_utils import apply_filter


class FrequencyFilterNode(Node):
    """周波数フィルターノード。"""

    def __init__(self, title: str, pos: Tuple[int, int] = (0, 0)):
        """
        FrequencyFilterNodeクラスのコンストラクタ。

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
        self.add_parameter("filter_type", (["lowpass", "highpass", "bandpass", "bandstop"], 0), "combo")
        self.add_parameter("cutoff_freq", 100.0, "float")
        self.add_parameter("cutoff_freq_high", 1000.0, "float")
        self.add_parameter("order", 4, "int")
        self.add_parameter("sample_rate", 1000.0, "float")
        self.add_parameter("auto_sample_rate", True, "bool")
        
        # サンプリングレートの自動設定フラグ
        self.sample_rate_set = False

    def process(self) -> Dict[str, np.ndarray]:
        """
        フィルタリング処理を実行する。

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
        
        # サンプリングレートの自動設定
        auto_sample_rate = self.get_parameter("auto_sample_rate")
        if auto_sample_rate and not self.sample_rate_set and len(time_array) > 1:
            # 時間間隔からサンプリングレートを計算
            dt = (time_array[-1] - time_array[0]) / (len(time_array) - 1)
            sample_rate = 1.0 / dt if dt > 0 else 1000.0
            self.set_parameter("sample_rate", float(sample_rate))
            self.sample_rate_set = True
        
        # パラメータの取得
        filter_options, filter_idx = self.get_parameter("filter_type")
        filter_type = filter_options[filter_idx]
        cutoff_freq = self.get_parameter("cutoff_freq")
        cutoff_freq_high = self.get_parameter("cutoff_freq_high")
        order = self.get_parameter("order")
        sample_rate = self.get_parameter("sample_rate")
        
        try:
            # フィルタータイプに応じたカットオフ周波数の設定
            if filter_type in ["bandpass", "bandstop"]:
                cutoff = (cutoff_freq, cutoff_freq_high)
            else:
                cutoff = cutoff_freq
            
            # フィルタの適用
            filtered_data = apply_filter(data_array, filter_type, cutoff, order, sample_rate)
            
            # 結果をキャッシュし、ダーティフラグをクリア
            self.result_cache = {
                "time": time_array,
                "data": filtered_data
            }
            self.is_dirty = False
            
            return self.result_cache
        
        except Exception as e:
            # エラーが発生した場合は入力データをそのまま返す
            print(f"フィルタリングエラー: {str(e)}")
            self.result_cache = {
                "time": time_array,
                "data": data_array
            }
            self.is_dirty = False
            return self.result_cache

    def create_ui(self, parent_id: int) -> None:
        """
        ノードのUIを作成する。

        Args:
            parent_id: 親ウィジェットのID
        """
        # 基本UIの作成
        super().create_ui(parent_id)
        
        # フィルタータイプの変更時のコールバックを設定
        filter_type_ui_id = self.ui_param_ids.get("filter_type")
        if filter_type_ui_id:
            dpg.set_item_callback(filter_type_ui_id, self._on_filter_type_changed)
        
        # 初期状態でのUIの更新
        self._update_cutoff_ui()

    def _on_filter_type_changed(self, sender: int, app_data: str) -> None:
        """
        フィルタータイプが変更されたときのコールバック。

        Args:
            sender: 送信元のID
            app_data: 選択されたフィルタータイプ
        """
        # フィルタータイプの更新
        filter_options, _ = self.get_parameter("filter_type")
        self.set_parameter("filter_type", (filter_options, filter_options.index(app_data)))
        
        # UIの更新
        self._update_cutoff_ui()

    def _update_cutoff_ui(self) -> None:
        """フィルタータイプに応じてカットオフ周波数のUIを更新する。"""
        # フィルタータイプの取得
        filter_options, filter_idx = self.get_parameter("filter_type")
        filter_type = filter_options[filter_idx]
        
        # カットオフ周波数のUI IDの取得
        cutoff_freq_ui_id = self.ui_param_ids.get("cutoff_freq")
        cutoff_freq_high_ui_id = self.ui_param_ids.get("cutoff_freq_high")
        
        if cutoff_freq_ui_id and cutoff_freq_high_ui_id:
            # バンドパスとバンドストップの場合は両方のカットオフ周波数を表示
            if filter_type in ["bandpass", "bandstop"]:
                dpg.show_item(cutoff_freq_ui_id)
                dpg.show_item(cutoff_freq_high_ui_id)
                dpg.set_item_label(cutoff_freq_ui_id, "低域カットオフ周波数 (Hz)")
                dpg.set_item_label(cutoff_freq_high_ui_id, "高域カットオフ周波数 (Hz)")
            else:
                # それ以外の場合は1つのカットオフ周波数のみを表示
                dpg.show_item(cutoff_freq_ui_id)
                dpg.hide_item(cutoff_freq_high_ui_id)
                dpg.set_item_label(cutoff_freq_ui_id, "カットオフ周波数 (Hz)")