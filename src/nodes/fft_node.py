#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FFTノードモジュール。

時間領域データを周波数領域に変換するノードを実装します。
"""

import numpy as np
from typing import Dict, Tuple, Any, Optional

import dearpygui.dearpygui as dpg

from src.nodes.base_node import Node
from src.utils.data_utils import compute_fft


class FFTNode(Node):
    """FFTノード。"""

    def __init__(self, title: str, pos: Tuple[int, int] = (0, 0)):
        """
        FFTNodeクラスのコンストラクタ。

        Args:
            title: ノードのタイトル
            pos: ノードの初期位置 (x, y)
        """
        super().__init__(title, pos)
        
        # 入力ピンの追加
        self.add_input_pin("time", "array")
        self.add_input_pin("data", "array")
        
        # 出力ピンの追加
        self.add_output_pin("freq", "array")
        self.add_output_pin("amplitude", "array")
        
        # パラメータの追加
        self.add_parameter("window_type", (["hann", "hamming", "blackman", "rectangular"], 0), "combo")
        self.add_parameter("sample_rate", 1.0, "float")
        self.add_parameter("auto_sample_rate", True, "bool")
        
        # サンプリングレートの自動設定フラグ
        self.sample_rate_set = False

    def process(self) -> Dict[str, np.ndarray]:
        """
        FFT処理を実行する。

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
                "freq": np.array([]),
                "amplitude": np.array([])
            }
            self.is_dirty = False
            return self.result_cache
        
        # サンプリングレートの自動設定
        auto_sample_rate = self.get_parameter("auto_sample_rate")
        if auto_sample_rate and not self.sample_rate_set and len(time_array) > 1:
            # 時間間隔からサンプリングレートを計算
            dt = (time_array[-1] - time_array[0]) / (len(time_array) - 1)
            sample_rate = 1.0 / dt if dt > 0 else 1.0
            self.set_parameter("sample_rate", float(sample_rate))
            self.sample_rate_set = True
        
        # パラメータの取得
        window_options, window_idx = self.get_parameter("window_type")
        window_type = window_options[window_idx]
        sample_rate = self.get_parameter("sample_rate")
        
        try:
            # FFTの計算
            freq, amplitude = compute_fft(data_array, sample_rate, window_type)
            
            # 結果をキャッシュし、ダーティフラグをクリア
            self.result_cache = {
                "freq": freq,
                "amplitude": amplitude
            }
            self.is_dirty = False
            
            return self.result_cache
        
        except Exception as e:
            # エラーが発生した場合は空の配列を返す
            print(f"FFT計算エラー: {str(e)}")
            self.result_cache = {
                "freq": np.array([]),
                "amplitude": np.array([])
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
        
        # 周波数と振幅データの取得
        freq = outputs.get("freq")
        amplitude = outputs.get("amplitude")
        
        if freq is None or amplitude is None or len(freq) == 0 or len(amplitude) == 0:
            return
        
        # 既存のグラフをクリア
        dpg.delete_item(self.ui_graph_id, children_only=True)
        
        # 対数スケールのチェックボックス
        with dpg.group(parent=self.ui_graph_id, horizontal=True):
            dpg.add_text("対数スケール:")
            log_scale_x = dpg.add_checkbox(label="X軸", default_value=True)
            log_scale_y = dpg.add_checkbox(label="Y軸", default_value=True)
        
        # FFTグラフの作成
        with dpg.plot(label="FFT", height=200, width=-1, parent=self.ui_graph_id):
            # 軸の設定
            dpg.add_plot_legend()
            
            # X軸（周波数）
            x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="周波数 (Hz)")
            if dpg.get_value(log_scale_x):
                dpg.set_axis_limits_log(x_axis)
            
            # Y軸（振幅）
            with dpg.plot_axis(dpg.mvYAxis, label="振幅") as y_axis:
                if dpg.get_value(log_scale_y):
                    dpg.set_axis_limits_log(y_axis)
                
                # データの追加（ゼロ以上の値のみをプロット）
                valid_indices = (freq > 0) & (amplitude > 0)
                if np.any(valid_indices):
                    valid_freq = freq[valid_indices]
                    valid_amplitude = amplitude[valid_indices]
                    
                    dpg.add_line_series(
                        valid_freq.tolist(),
                        valid_amplitude.tolist(),
                        label="FFT"
                    )
            
            # 対数スケール切り替えのコールバック
            def toggle_log_scale_x(sender, data):
                if data:
                    dpg.set_axis_limits_log(x_axis)
                else:
                    dpg.set_axis_limits_auto(x_axis)
            
            def toggle_log_scale_y(sender, data):
                if data:
                    dpg.set_axis_limits_log(y_axis)
                else:
                    dpg.set_axis_limits_auto(y_axis)
            
            dpg.set_item_callback(log_scale_x, toggle_log_scale_x)
            dpg.set_item_callback(log_scale_y, toggle_log_scale_y)