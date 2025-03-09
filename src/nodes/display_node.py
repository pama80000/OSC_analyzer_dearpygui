#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
波形表示ノードモジュール。

波形データをグラフとして表示するノードを実装します。
"""

import numpy as np
from typing import Dict, Tuple, Any, Optional, List

import dearpygui.dearpygui as dpg

from src.nodes.base_node import Node


class WaveformDisplayNode(Node):
    """波形表示ノード。"""

    def __init__(self, title: str, pos: Tuple[int, int] = (0, 0)):
        """
        WaveformDisplayNodeクラスのコンストラクタ。

        Args:
            title: ノードのタイトル
            pos: ノードの初期位置 (x, y)
        """
        super().__init__(title, pos)
        
        # 入力ピンの追加
        self.add_input_pin("time", "array")
        self.add_input_pin("data", "array")
        
        # 出力ピンの追加（入力をそのまま出力）
        self.add_output_pin("time", "array")
        self.add_output_pin("data", "array")
        
        # パラメータの追加
        self.add_parameter("line_color", (0, 255, 0, 255), "color")
        self.add_parameter("line_thickness", 1.0, "float")
        self.add_parameter("marker_type", (["None", "Circle", "Square", "Diamond", "Up", "Down", "Left", "Right"], 0), "combo")
        self.add_parameter("marker_size", 4.0, "float")
        self.add_parameter("show_grid", True, "bool")
        self.add_parameter("auto_fit", True, "bool")
        self.add_parameter("y_min", 0.0, "float")
        self.add_parameter("y_max", 1.0, "float")
        
        # グラフ表示のオプション
        self.show_graph = True  # 常に表示
        self.plot_id = None

    def process(self) -> Dict[str, np.ndarray]:
        """
        波形表示処理を実行する（入力をそのまま出力）。

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
        if time_array is None or data_array is None:
            self.result_cache = {
                "time": np.array([]),
                "data": np.array([])
            }
        else:
            # 入力をそのまま出力
            self.result_cache = {
                "time": time_array,
                "data": data_array
            }
        
        self.is_dirty = False
        return self.result_cache

    def update_ui(self) -> None:
        """ノードのUI（特にグラフ）を更新する。"""
        # グラフが表示されていない場合は更新しない
        if not self.show_graph or self.ui_graph_id is None:
            return
        
        # 処理結果を取得
        outputs = self.process()
        if not outputs:
            return
        
        # 時間と波形データの取得
        time_array = outputs.get("time")
        data_array = outputs.get("data")
        
        if time_array is None or data_array is None or len(time_array) == 0 or len(data_array) == 0:
            return
        
        # 既存のグラフをクリア
        dpg.delete_item(self.ui_graph_id, children_only=True)
        
        # パラメータの取得
        line_color = self.get_parameter("line_color")
        line_thickness = self.get_parameter("line_thickness")
        marker_options, marker_idx = self.get_parameter("marker_type")
        marker_type = marker_options[marker_idx]
        marker_size = self.get_parameter("marker_size")
        show_grid = self.get_parameter("show_grid")
        auto_fit = self.get_parameter("auto_fit")
        y_min = self.get_parameter("y_min")
        y_max = self.get_parameter("y_max")
        
        # マーカータイプの変換
        marker_symbol = dpg.mvPlotMarker_None
        if marker_type == "Circle":
            marker_symbol = dpg.mvPlotMarker_Circle
        elif marker_type == "Square":
            marker_symbol = dpg.mvPlotMarker_Square
        elif marker_type == "Diamond":
            marker_symbol = dpg.mvPlotMarker_Diamond
        elif marker_type == "Up":
            marker_symbol = dpg.mvPlotMarker_Up
        elif marker_type == "Down":
            marker_symbol = dpg.mvPlotMarker_Down
        elif marker_type == "Left":
            marker_symbol = dpg.mvPlotMarker_Left
        elif marker_type == "Right":
            marker_symbol = dpg.mvPlotMarker_Right
        
        # データ点数が多い場合は間引く
        if len(time_array) > 2000:
            step = len(time_array) // 2000
            plot_time = time_array[::step]
            plot_data = data_array[::step]
        else:
            plot_time = time_array
            plot_data = data_array
        
        # 波形グラフの作成
        with dpg.plot(label="波形", height=300, width=-1, parent=self.ui_graph_id) as plot:
            self.plot_id = plot
            
            # 凡例の追加
            dpg.add_plot_legend()
            
            # グリッドの設定
            if show_grid:
                dpg.add_plot_grid()
            
            # X軸（時間）
            x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="時間")
            dpg.set_axis_limits_auto(x_axis)
            
            # Y軸（振幅）
            with dpg.plot_axis(dpg.mvYAxis, label="振幅") as y_axis:
                # Y軸の範囲設定
                if not auto_fit:
                    dpg.set_axis_limits(y_axis, y_min, y_max)
                else:
                    dpg.set_axis_limits_auto(y_axis)
                
                # 波形データの表示
                dpg.add_line_series(
                    plot_time.tolist(),
                    plot_data.tolist(),
                    label=self.title,
                    color=line_color,
                    weight=line_thickness,
                    marker=marker_symbol,
                    marker_size=marker_size
                )
        
        # 表示オプションのコントロール
        with dpg.group(parent=self.ui_graph_id, horizontal=True):
            # 自動フィットのチェックボックス
            auto_fit_cb = dpg.add_checkbox(
                label="自動フィット",
                default_value=auto_fit,
                callback=lambda sender, data: self._toggle_auto_fit(data)
            )
            
            # Y軸範囲の入力（自動フィットがオフの場合）
            y_min_input = dpg.add_input_float(
                label="Y最小値",
                default_value=y_min,
                width=100,
                enabled=not auto_fit,
                callback=lambda sender, data: self.set_parameter("y_min", data)
            )
            
            y_max_input = dpg.add_input_float(
                label="Y最大値",
                default_value=y_max,
                width=100,
                enabled=not auto_fit,
                callback=lambda sender, data: self.set_parameter("y_max", data)
            )
            
            # 自動フィット切り替え時のコールバック
            def toggle_y_inputs(auto_fit):
                dpg.configure_item(y_min_input, enabled=not auto_fit)
                dpg.configure_item(y_max_input, enabled=not auto_fit)
            
            dpg.set_item_callback(auto_fit_cb, lambda sender, data: (
                self.set_parameter("auto_fit", data),
                toggle_y_inputs(data)
            ))
        
        # 統計情報の表示
        with dpg.group(parent=self.ui_graph_id):
            # 基本統計量の計算
            stats = {
                "データ点数": len(data_array),
                "最小値": float(np.min(data_array)),
                "最大値": float(np.max(data_array)),
                "平均値": float(np.mean(data_array)),
                "標準偏差": float(np.std(data_array)),
                "RMS": float(np.sqrt(np.mean(np.square(data_array))))
            }
            
            # 統計情報テーブルの作成
            with dpg.table(header_row=True, borders_innerH=True, borders_outerH=True, borders_innerV=True, borders_outerV=True):
                dpg.add_table_column(label="統計量")
                dpg.add_table_column(label="値")
                
                for stat_name, stat_value in stats.items():
                    with dpg.table_row():
                        dpg.add_text(stat_name)
                        dpg.add_text(f"{stat_value:.6g}")

    def _toggle_auto_fit(self, auto_fit: bool) -> None:
        """
        自動フィットの切り替え。

        Args:
            auto_fit: 自動フィットを有効にするかどうか
        """
        self.set_parameter("auto_fit", auto_fit)
        
        # プロットが存在する場合、Y軸の範囲を更新
        if self.plot_id is not None and dpg.does_item_exist(self.plot_id):
            # Y軸を取得
            for item in dpg.get_item_children(self.plot_id, 1):
                if dpg.get_item_type(item) == "mvAppItemType::mvPlotAxis" and dpg.get_axis_direction(item) == 1:  # Y軸
                    if auto_fit:
                        dpg.set_axis_limits_auto(item)
                    else:
                        y_min = self.get_parameter("y_min")
                        y_max = self.get_parameter("y_max")
                        dpg.set_axis_limits(item, y_min, y_max)
                    break