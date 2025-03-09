#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
特徴量抽出ノードモジュール。

波形から特徴量を抽出するノードを実装します。
"""

import numpy as np
from typing import Dict, Tuple, Any, Optional

import dearpygui.dearpygui as dpg

from src.nodes.base_node import Node
from src.utils.data_utils import extract_features


class FeatureExtractionNode(Node):
    """特徴量抽出ノード。"""

    def __init__(self, title: str, pos: Tuple[int, int] = (0, 0)):
        """
        FeatureExtractionNodeクラスのコンストラクタ。

        Args:
            title: ノードのタイトル
            pos: ノードの初期位置 (x, y)
        """
        super().__init__(title, pos)
        
        # 入力ピンの追加
        self.add_input_pin("time", "array")
        self.add_input_pin("data", "array")
        
        # 出力ピンの追加
        self.add_output_pin("features", "dict")
        
        # パラメータの追加
        self.add_parameter("extract_basic", True, "bool")
        self.add_parameter("extract_peaks", True, "bool")
        self.add_parameter("extract_time", True, "bool")

    def process(self) -> Dict[str, Any]:
        """
        特徴量抽出処理を実行する。

        Returns:
            出力ピン名をキーとする出力データの辞書
        """
        # キャッシュがあり、かつ再計算が不要な場合はキャッシュを返す
        if self.result_cache is not None and not self.is_dirty:
            return self.result_cache
        
        # 入力データの取得
        time_array = self.get_input_data("time")
        data_array = self.get_input_data("data")
        
        # 入力データがない場合は空の辞書を返す
        if data_array is None or len(data_array) == 0:
            self.result_cache = {
                "features": {}
            }
            self.is_dirty = False
            return self.result_cache
        
        try:
            # 特徴量の抽出
            features = extract_features(data_array, time_array)
            
            # パラメータに基づいて特徴量をフィルタリング
            filtered_features = {}
            
            # 基本統計量
            if self.get_parameter("extract_basic"):
                basic_features = ["max", "min", "peak_to_peak", "mean", "median", "std", "rms"]
                for feature in basic_features:
                    if feature in features:
                        filtered_features[feature] = features[feature]
            
            # ピーク関連
            if self.get_parameter("extract_peaks"):
                peak_features = ["zero_crossing_count", "peak_count"]
                for feature in peak_features:
                    if feature in features:
                        filtered_features[feature] = features[feature]
            
            # 時間関連
            if self.get_parameter("extract_time") and time_array is not None and len(time_array) > 0:
                time_features = ["duration", "mean_peak_interval"]
                for feature in time_features:
                    if feature in features:
                        filtered_features[feature] = features[feature]
            
            # 結果をキャッシュし、ダーティフラグをクリア
            self.result_cache = {
                "features": filtered_features
            }
            self.is_dirty = False
            
            return self.result_cache
        
        except Exception as e:
            # エラーが発生した場合は空の辞書を返す
            print(f"特徴量抽出エラー: {str(e)}")
            self.result_cache = {
                "features": {}
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
        
        # 特徴量の取得
        features = outputs.get("features", {})
        if not features:
            return
        
        # 既存のグラフをクリア
        dpg.delete_item(self.ui_graph_id, children_only=True)
        
        # 特徴量の表示
        with dpg.table(header_row=True, parent=self.ui_graph_id, borders_innerH=True, borders_outerH=True, borders_innerV=True, borders_outerV=True):
            # 列の設定
            dpg.add_table_column(label="特徴量")
            dpg.add_table_column(label="値")
            
            # 特徴量の行を追加
            for feature_name, feature_value in features.items():
                with dpg.table_row():
                    dpg.add_text(feature_name)
                    
                    # 値の表示（型に応じて）
                    if isinstance(feature_value, (int, np.integer)):
                        dpg.add_text(f"{feature_value}")
                    elif isinstance(feature_value, (float, np.floating)):
                        dpg.add_text(f"{feature_value:.6g}")
                    else:
                        dpg.add_text(str(feature_value))
        
        # 棒グラフの表示（数値特徴量のみ）
        numeric_features = {k: v for k, v in features.items() if isinstance(v, (int, float, np.integer, np.floating))}
        if numeric_features:
            with dpg.plot(label="特徴量", height=200, width=-1, parent=self.ui_graph_id):
                # 軸の設定
                dpg.add_plot_legend()
                
                # X軸（特徴量名）
                x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="特徴量")
                dpg.set_axis_ticks(x_axis, list(zip(range(len(numeric_features)), numeric_features.keys())))
                
                # Y軸（値）
                with dpg.plot_axis(dpg.mvYAxis, label="値") as y_axis:
                    # 棒グラフの追加
                    dpg.add_bar_series(
                        list(range(len(numeric_features))),
                        list(numeric_features.values()),
                        label="特徴量値"
                    )