#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
データ読み込みノードモジュール。

.trcファイルまたはCSVファイルからデータを読み込むノードを実装します。
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, Optional

import dearpygui.dearpygui as dpg

from src.nodes.base_node import Node
from src.utils.trc_reader import read_trc_file


class DataLoadNode(Node):
    """データ読み込みノード。"""

    def __init__(self, title: str, pos: Tuple[int, int] = (0, 0)):
        """
        DataLoadNodeクラスのコンストラクタ。

        Args:
            title: ノードのタイトル
            pos: ノードの初期位置 (x, y)
        """
        super().__init__(title, pos)
        
        # 出力ピンの追加
        self.add_output_pin("time", "array")
        self.add_output_pin("data", "array")
        
        # パラメータの追加
        self.add_parameter("file_path", "", "text")
        self.add_parameter("file_type", (["Auto", "TRC", "CSV"], 0), "combo")
        
        # ファイル選択ダイアログのID
        self.file_dialog_id = None

    def process(self) -> Dict[str, np.ndarray]:
        """
        データ読み込み処理を実行する。

        Returns:
            出力ピン名をキーとする出力データの辞書
        """
        # キャッシュがあり、かつ再計算が不要な場合はキャッシュを返す
        if self.result_cache is not None and not self.is_dirty:
            return self.result_cache
        
        # ファイルパスの取得
        file_path = self.get_parameter("file_path")
        if not file_path or not os.path.exists(file_path):
            # ファイルが存在しない場合は空の配列を返す
            self.result_cache = {
                "time": np.array([]),
                "data": np.array([])
            }
            self.is_dirty = False
            return self.result_cache
        
        # ファイルタイプの取得
        file_type_options, file_type_idx = self.get_parameter("file_type")
        file_type = file_type_options[file_type_idx]
        
        # ファイルタイプの自動判定
        if file_type == "Auto":
            ext = os.path.splitext(file_path)[1].lower()
            if ext == ".trc":
                file_type = "TRC"
            elif ext == ".csv":
                file_type = "CSV"
            else:
                # 未対応の拡張子の場合はCSVとして扱う
                file_type = "CSV"
        
        try:
            # ファイルタイプに応じた読み込み処理
            if file_type == "TRC":
                time_array, data_array, _ = read_trc_file(file_path)
            else:  # CSV
                # CSVファイルの読み込み
                df = pd.read_csv(file_path)
                
                # 列数に応じた処理
                if len(df.columns) >= 2:
                    # 最初の列を時間、2列目をデータとして扱う
                    time_array = df.iloc[:, 0].values
                    data_array = df.iloc[:, 1].values
                else:
                    # 1列のみの場合はインデックスを時間として扱う
                    time_array = np.arange(len(df))
                    data_array = df.iloc[:, 0].values
            
            # 結果をキャッシュし、ダーティフラグをクリア
            self.result_cache = {
                "time": time_array,
                "data": data_array
            }
            self.is_dirty = False
            
            return self.result_cache
        
        except Exception as e:
            # エラーが発生した場合は空の配列を返す
            print(f"データ読み込みエラー: {str(e)}")
            self.result_cache = {
                "time": np.array([]),
                "data": np.array([])
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
        
        # ファイル選択ダイアログの作成
        with dpg.file_dialog(
            directory_selector=False, show=False, callback=self._on_file_selected, id=f"file_dialog_{self.id}",
            width=700, height=400
        ):
            dpg.add_file_extension(".trc", color=(0, 255, 0, 255))
            dpg.add_file_extension(".csv", color=(0, 255, 255, 255))
        
        self.file_dialog_id = f"file_dialog_{self.id}"
        
        # ファイル選択ボタンの追加
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static, parent=self.ui_node_id):
            with dpg.group(horizontal=True):
                dpg.add_button(label="ファイル選択", callback=self._on_file_button_click)

    def _on_file_button_click(self) -> None:
        """ファイル選択ボタンがクリックされたときのコールバック。"""
        dpg.show_item(self.file_dialog_id)

    def _on_file_selected(self, sender: int, app_data: Dict) -> None:
        """
        ファイル選択ダイアログでファイルが選択されたときのコールバック。

        Args:
            sender: 送信元のID
            app_data: ファイル情報
        """
        file_path = app_data["file_path_name"]
        self.set_parameter("file_path", file_path)
        
        # ファイルタイプの自動設定
        ext = os.path.splitext(file_path)[1].lower()
        file_type_options, _ = self.get_parameter("file_type")
        
        if ext == ".trc":
            self.set_parameter("file_type", (file_type_options, file_type_options.index("TRC")))
        elif ext == ".csv":
            self.set_parameter("file_type", (file_type_options, file_type_options.index("CSV")))
        else:
            self.set_parameter("file_type", (file_type_options, file_type_options.index("Auto")))