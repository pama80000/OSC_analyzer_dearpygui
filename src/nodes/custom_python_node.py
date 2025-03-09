#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
カスタムPythonノードモジュール。

ユーザーが入力したPythonコードでデータを変換するノードを実装します。
"""

import numpy as np
from typing import Dict, Tuple, Any, Optional

import dearpygui.dearpygui as dpg

from src.nodes.base_node import Node
from src.utils.data_utils import execute_custom_code


class CustomPythonNode(Node):
    """カスタムPythonノード。"""

    def __init__(self, title: str, pos: Tuple[int, int] = (0, 0)):
        """
        CustomPythonNodeクラスのコンストラクタ。

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
        self.add_output_pin("result", "array")
        
        # パラメータの追加
        self.add_parameter("code", DEFAULT_CODE, "text")
        self.add_parameter("input_var", "data", "text")
        self.add_parameter("output_var", "result", "text")
        self.add_parameter("process_time", False, "bool")
        
        # コードエディタのID
        self.code_editor_id = None
        self.code_window_id = None
        self.error_text_id = None

    def process(self) -> Dict[str, np.ndarray]:
        """
        カスタムPython処理を実行する。

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
        if data_array is None or len(data_array) == 0:
            self.result_cache = {
                "time": np.array([]) if time_array is None else time_array,
                "result": np.array([])
            }
            self.is_dirty = False
            return self.result_cache
        
        # パラメータの取得
        code = self.get_parameter("code")
        input_var = self.get_parameter("input_var")
        output_var = self.get_parameter("output_var")
        process_time = self.get_parameter("process_time")
        
        try:
            # カスタムコードの実行
            if process_time and time_array is not None and len(time_array) > 0:
                # 時間データも処理する場合
                locals_dict = {
                    input_var: data_array,
                    "time": time_array,
                    'np': np
                }
                
                # コードの実行
                exec(code, globals(), locals_dict)
                
                # 結果の取得
                if output_var not in locals_dict:
                    raise ValueError(f"出力変数 '{output_var}' が見つかりません")
                
                result = locals_dict[output_var]
                
                # 時間データの更新（オプション）
                if "time_out" in locals_dict and isinstance(locals_dict["time_out"], np.ndarray):
                    time_result = locals_dict["time_out"]
                else:
                    time_result = time_array
            else:
                # データのみを処理する場合
                result = execute_custom_code(data_array, code, input_var, output_var)
                time_result = time_array
            
            # エラーテキストをクリア
            if self.error_text_id is not None:
                dpg.set_value(self.error_text_id, "")
            
            # 結果をキャッシュし、ダーティフラグをクリア
            self.result_cache = {
                "time": time_result,
                "result": result
            }
            self.is_dirty = False
            
            return self.result_cache
        
        except Exception as e:
            # エラーが発生した場合はエラーメッセージを表示
            error_msg = f"コード実行エラー: {str(e)}"
            print(error_msg)
            
            # エラーテキストを更新
            if self.error_text_id is not None:
                dpg.set_value(self.error_text_id, error_msg)
            
            # 入力データをそのまま返す
            self.result_cache = {
                "time": time_array,
                "result": data_array
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
        
        # コードエディタボタンの追加
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static, parent=self.ui_node_id):
            with dpg.group(horizontal=True):
                dpg.add_button(label="コードエディタを開く", callback=self._open_code_editor)
                
                # エラーテキスト
                with dpg.group():
                    self.error_text_id = dpg.add_text("", color=(255, 0, 0))

    def _open_code_editor(self) -> None:
        """コードエディタウィンドウを開く。"""
        # 既存のウィンドウがあれば閉じる
        if self.code_window_id is not None and dpg.does_item_exist(self.code_window_id):
            dpg.delete_item(self.code_window_id)
        
        # コードの取得
        code = self.get_parameter("code")
        
        # コードエディタウィンドウの作成
        with dpg.window(label=f"Pythonコードエディタ - {self.title}", width=800, height=600, on_close=self._on_editor_close) as window:
            self.code_window_id = window
            
            # コードエディタ
            with dpg.group():
                # 入力変数と出力変数の設定
                with dpg.group(horizontal=True):
                    dpg.add_text("入力変数名:")
                    input_var = dpg.add_input_text(
                        default_value=self.get_parameter("input_var"),
                        width=100,
                        callback=lambda sender, data: self.set_parameter("input_var", data)
                    )
                    
                    dpg.add_text("出力変数名:")
                    output_var = dpg.add_input_text(
                        default_value=self.get_parameter("output_var"),
                        width=100,
                        callback=lambda sender, data: self.set_parameter("output_var", data)
                    )
                    
                    dpg.add_checkbox(
                        label="時間データも処理",
                        default_value=self.get_parameter("process_time"),
                        callback=lambda sender, data: self.set_parameter("process_time", data)
                    )
                
                # コードエディタ
                self.code_editor_id = dpg.add_input_text(
                    default_value=code,
                    width=-1,
                    height=-40,
                    multiline=True,
                    tab_input=True,
                    callback=lambda sender, data: self.set_parameter("code", data)
                )
                
                # ボタン
                with dpg.group(horizontal=True):
                    dpg.add_button(label="適用", callback=self._apply_code)
                    dpg.add_button(label="キャンセル", callback=lambda: dpg.delete_item(self.code_window_id))
                    dpg.add_button(label="デフォルトコードに戻す", callback=self._reset_to_default)

    def _apply_code(self) -> None:
        """コードエディタの内容を適用する。"""
        if self.code_editor_id is not None:
            code = dpg.get_value(self.code_editor_id)
            self.set_parameter("code", code)
            self.is_dirty = True  # 再計算が必要

    def _on_editor_close(self) -> None:
        """コードエディタが閉じられたときのコールバック。"""
        self.code_window_id = None
        self.code_editor_id = None

    def _reset_to_default(self) -> None:
        """コードをデフォルトに戻す。"""
        if self.code_editor_id is not None:
            dpg.set_value(self.code_editor_id, DEFAULT_CODE)
            self.set_parameter("code", DEFAULT_CODE)
            self.is_dirty = True  # 再計算が必要


# デフォルトのPythonコード
DEFAULT_CODE = """# 入力データは 'data' 変数に格納されています
# 時間データは 'time' 変数に格納されています（process_timeがTrueの場合）
# numpy は 'np' としてインポート済みです

# 例: 移動平均フィルタ
window_size = 10
result = np.convolve(data, np.ones(window_size)/window_size, mode='same')

# 時間データを処理する例（process_timeがTrueの場合）
# time_out = time  # 時間データを変更する場合は time_out に代入

# 出力は 'result' 変数に格納してください
"""