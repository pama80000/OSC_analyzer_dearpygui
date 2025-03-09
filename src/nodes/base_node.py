#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基本ノードクラスモジュール。

すべてのノードの基本となるクラスを定義します。
"""

import uuid
from typing import Dict, List, Any, Optional, Tuple, Callable
import numpy as np
import dearpygui.dearpygui as dpg


class NodePin:
    """ノードの入出力ピンを表すクラス。"""

    def __init__(self, name: str, pin_type: str = "float", shape: Tuple = None):
        """
        NodePinクラスのコンストラクタ。

        Args:
            name: ピンの名前
            pin_type: データ型 ("float", "int", "array" など)
            shape: 配列の場合の形状 (例: (1000,) は1000要素の1次元配列)
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.pin_type = pin_type
        self.shape = shape
        self.connected_to = None  # 接続先のピンID
        self.parent_node = None  # 親ノード
        self.ui_id = None  # DearPyGUIのアイテムID

    def connect_to(self, target_pin: 'NodePin') -> None:
        """
        このピンを別のピンに接続する。

        Args:
            target_pin: 接続先のピン
        """
        self.connected_to = target_pin.id
        target_pin.connected_to = self.id

    def disconnect(self) -> None:
        """このピンの接続を解除する。"""
        self.connected_to = None


class Node:
    """すべてのノードの基本クラス。"""

    def __init__(self, title: str, pos: Tuple[int, int] = (0, 0)):
        """
        Nodeクラスのコンストラクタ。

        Args:
            title: ノードのタイトル
            pos: ノードの初期位置 (x, y)
        """
        self.id = str(uuid.uuid4())
        self.title = title
        self.pos = pos
        self.input_pins: Dict[str, NodePin] = {}  # 入力ピンの辞書 {名前: ピンオブジェクト}
        self.output_pins: Dict[str, NodePin] = {}  # 出力ピンの辞書
        self.parameters: Dict[str, Any] = {}  # パラメータの辞書
        self.ui_node_id = None  # DearPyGUIのノードID
        self.ui_param_ids = {}  # パラメータUIのID辞書
        self.ui_graph_id = None  # グラフ表示用のID
        self.show_graph = True  # グラフ表示のオン/オフ
        self.result_cache = None  # 処理結果のキャッシュ
        self.is_dirty = True  # 再計算が必要かどうか

    def add_input_pin(self, name: str, pin_type: str = "float", shape: Tuple = None) -> NodePin:
        """
        入力ピンを追加する。

        Args:
            name: ピンの名前
            pin_type: データ型
            shape: 配列の形状

        Returns:
            作成されたNodePinオブジェクト
        """
        pin = NodePin(name, pin_type, shape)
        pin.parent_node = self
        self.input_pins[name] = pin
        return pin

    def add_output_pin(self, name: str, pin_type: str = "float", shape: Tuple = None) -> NodePin:
        """
        出力ピンを追加する。

        Args:
            name: ピンの名前
            pin_type: データ型
            shape: 配列の形状

        Returns:
            作成されたNodePinオブジェクト
        """
        pin = NodePin(name, pin_type, shape)
        pin.parent_node = self
        self.output_pins[name] = pin
        return pin

    def add_parameter(self, name: str, value: Any, param_type: str = "float") -> None:
        """
        パラメータを追加する。

        Args:
            name: パラメータ名
            value: 初期値
            param_type: パラメータの型
        """
        self.parameters[name] = {
            "value": value,
            "type": param_type
        }

    def get_parameter(self, name: str) -> Any:
        """
        パラメータの値を取得する。

        Args:
            name: パラメータ名

        Returns:
            パラメータの値
        """
        return self.parameters[name]["value"]

    def set_parameter(self, name: str, value: Any) -> None:
        """
        パラメータの値を設定する。

        Args:
            name: パラメータ名
            value: 設定する値
        """
        if name in self.parameters:
            self.parameters[name]["value"] = value
            self.is_dirty = True  # パラメータが変更されたので再計算が必要

    def get_input_data(self, pin_name: str) -> Optional[np.ndarray]:
        """
        指定された入力ピンからデータを取得する。

        Args:
            pin_name: 入力ピン名

        Returns:
            入力データ（接続されていない場合はNone）
        """
        if pin_name not in self.input_pins:
            return None

        pin = self.input_pins[pin_name]
        if pin.connected_to is None:
            return None

        # 接続元のノードとピンを見つける
        for node in self.get_pipeline().nodes:
            for out_pin_name, out_pin in node.output_pins.items():
                if out_pin.id == pin.connected_to:
                    # 接続元ノードの出力を取得
                    return node.process()

        return None

    def get_pipeline(self):
        """
        このノードが属するパイプラインを取得する。

        Returns:
            Pipelineオブジェクト
        """
        # 循環インポートを避けるため、ここで動的にインポート
        from src.pipeline.pipeline import Pipeline
        return Pipeline.get_instance()

    def process(self) -> Dict[str, np.ndarray]:
        """
        ノードの処理を実行する。

        Returns:
            出力ピン名をキーとする出力データの辞書
        """
        # キャッシュがあり、かつ再計算が不要な場合はキャッシュを返す
        if self.result_cache is not None and not self.is_dirty:
            return self.result_cache

        # サブクラスでオーバーライドする処理ロジック
        # デフォルトでは何もせず入力をそのまま出力
        result = {}
        for out_name in self.output_pins.keys():
            # 対応する入力ピンがあれば、その値を使用
            if out_name in self.input_pins:
                result[out_name] = self.get_input_data(out_name)
            else:
                # デフォルト値（空の配列）
                result[out_name] = np.array([])

        # 結果をキャッシュし、ダーティフラグをクリア
        self.result_cache = result
        self.is_dirty = False
        return result

    def create_ui(self, parent_id: int) -> None:
        """
        ノードのUIを作成する。

        Args:
            parent_id: 親ウィジェットのID
        """
        # ノード本体を作成
        with dpg.node(label=self.title, pos=self.pos, parent=parent_id) as node_id:
            self.ui_node_id = node_id

            # 入力ピンを作成
            for name, pin in self.input_pins.items():
                with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input) as attr_id:
                    pin.ui_id = attr_id
                    dpg.add_text(name)

            # パラメータUIを作成
            if self.parameters:
                with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
                    for name, param in self.parameters.items():
                        self._create_parameter_ui(name, param)

            # グラフ表示領域（表示フラグがオンの場合のみ）
            if self.show_graph:
                with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
                    # グラフ表示/非表示のチェックボックス
                    with dpg.group(horizontal=True):
                        dpg.add_text("グラフ表示")
                        show_graph_cb = dpg.add_checkbox(
                            default_value=self.show_graph,
                            callback=lambda sender, data: self._toggle_graph_visibility(data)
                        )

                    # グラフウィジェット
                    with dpg.group() as graph_group:
                        self.ui_graph_id = graph_group
                        # グラフの内容はupdate_ui()で更新

            # 出力ピンを作成
            for name, pin in self.output_pins.items():
                with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output) as attr_id:
                    pin.ui_id = attr_id
                    dpg.add_text(name)

    def _create_parameter_ui(self, name: str, param: Dict) -> None:
        """
        パラメータのUIコントロールを作成する。

        Args:
            name: パラメータ名
            param: パラメータ情報の辞書
        """
        param_type = param["type"]
        value = param["value"]

        with dpg.group(horizontal=True):
            dpg.add_text(f"{name}: ")
            
            # パラメータの型に応じたUIコントロールを作成
            if param_type == "float":
                ui_id = dpg.add_input_float(
                    default_value=value,
                    width=150,
                    callback=lambda sender, data: self.set_parameter(name, data)
                )
            elif param_type == "int":
                ui_id = dpg.add_input_int(
                    default_value=value,
                    width=150,
                    callback=lambda sender, data: self.set_parameter(name, data)
                )
            elif param_type == "text":
                ui_id = dpg.add_input_text(
                    default_value=value,
                    width=150,
                    callback=lambda sender, data: self.set_parameter(name, data)
                )
            elif param_type == "bool":
                ui_id = dpg.add_checkbox(
                    default_value=value,
                    callback=lambda sender, data: self.set_parameter(name, data)
                )
            elif param_type == "combo":
                # コンボボックスの場合、valueは[選択肢リスト, 初期選択インデックス]の形式
                options, default_idx = value
                ui_id = dpg.add_combo(
                    items=options,
                    default_value=options[default_idx],
                    width=150,
                    callback=lambda sender, data: self.set_parameter(
                        name, (options, options.index(data))
                    )
                )
            else:
                # その他の型はテキスト表示のみ
                ui_id = dpg.add_text(str(value))

            self.ui_param_ids[name] = ui_id

    def _toggle_graph_visibility(self, show: bool) -> None:
        """
        グラフ表示の可視性を切り替える。

        Args:
            show: 表示するかどうか
        """
        self.show_graph = show
        if self.ui_graph_id is not None:
            if show:
                dpg.show_item(self.ui_graph_id)
            else:
                dpg.hide_item(self.ui_graph_id)

    def update_ui(self) -> None:
        """ノードのUI（特にグラフ）を更新する。"""
        # グラフが表示されていない場合は更新しない
        if not self.show_graph or self.ui_graph_id is None:
            return

        # 既存のグラフをクリア
        dpg.delete_item(self.ui_graph_id, children_only=True)

        # 処理結果を取得
        outputs = self.process()
        if not outputs:
            return

        # 各出力データに対してグラフを作成
        for name, data in outputs.items():
            if data is None or not isinstance(data, np.ndarray) or len(data) == 0:
                continue

            # プロットするデータの準備
            if data.ndim == 1:
                # 1次元データの場合
                y_data = data
                x_data = np.arange(len(y_data))
            else:
                # 多次元データの場合は最初の次元のみプロット
                y_data = data[0]
                x_data = np.arange(len(y_data))

            # グラフの作成
            with dpg.plot(label=f"{name}", height=200, width=-1, parent=self.ui_graph_id):
                # 軸の設定
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label="X")
                with dpg.plot_axis(dpg.mvYAxis, label="Y"):
                    # データの追加
                    dpg.add_line_series(
                        x_data.tolist(),
                        y_data.tolist(),
                        label=name
                    )

    def serialize(self) -> Dict:
        """
        ノードの状態をシリアライズする。

        Returns:
            シリアライズされたノードの状態
        """
        # 基本情報
        data = {
            "id": self.id,
            "type": self.__class__.__name__,
            "title": self.title,
            "pos": self.pos,
            "show_graph": self.show_graph,
            "parameters": {}
        }

        # パラメータ
        for name, param in self.parameters.items():
            data["parameters"][name] = {
                "value": param["value"],
                "type": param["type"]
            }

        # 入力ピン
        data["input_pins"] = {}
        for name, pin in self.input_pins.items():
            data["input_pins"][name] = {
                "id": pin.id,
                "pin_type": pin.pin_type,
                "shape": pin.shape,
                "connected_to": pin.connected_to
            }

        # 出力ピン
        data["output_pins"] = {}
        for name, pin in self.output_pins.items():
            data["output_pins"][name] = {
                "id": pin.id,
                "pin_type": pin.pin_type,
                "shape": pin.shape,
                "connected_to": pin.connected_to
            }

        return data

    @classmethod
    def deserialize(cls, data: Dict) -> 'Node':
        """
        シリアライズされたデータからノードを復元する。

        Args:
            data: シリアライズされたノードデータ

        Returns:
            復元されたNodeオブジェクト
        """
        # 基本情報で新しいノードを作成
        node = cls(data["title"], data["pos"])
        node.id = data["id"]
        node.show_graph = data["show_graph"]

        # パラメータを復元
        for name, param_data in data["parameters"].items():
            node.parameters[name] = {
                "value": param_data["value"],
                "type": param_data["type"]
            }

        # 入力ピンを復元
        for name, pin_data in data["input_pins"].items():
            pin = node.add_input_pin(name, pin_data["pin_type"], pin_data["shape"])
            pin.id = pin_data["id"]
            pin.connected_to = pin_data["connected_to"]

        # 出力ピンを復元
        for name, pin_data in data["output_pins"].items():
            pin = node.add_output_pin(name, pin_data["pin_type"], pin_data["shape"])
            pin.id = pin_data["id"]
            pin.connected_to = pin_data["connected_to"]

        return node