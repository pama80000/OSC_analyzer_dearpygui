#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ノードエディタモジュール。

ノードの配置、接続、実行などの機能を提供するGUIを実装します。
"""

import os
import json
from typing import Dict, List, Any, Optional, Tuple, Type, Set
import dearpygui.dearpygui as dpg
import numpy as np

from src.nodes.base_node import Node
from src.pipeline.pipeline import Pipeline


class NodeEditor:
    """ノードエディタのGUIを管理するクラス。"""

    def __init__(self, width: int = 1200, height: int = 800):
        """
        NodeEditorクラスのコンストラクタ。

        Args:
            width: ウィンドウの幅
            height: ウィンドウの高さ
        """
        self.width = width
        self.height = height
        self.pipeline = Pipeline.get_instance()
        self.node_registry = {}  # ノードタイプの登録辞書 {名前: クラス}
        self.node_editor_id = None  # ノードエディタのID
        self.link_handler_id = None  # リンクハンドラのID
        self.context_menu_id = None  # コンテキストメニューのID
        self.selected_node = None  # 選択中のノード
        self.is_running = False  # パイプラインの実行中フラグ

    def register_node_type(self, node_class: Type[Node], name: str = None) -> None:
        """
        ノードタイプを登録する。

        Args:
            node_class: 登録するノードクラス
            name: 表示名（Noneの場合はクラス名を使用）
        """
        if name is None:
            name = node_class.__name__
        self.node_registry[name] = node_class

    def create_ui(self) -> None:
        """ノードエディタのUIを作成する。"""
        # DearPyGUIのセットアップ
        dpg.create_context()
        dpg.create_viewport(title="オシロスコープデータ処理フロー", width=self.width, height=self.height)
        dpg.setup_dearpygui()

        # テーマの設定
        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                # 基本的なUIの色設定
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (42, 42, 42))
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (32, 32, 32))
                
                # ノードエディタの色設定
                dpg.add_theme_color(dpg.mvNodeCol_NodeBackground, (50, 50, 50))
                dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundHovered, (55, 55, 55))
                dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundSelected, (60, 60, 60))
                dpg.add_theme_color(dpg.mvNodeCol_NodeOutline, (100, 100, 100))
                
                # ノードのスタイル設定
                dpg.add_theme_style(dpg.mvNodeStyleVar_NodePadding, 8, 8)
                dpg.add_theme_style(dpg.mvNodeStyleVar_NodeCornerRounding, 4)

        dpg.bind_theme(global_theme)

        # フォントの設定
        with dpg.font_registry():
            default_font = dpg.add_font("C:/Windows/Fonts/meiryo.ttc", 16) if os.name == 'nt' else None
            if default_font:
                dpg.bind_font(default_font)

        # メインウィンドウ
        with dpg.window(label="ノードエディタ", width=self.width, height=self.height) as main_window:
            # メニューバー
            with dpg.menu_bar():
                with dpg.menu(label="ファイル"):
                    dpg.add_menu_item(label="新規", callback=self._on_new)
                    dpg.add_menu_item(label="開く", callback=self._on_open)
                    dpg.add_menu_item(label="保存", callback=self._on_save)
                    dpg.add_menu_item(label="名前を付けて保存", callback=self._on_save_as)
                    dpg.add_separator()
                    dpg.add_menu_item(label="終了", callback=lambda: dpg.stop_dearpygui())

                with dpg.menu(label="編集"):
                    dpg.add_menu_item(label="ノードを削除", callback=self._on_delete_selected)
                    dpg.add_menu_item(label="すべてクリア", callback=self._on_clear_all)

                with dpg.menu(label="実行"):
                    dpg.add_menu_item(label="パイプラインを実行", callback=self._on_execute_pipeline)
                    dpg.add_menu_item(label="モジュールとしてエクスポート", callback=self._on_export_module)

                with dpg.menu(label="ノード追加"):
                    # 登録されたノードタイプをメニューに追加
                    for node_name, node_class in self.node_registry.items():
                        dpg.add_menu_item(
                            label=node_name,
                            callback=lambda sender, data, node_cls=node_class: self._on_add_node(node_cls)
                        )

            # ノードエディタ
            with dpg.child_window(width=-1, height=-1, border=False):
                # ノードエディタウィジェット
                with dpg.node_editor(callback=self._on_link_connect, delink_callback=self._on_link_disconnect) as node_editor:
                    self.node_editor_id = node_editor

                    # コンテキストメニュー
                    with dpg.popup(node_editor, mousebutton=dpg.mvMouseButton_Right) as context_menu:
                        self.context_menu_id = context_menu
                        with dpg.menu(label="ノード追加"):
                            for node_name, node_class in self.node_registry.items():
                                dpg.add_menu_item(
                                    label=node_name,
                                    callback=lambda sender, data, node_cls=node_class: self._on_add_node_at_mouse(node_cls)
                                )

        # ステータスバーウィンドウ
        with dpg.window(label="ステータス", width=self.width, height=30, pos=(0, self.height - 30), no_title_bar=True, no_resize=True):
            self.status_text = dpg.add_text("準備完了")

        # ファイル選択ダイアログ
        with dpg.file_dialog(
            directory_selector=False, show=False, callback=self._on_file_selected, id="file_dialog",
            width=700, height=400
        ):
            dpg.add_file_extension(".json", color=(0, 255, 0, 255))

        # ビューポートの設定
        dpg.set_primary_window(main_window, True)
        dpg.show_viewport()

    def run(self) -> None:
        """アプリケーションのメインループを実行する。"""
        # メインループ
        while dpg.is_dearpygui_running():
            # UIの更新
            self._update_ui()
            
            # DearPyGUIのレンダリング
            dpg.render_dearpygui_frame()

        # クリーンアップ
        dpg.destroy_context()

    def _update_ui(self) -> None:
        """UIを更新する。"""
        # 各ノードのUIを更新
        for node in self.pipeline.nodes:
            node.update_ui()

    def _on_add_node(self, node_class: Type[Node]) -> None:
        """
        新しいノードを追加する。

        Args:
            node_class: 追加するノードのクラス
        """
        # ノードの作成
        node = node_class(f"{node_class.__name__}_{len(self.pipeline.nodes)}", (100, 100))
        
        # パイプラインに追加
        self.pipeline.add_node(node)
        
        # ノードのUIを作成
        node.create_ui(self.node_editor_id)

    def _on_add_node_at_mouse(self, node_class: Type[Node]) -> None:
        """
        マウス位置に新しいノードを追加する。

        Args:
            node_class: 追加するノードのクラス
        """
        # マウス位置の取得
        mouse_pos = dpg.get_mouse_pos(local=False)
        node_editor_pos = dpg.get_item_pos(self.node_editor_id)
        
        # ノードエディタ内の相対位置を計算
        rel_pos = (mouse_pos[0] - node_editor_pos[0], mouse_pos[1] - node_editor_pos[1])
        
        # ノードの作成
        node = node_class(f"{node_class.__name__}_{len(self.pipeline.nodes)}", rel_pos)
        
        # パイプラインに追加
        self.pipeline.add_node(node)
        
        # ノードのUIを作成
        node.create_ui(self.node_editor_id)

    def _on_delete_selected(self) -> None:
        """選択中のノードを削除する。"""
        if self.selected_node:
            # パイプラインからノードを削除
            self.pipeline.remove_node(self.selected_node)
            
            # UIからノードを削除
            dpg.delete_item(self.selected_node.ui_node_id)
            
            self.selected_node = None

    def _on_clear_all(self) -> None:
        """すべてのノードとリンクをクリアする。"""
        # UIからすべてのノードを削除
        for node in self.pipeline.nodes:
            dpg.delete_item(node.ui_node_id)
        
        # パイプラインをクリア
        self.pipeline.clear()

    def _on_link_connect(self, sender: int, app_data: Dict) -> None:
        """
        ノード間のリンクが接続されたときのコールバック。

        Args:
            sender: 送信元のID
            app_data: 接続情報
        """
        # 接続情報の取得
        from_attr_id = app_data[0]
        to_attr_id = app_data[1]
        
        # 接続するピンを見つける
        from_pin = None
        to_pin = None
        
        for node in self.pipeline.nodes:
            for pin in node.output_pins.values():
                if pin.ui_id == from_attr_id:
                    from_pin = pin
                    break
            
            for pin in node.input_pins.values():
                if pin.ui_id == to_attr_id:
                    to_pin = pin
                    break
        
        # ピンが見つかった場合、接続を作成
        if from_pin and to_pin:
            # 既存の接続を解除
            if to_pin.connected_to:
                self.pipeline.disconnect_pins(to_pin.connected_to, to_pin.id)
            
            # 新しい接続を作成
            self.pipeline.connect_pins(from_pin.id, to_pin.id)
            
            # ノードを再計算が必要な状態にする
            to_pin.parent_node.is_dirty = True

    def _on_link_disconnect(self, sender: int, app_data: Dict) -> None:
        """
        ノード間のリンクが切断されたときのコールバック。

        Args:
            sender: 送信元のID
            app_data: 切断情報
        """
        # 切断情報の取得
        link_id = app_data
        
        # 切断するピンを見つける
        for node in self.pipeline.nodes:
            for pin in node.input_pins.values():
                if pin.connected_to:
                    # 接続を解除
                    self.pipeline.disconnect_pins(pin.connected_to, pin.id)
                    
                    # ノードを再計算が必要な状態にする
                    pin.parent_node.is_dirty = True

    def _on_execute_pipeline(self) -> None:
        """パイプラインを実行する。"""
        if self.is_running:
            return
        
        self.is_running = True
        dpg.set_value(self.status_text, "パイプライン実行中...")
        
        try:
            # パイプラインの実行
            results = self.pipeline.execute()
            
            # 各ノードのUIを更新
            for node in self.pipeline.nodes:
                node.update_ui()
            
            dpg.set_value(self.status_text, "パイプライン実行完了")
        except Exception as e:
            dpg.set_value(self.status_text, f"エラー: {str(e)}")
        finally:
            self.is_running = False

    def _on_new(self) -> None:
        """新しいパイプラインを作成する。"""
        self._on_clear_all()
        dpg.set_value(self.status_text, "新しいパイプラインを作成しました")

    def _on_open(self) -> None:
        """パイプラインを開く。"""
        dpg.configure_item("file_dialog", callback=self._on_open_file_selected)
        dpg.show_item("file_dialog")

    def _on_save(self) -> None:
        """パイプラインを保存する。"""
        if self.pipeline.file_path:
            self.pipeline.save(self.pipeline.file_path)
            dpg.set_value(self.status_text, f"パイプラインを保存しました: {self.pipeline.file_path}")
        else:
            self._on_save_as()

    def _on_save_as(self) -> None:
        """パイプラインを名前を付けて保存する。"""
        dpg.configure_item("file_dialog", callback=self._on_save_file_selected)
        dpg.show_item("file_dialog")

    def _on_export_module(self) -> None:
        """パイプラインをPythonモジュールとしてエクスポートする。"""
        dpg.configure_item("file_dialog", callback=self._on_export_file_selected)
        dpg.show_item("file_dialog")

    def _on_open_file_selected(self, sender: int, app_data: Dict) -> None:
        """
        ファイル選択ダイアログでファイルが選択されたときのコールバック（開く）。

        Args:
            sender: 送信元のID
            app_data: ファイル情報
        """
        file_path = app_data["file_path_name"]
        try:
            # 既存のパイプラインをクリア
            self._on_clear_all()
            
            # パイプラインを読み込み
            self.pipeline.load(file_path)
            
            # ノードのUIを作成
            for node in self.pipeline.nodes:
                node.create_ui(self.node_editor_id)
            
            dpg.set_value(self.status_text, f"パイプラインを読み込みました: {file_path}")
        except Exception as e:
            dpg.set_value(self.status_text, f"読み込みエラー: {str(e)}")

    def _on_save_file_selected(self, sender: int, app_data: Dict) -> None:
        """
        ファイル選択ダイアログでファイルが選択されたときのコールバック（保存）。

        Args:
            sender: 送信元のID
            app_data: ファイル情報
        """
        file_path = app_data["file_path_name"]
        # 拡張子の確認と追加
        if not file_path.endswith(".json"):
            file_path += ".json"
        
        try:
            # パイプラインを保存
            self.pipeline.save(file_path)
            dpg.set_value(self.status_text, f"パイプラインを保存しました: {file_path}")
        except Exception as e:
            dpg.set_value(self.status_text, f"保存エラー: {str(e)}")

    def _on_export_file_selected(self, sender: int, app_data: Dict) -> None:
        """
        ファイル選択ダイアログでファイルが選択されたときのコールバック（エクスポート）。

        Args:
            sender: 送信元のID
            app_data: ファイル情報
        """
        file_path = app_data["file_path_name"]
        # 拡張子の確認と追加
        if not file_path.endswith(".py"):
            file_path += ".py"
        
        try:
            # パイプラインをエクスポート
            self.pipeline.export_as_module(file_path)
            dpg.set_value(self.status_text, f"モジュールをエクスポートしました: {file_path}")
        except Exception as e:
            dpg.set_value(self.status_text, f"エクスポートエラー: {str(e)}")

    def _on_file_selected(self, sender: int, app_data: Dict) -> None:
        """
        ファイル選択ダイアログでファイルが選択されたときの汎用コールバック。

        Args:
            sender: 送信元のID
            app_data: ファイル情報
        """
        # 特定のコールバックが設定されていない場合のデフォルト処理
        pass