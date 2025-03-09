#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
パイプライン管理モジュール。

ノードの管理、接続の管理、パイプラインの保存・読み込み、実行などの機能を提供します。
"""

import logging
import os
import json
import importlib.util
from typing import Dict, List, Any, Optional, Tuple, Type, Set
import numpy as np

# 循環インポートを避けるため、型ヒントのみをインポート
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.nodes.base_node import Node


class Pipeline:
    """パイプライン全体を管理するクラス。"""

    _instance = None  # シングルトンインスタンス

    @classmethod
    def get_instance(cls) -> 'Pipeline':
        """
        Pipelineのシングルトンインスタンスを取得する。

        Returns:
            Pipelineインスタンス
        """
        if cls._instance is None:
            cls._instance = Pipeline()
        return cls._instance

    def __init__(self):
        """Pipelineクラスのコンストラクタ。"""
        self.nodes = []  # ノードのリスト
        self.file_path = None  # 保存されているファイルパス
        self.node_registry = {}  # ノードタイプの登録辞書 {クラス名: クラス}

    def register_node_type(self, node_class: Type['Node']) -> None:
        """
        ノードタイプを登録する。

        Args:
            node_class: 登録するノードクラス
        """
        self.node_registry[node_class.__name__] = node_class

    def add_node(self, node: 'Node') -> None:
        """
        ノードを追加する。

        Args:
            node: 追加するノード
        """
        self.nodes.append(node)

    def remove_node(self, node: 'Node') -> None:
        """
        ノードを削除する。

        Args:
            node: 削除するノード
        """
        # ノードに接続されているすべての接続を削除
        for input_pin in node.input_pins.values():
            if input_pin.connected_to:
                self.disconnect_pins(input_pin.connected_to, input_pin.id)
        
        for output_pin in node.output_pins.values():
            if output_pin.connected_to:
                self.disconnect_pins(output_pin.id, output_pin.connected_to)
        
        # ノードをリストから削除
        if node in self.nodes:
            self.nodes.remove(node)

    def clear(self) -> None:
        """パイプラインをクリアする。"""
        self.nodes = []
        self.file_path = None

    def connect_pins(self, from_pin_id: str, to_pin_id: str) -> bool:
        """
        ピン同士を接続する。

        Args:
            from_pin_id: 出力ピンのID
            to_pin_id: 入力ピンのID

        Returns:
            接続が成功したかどうか
        """
        # ピンを見つける
        from_pin = None
        to_pin = None
        
        for node in self.nodes:
            for pin in node.output_pins.values():
                if pin.id == from_pin_id:
                    from_pin = pin
                    break
            
            for pin in node.input_pins.values():
                if pin.id == to_pin_id:
                    to_pin = pin
                    break
        
        # ピンが見つからない場合
        if not from_pin or not to_pin:
            return False
        
        # 既存の接続を解除
        if to_pin.connected_to:
            self.disconnect_pins(to_pin.connected_to, to_pin.id)
        
        # 新しい接続を作成
        from_pin.connected_to = to_pin.id
        to_pin.connected_to = from_pin.id
        
        # 接続先ノードを再計算が必要な状態にする
        to_pin.parent_node.is_dirty = True
        
        return True

    def disconnect_pins(self, from_pin_id: str, to_pin_id: str) -> bool:
        """
        ピン間の接続を解除する。

        Args:
            from_pin_id: 出力ピンのID
            to_pin_id: 入力ピンのID

        Returns:
            切断が成功したかどうか
        """
        # ピンを見つける
        from_pin = None
        to_pin = None
        
        for node in self.nodes:
            for pin in node.output_pins.values():
                if pin.id == from_pin_id:
                    from_pin = pin
                    break
            
            for pin in node.input_pins.values():
                if pin.id == to_pin_id:
                    to_pin = pin
                    break
        
        # ピンが見つからない場合
        if not from_pin or not to_pin:
            return False
        
        # 接続を解除
        from_pin.connected_to = None
        to_pin.connected_to = None
        
        # 接続先ノードを再計算が必要な状態にする
        to_pin.parent_node.is_dirty = True
        
        return True

    def get_execution_order(self) -> List['Node']:
        """
        ノードの実行順序を決定する（トポロジカルソート）。

        Returns:
            実行順序に並べられたノードのリスト
        """
        # 依存関係グラフの構築
        graph = {}
        for node in self.nodes:
            graph[node.id] = set()
        
        # 依存関係の追加
        for node in self.nodes:
            for input_pin in node.input_pins.values():
                if input_pin.connected_to:
                    # 接続元のノードを見つける
                    for other_node in self.nodes:
                        for output_pin in other_node.output_pins.values():
                            if output_pin.id == input_pin.connected_to:
                                # このノードは他のノードに依存している
                                graph[node.id].add(other_node.id)
        
        # トポロジカルソート
        visited = set()
        temp = set()
        order = []
        
        def visit(node_id):
            if node_id in temp:
                # 循環依存関係が検出された
                raise ValueError("循環依存関係が検出されました")
            
            if node_id not in visited:
                temp.add(node_id)
                
                # 依存するノードを先に処理
                for dep_id in graph[node_id]:
                    visit(dep_id)
                
                temp.remove(node_id)
                visited.add(node_id)
                
                # このノードを実行順序に追加
                for node in self.nodes:
                    if node.id == node_id:
                        order.append(node)
                        break
        
        # すべてのノードを処理
        for node in self.nodes:
            if node.id not in visited:
                visit(node.id)
        
        # 実行順序のログ出力（依存関係の順序）
        logging.debug("Execution order: " + str([node.__class__.__name__ for node in order]))
        return order

    def execute(self) -> Dict[str, Any]:
        """
        パイプラインを実行する。

        Returns:
            各ノードの出力結果の辞書
        """
        # 実行順序の決定
        try:
            execution_order = self.get_execution_order()
        except ValueError as e:
            raise RuntimeError(f"パイプラインの実行エラー: {str(e)}")
        
        # 各ノードを順番に実行
        results = {}
        for node in execution_order:
            logging.debug(f"Processing node '{node.title}' with ID '{node.id}'.")
            try:
                node_results = node.process()
                logging.debug(f"Node '{node.title}' result: {node_results}")
                results[node.id] = node_results
            except Exception as e:
                logging.error(f"Error processing node '{node.title}' (ID: {node.id}): {str(e)}")
                raise RuntimeError(f"ノード '{node.title}' の実行エラー: {str(e)}")
        
        return results

    def save(self, file_path: str) -> None:
        """
        パイプラインをJSONファイルに保存する。

        Args:
            file_path: 保存先のファイルパス
        """
        # パイプラインの状態をシリアライズ
        data = {
            "nodes": [node.serialize() for node in self.nodes],
            "connections": []
        }
        
        # 接続情報の収集
        for node in self.nodes:
            for input_pin in node.input_pins.values():
                if input_pin.connected_to:
                    data["connections"].append({
                        "from_pin": input_pin.connected_to,
                        "to_pin": input_pin.id
                    })
        
        # JSONファイルに保存
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        self.file_path = file_path

    def load(self, file_path: str) -> None:
        """
        JSONファイルからパイプラインを読み込む。

        Args:
            file_path: 読み込むファイルのパス
        """
        # JSONファイルの読み込み
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # パイプラインをクリア
        self.clear()
        
        # ノードの復元
        node_map = {}  # 一時的なノードIDマッピング
        
        for node_data in data["nodes"]:
            node_type = node_data["type"]
            
            # ノードクラスの取得
            if node_type not in self.node_registry:
                raise ValueError(f"未登録のノードタイプ: {node_type}")
            
            node_class = self.node_registry[node_type]
            
            # ノードの復元
            node = node_class.deserialize(node_data)
            self.add_node(node)
            
            # IDマッピングの更新
            node_map[node.id] = node
        
        # 接続の復元
        for conn_data in data["connections"]:
            from_pin_id = conn_data["from_pin"]
            to_pin_id = conn_data["to_pin"]
            
            self.connect_pins(from_pin_id, to_pin_id)
        
        self.file_path = file_path

    def export_as_module(self, file_path: str) -> None:
        """
        パイプラインをPythonモジュールとしてエクスポートする。

        Args:
            file_path: エクスポート先のファイルパス
        """
        # 実行順序の決定
        try:
            execution_order = self.get_execution_order()
        except ValueError as e:
            raise RuntimeError(f"パイプラインのエクスポートエラー: {str(e)}")
        
        # モジュールのコード生成
        code = [
            "#!/usr/bin/env python3",
            "# -*- coding: utf-8 -*-",
            "",
            "\"\"\"",
            "自動生成されたオシロスコープデータ処理パイプライン。",
            "",
            "このモジュールは、オシロスコープデータ処理フローから自動生成されました。",
            "\"\"\"",
            "",
            "import numpy as np",
            "import pandas as pd",
            "from scipy import signal",
            "from scipy.fft import fft, ifft, fftfreq",
            "",
            ""
        ]
        
        # 各ノードの処理関数を生成
        for i, node in enumerate(execution_order):
            node_type = node.__class__.__name__
            func_name = f"process_{i}_{node_type.lower()}"
            
            code.append(f"def {func_name}(data=None, **kwargs):")
            code.append(f"    \"\"\"")
            code.append(f"    {node.title} の処理を実行する。")
            code.append(f"    \"\"\"")
            
            # パラメータの取得
            param_lines = []
            for name, param in node.parameters.items():
                value = param["value"]
                if isinstance(value, str):
                    value_str = f'"{value}"'
                else:
                    value_str = str(value)
                param_lines.append(f"    {name} = kwargs.get('{name}', {value_str})")
            
            if param_lines:
                code.extend(param_lines)
                code.append("")
            
            # 処理ロジックの生成（ノードタイプに応じて）
            if node_type == "DataLoadNode":
                code.append("    # データ読み込み処理")
                code.append("    if 'file_path' in kwargs:")
                code.append("        file_path = kwargs['file_path']")
                code.append("        if file_path.endswith('.csv'):")
                code.append("            data = pd.read_csv(file_path).values")
                code.append("        elif file_path.endswith('.trc'):")
                code.append("            # .trcファイルの読み込み処理（実装が必要）")
                code.append("            pass")
                code.append("    return data")
            
            elif node_type == "TimeSliceNode":
                code.append("    # 時間指定切り出し処理")
                code.append("    if data is None:")
                code.append("        return None")
                code.append("    start_idx = int(kwargs.get('start_time', 0) * kwargs.get('sample_rate', 1))")
                code.append("    end_idx = int(kwargs.get('end_time', len(data)) * kwargs.get('sample_rate', 1))")
                code.append("    return data[start_idx:end_idx]")
            
            elif node_type == "FFTNode":
                code.append("    # FFT処理")
                code.append("    if data is None:")
                code.append("        return None")
                code.append("    window = kwargs.get('window', 'hann')")
                code.append("    if window == 'hann':")
                code.append("        win = np.hanning(len(data))")
                code.append("    elif window == 'hamming':")
                code.append("        win = np.hamming(len(data))")
                code.append("    else:")
                code.append("        win = np.ones(len(data))")
                code.append("    return np.abs(fft(data * win))")
            
            elif node_type == "FrequencyFilterNode":
                code.append("    # 周波数フィルター処理")
                code.append("    if data is None:")
                code.append("        return None")
                code.append("    filter_type = kwargs.get('filter_type', 'lowpass')")
                code.append("    cutoff = kwargs.get('cutoff_freq', 1000)")
                code.append("    order = kwargs.get('order', 4)")
                code.append("    sample_rate = kwargs.get('sample_rate', 1000)")
                code.append("    nyq = 0.5 * sample_rate")
                code.append("    normal_cutoff = cutoff / nyq")
                code.append("    b, a = signal.butter(order, normal_cutoff, btype=filter_type, analog=False)")
                code.append("    return signal.filtfilt(b, a, data)")
            
            elif node_type == "FeatureExtractionNode":
                code.append("    # 特徴量抽出処理")
                code.append("    if data is None:")
                code.append("        return None")
                code.append("    features = {}")
                code.append("    features['max'] = np.max(data)")
                code.append("    features['min'] = np.min(data)")
                code.append("    features['mean'] = np.mean(data)")
                code.append("    features['std'] = np.std(data)")
                code.append("    features['rms'] = np.sqrt(np.mean(np.square(data)))")
                code.append("    return features")
            
            elif node_type == "EnvelopeNode":
                code.append("    # 包絡線抽出処理")
                code.append("    if data is None:")
                code.append("        return None")
                code.append("    analytic_signal = signal.hilbert(data)")
                code.append("    amplitude_envelope = np.abs(analytic_signal)")
                code.append("    return amplitude_envelope")
            
            elif node_type == "CustomPythonNode":
                code.append("    # カスタムPython処理")
                code.append("    if data is None:")
                code.append("        return None")
                code.append("    # ユーザー定義のコード（実装が必要）")
                code.append("    return data")
            
            elif node_type == "WaveformDisplayNode":
                code.append("    # 波形表示処理（実際の表示は行わず、データをそのまま返す）")
                code.append("    return data")
            
            else:
                code.append("    # 未知のノードタイプ")
                code.append("    return data")
            
            code.append("")
        
        # メイン処理関数
        code.append("def process_pipeline(input_file=None, **kwargs):")
        code.append("    \"\"\"")
        code.append("    パイプライン全体の処理を実行する。")
        code.append("")
        code.append("    Args:")
        code.append("        input_file: 入力ファイルのパス")
        code.append("        **kwargs: 各ノードのパラメータ")
        code.append("")
        code.append("    Returns:")
        code.append("        処理結果")
        code.append("    \"\"\"")
        code.append("    # 初期データ")
        code.append("    data = None")
        code.append("    if input_file:")
        code.append("        kwargs['file_path'] = input_file")
        code.append("")
        
        # 各ノードの処理を順番に呼び出し
        for i, node in enumerate(execution_order):
            node_type = node.__class__.__name__
            func_name = f"process_{i}_{node_type.lower()}"
            code.append(f"    # {node.title}")
            code.append(f"    data = {func_name}(data, **kwargs)")
            code.append("")
        
        code.append("    return data")
        code.append("")
        
        # メイン実行部分
        code.append("if __name__ == \"__main__\":")
        code.append("    import sys")
        code.append("    import matplotlib.pyplot as plt")
        code.append("")
        code.append("    if len(sys.argv) > 1:")
        code.append("        input_file = sys.argv[1]")
        code.append("    else:")
        code.append("        input_file = None")
        code.append("")
        code.append("    # パイプラインの実行")
        code.append("    result = process_pipeline(input_file)")
        code.append("")
        code.append("    # 結果の表示")
        code.append("    if isinstance(result, np.ndarray):")
        code.append("        plt.figure(figsize=(10, 6))")
        code.append("        plt.plot(result)")
        code.append("        plt.title('Processing Result')")
        code.append("        plt.grid(True)")
        code.append("        plt.show()")
        code.append("    else:")
        code.append("        print(result)")
        
        # ファイルに書き込み
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(code))