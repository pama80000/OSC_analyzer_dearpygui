#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
.trcファイル読み込みユーティリティモジュール。

Lecroy社のオシロスコープで生成される.trcバイナリファイルを読み込むための機能を提供します。
"""

import struct
import numpy as np
from typing import Dict, Tuple, Any, Optional


class TrcReader:
    """Lecroy社の.trcファイルを読み込むクラス。"""

    def __init__(self, file_path: str):
        """
        TrcReaderクラスのコンストラクタ。

        Args:
            file_path: 読み込む.trcファイルのパス
        """
        self.file_path = file_path
        self.header = {}  # ヘッダー情報
        self.data = None  # 波形データ
        self.time_array = None  # 時間軸データ

    def read(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        .trcファイルを読み込む。

        Returns:
            (時間軸データ, 波形データ, ヘッダー情報)のタプル
        """
        with open(self.file_path, 'rb') as f:
            # ヘッダー情報の読み込み
            self._read_header(f)
            
            # データ部分の読み込み
            self._read_data(f)
            
            # 時間軸の生成
            self._generate_time_array()
        
        return self.time_array, self.data, self.header

    def _read_header(self, file_obj) -> None:
        """
        ファイルからヘッダー情報を読み込む。

        Args:
            file_obj: オープンされたファイルオブジェクト
        """
        # ヘッダーの基本情報
        header_bytes = file_obj.read(346)
        
        # WAVEDESC文字列の確認
        wavedesc = header_bytes[0:8].decode('ascii', errors='ignore').strip()
        if wavedesc != "WAVEDESC":
            raise ValueError("無効な.trcファイル形式です。WAVEDESCマーカーが見つかりません。")
        
        # ヘッダー情報の解析
        template_name = header_bytes[16:24].decode('ascii', errors='ignore').strip()
        comm_type = struct.unpack('<H', header_bytes[32:34])[0]
        comm_order = struct.unpack('<H', header_bytes[34:36])[0]
        wave_descriptor = struct.unpack('<I', header_bytes[36:40])[0]
        user_text = struct.unpack('<I', header_bytes[40:44])[0]
        res_desc1 = struct.unpack('<I', header_bytes[44:48])[0]
        trig_time_array = struct.unpack('<I', header_bytes[48:52])[0]
        ris_time_array = struct.unpack('<I', header_bytes[52:56])[0]
        res_array1 = struct.unpack('<I', header_bytes[56:60])[0]
        wave_array_1 = struct.unpack('<I', header_bytes[60:64])[0]
        wave_array_2 = struct.unpack('<I', header_bytes[64:68])[0]
        res_array2 = struct.unpack('<I', header_bytes[68:72])[0]
        res_array3 = struct.unpack('<I', header_bytes[72:76])[0]
        
        # 波形フォーマット情報
        instrument_name = header_bytes[76:92].decode('ascii', errors='ignore').strip()
        instrument_number = struct.unpack('<I', header_bytes[92:96])[0]
        trace_label = header_bytes[96:112].decode('ascii', errors='ignore').strip()
        
        # 波形の時間情報
        wave_array_count = struct.unpack('<I', header_bytes[116:120])[0]
        first_point = struct.unpack('<I', header_bytes[124:128])[0]
        last_point = struct.unpack('<I', header_bytes[128:132])[0]
        first_valid_point = struct.unpack('<I', header_bytes[132:136])[0]
        last_valid_point = struct.unpack('<I', header_bytes[136:140])[0]
        
        # 垂直軸情報
        vertical_gain = struct.unpack('<f', header_bytes[156:160])[0]
        vertical_offset = struct.unpack('<f', header_bytes[160:164])[0]
        
        # 水平軸情報
        horiz_interval = struct.unpack('<f', header_bytes[176:180])[0]
        horiz_offset = struct.unpack('<d', header_bytes[180:188])[0]
        
        # データ形式
        data_type = struct.unpack('<H', header_bytes[240:242])[0]
        
        # ヘッダー情報の保存
        self.header = {
            'template_name': template_name,
            'comm_type': comm_type,
            'comm_order': comm_order,
            'wave_descriptor': wave_descriptor,
            'user_text': user_text,
            'res_desc1': res_desc1,
            'trig_time_array': trig_time_array,
            'ris_time_array': ris_time_array,
            'res_array1': res_array1,
            'wave_array_1': wave_array_1,
            'wave_array_2': wave_array_2,
            'res_array2': res_array2,
            'res_array3': res_array3,
            'instrument_name': instrument_name,
            'instrument_number': instrument_number,
            'trace_label': trace_label,
            'wave_array_count': wave_array_count,
            'first_point': first_point,
            'last_point': last_point,
            'first_valid_point': first_valid_point,
            'last_valid_point': last_valid_point,
            'vertical_gain': vertical_gain,
            'vertical_offset': vertical_offset,
            'horiz_interval': horiz_interval,
            'horiz_offset': horiz_offset,
            'data_type': data_type
        }

    def _read_data(self, file_obj) -> None:
        """
        ファイルからデータ部分を読み込む。

        Args:
            file_obj: オープンされたファイルオブジェクト
        """
        # ヘッダーサイズの計算
        header_size = (
            self.header['wave_descriptor'] +
            self.header['user_text'] +
            self.header['res_desc1']
        )
        
        # ファイルポインタをデータ部分に移動
        file_obj.seek(header_size)
        
        # データ形式に応じた読み込み
        data_type = self.header['data_type']
        count = self.header['wave_array_count']
        
        if data_type == 0:  # BYTE
            data_raw = np.frombuffer(file_obj.read(count), dtype=np.int8)
        elif data_type == 1:  # WORD
            data_raw = np.frombuffer(file_obj.read(count * 2), dtype=np.int16)
        elif data_type == 2:  # FLOAT
            data_raw = np.frombuffer(file_obj.read(count * 4), dtype=np.float32)
        elif data_type == 3:  # DOUBLE
            data_raw = np.frombuffer(file_obj.read(count * 8), dtype=np.float64)
        else:
            raise ValueError(f"未対応のデータ形式です: {data_type}")
        
        # 垂直軸のスケーリング
        vertical_gain = self.header['vertical_gain']
        vertical_offset = self.header['vertical_offset']
        
        # スケーリングされたデータの計算
        self.data = data_raw * vertical_gain - vertical_offset

    def _generate_time_array(self) -> None:
        """時間軸データを生成する。"""
        if self.data is None:
            return
        
        # 時間軸の情報
        horiz_interval = self.header['horiz_interval']
        horiz_offset = self.header['horiz_offset']
        count = len(self.data)
        
        # 時間軸データの生成
        self.time_array = np.arange(count) * horiz_interval + horiz_offset


def read_trc_file(file_path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    .trcファイルを読み込む便利関数。

    Args:
        file_path: 読み込む.trcファイルのパス

    Returns:
        (時間軸データ, 波形データ, ヘッダー情報)のタプル
    """
    reader = TrcReader(file_path)
    return reader.read()