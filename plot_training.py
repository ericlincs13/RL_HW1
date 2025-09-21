#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training Progress Visualization Script for 2048 TDL
生成訓練過程的折線圖，縱軸為平均分數，橫軸為 episode
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def plot_training_progress(csv_file='training_data.csv', output_file='training_progress.png'):
    """
    讀取訓練資料並生成折線圖
    
    Args:
        csv_file: 包含訓練資料的 CSV 檔案路径
        output_file: 輸出圖片的檔案名
    """
    
    # 檢查檔案是否存在
    if not os.path.exists(csv_file):
        print(f"錯誤：找不到檔案 {csv_file}")
        print("請確保已經運行 C++ 程式並生成了統計資料。")
        return False
    
    try:
        # 讀取 CSV 資料
        data = pd.read_csv(csv_file)
        
        # 檢查資料格式
        if 'Episode' not in data.columns or 'Average_Score' not in data.columns:
            print("錯誤：CSV 檔案格式不正確")
            print("需要包含 'Episode' 和 'Average_Score' 欄位")
            return False
        
        # 設定中文字體（如果系統支援）
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 創建圖表
        plt.figure(figsize=(12, 8))
        
        # 繪製折線圖
        plt.plot(data['Episode'], data['Average_Score'], 
                linewidth=2, color='blue', marker='o', markersize=4)
        
        # 設定標題和標籤
        plt.title('2048 訓練過程 - 平均分數變化', fontsize=16, fontweight='bold')
        plt.xlabel('Episode', fontsize=14)
        plt.ylabel('平均分數', fontsize=14)
        
        # 設定 X 軸刻度（每 20000 為一個單位）
        max_episode = data['Episode'].max()
        x_ticks = np.arange(0, max_episode + 20000, 20000)
        plt.xticks(x_ticks, rotation=45)
        
        # 設定 Y 軸刻度（根據數據範圍自動調整）
        y_min = data['Average_Score'].min()
        y_max = data['Average_Score'].max()
        y_range = y_max - y_min
        y_ticks = np.arange(int(y_min - y_range * 0.1), 
                           int(y_max + y_range * 0.1), 
                           max(1, int(y_range / 10)))
        plt.yticks(y_ticks)
        
        # 添加網格
        plt.grid(True, alpha=0.3)
        
        # 調整佈局
        plt.tight_layout()
        
        # 儲存圖片
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"折線圖已儲存為: {output_file}")
        
        # 顯示圖表（可選）
        plt.show()
        
        # 顯示統計資訊
        print(f"\n訓練統計資訊:")
        print(f"總 Episode 數: {data['Episode'].max()}")
        print(f"最低平均分數: {data['Average_Score'].min():.2f}")
        print(f"最高平均分數: {data['Average_Score'].max():.2f}")
        print(f"最終平均分數: {data['Average_Score'].iloc[-1]:.2f}")
        
        return True
        
    except Exception as e:
        print(f"錯誤：無法處理資料 - {e}")
        return False

def main():
    """主函數"""
    csv_file = 'training_data.csv'
    output_file = 'training_progress.png'
    
    # 檢查命令行參數
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    print("2048 訓練進度視覺化工具")
    print("=" * 40)
    print(f"輸入檔案: {csv_file}")
    print(f"輸出檔案: {output_file}")
    print("=" * 40)
    
    success = plot_training_progress(csv_file, output_file)
    
    if success:
        print("處理完成！")
    else:
        print("處理失敗，請檢查錯誤訊息。")
        sys.exit(1)

if __name__ == "__main__":
    main()
