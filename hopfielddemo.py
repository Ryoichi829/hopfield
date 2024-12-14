import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))
    
    def train(self, pattern):
        # パターンを±1に変換
        pattern_1d = pattern.flatten() * 2 - 1
        # 重み行列の更新（自己結合は除外）
        self.weights += np.outer(pattern_1d, pattern_1d)
        np.fill_diagonal(self.weights, 0)
    
    def update(self, pattern, max_iterations=20):
        current_pattern = pattern.flatten() * 2 - 1
        
        for _ in range(max_iterations):
            old_pattern = current_pattern.copy()
            # ランダムな順序で更新
            for i in np.random.permutation(self.size):
                current_pattern[i] = np.sign(self.weights[i] @ current_pattern)
            
            # 収束チェック
            if np.array_equal(old_pattern, current_pattern):
                break
        
        return ((current_pattern + 1) / 2).reshape(int(np.sqrt(self.size)), -1)

def create_digit_3():
    # 8x8の「3」のパターン
    digit = np.array([
        [0,1,1,1,1,1,0,0],
        [0,0,0,0,0,1,0,0],
        [0,0,0,0,0,1,0,0],
        [0,1,1,1,1,1,0,0],
        [0,0,0,0,0,1,0,0],
        [0,0,0,0,0,1,0,0],
        [0,1,1,1,1,1,0,0],
        [0,0,0,0,0,0,0,0]
    ])
    return digit

def add_noise(pattern, noise_level):
    noisy_pattern = pattern.copy()
    mask = np.random.random(pattern.shape) < noise_level
    noisy_pattern[mask] = 1 - noisy_pattern[mask]
    return noisy_pattern

def plot_pattern(pattern, title=""):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(pattern, cmap='binary')
    ax.grid(True)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)
    return fig

def main():
    st.set_page_config(page_title="ホップフィールドネットワークデモ", layout="wide")
    
    st.title("ホップフィールドネットワークによる画像の復元")
    st.write("2024年ノーベル物理学賞受賞 - John Hopfield の研究を体験してみよう")

    # サイドバーに説明を追加
    st.sidebar.title("学習ステップ")
    st.sidebar.write("""
    1. まず、数字の「3」のパターンを見てみましょう
    2. ノイズ（汚れ）を加えてみましょう
    3. ホップフィールドネットワークで復元してみましょう
    """)
    
    # メインの処理
    step = st.radio(
        "学習ステップを選択してください：",
        ["1. 元のパターンを見る", "2. ノイズを加える", "3. パターンを復元する"]
    )
    
    # ホップフィールドネットワークの初期化
    digit = create_digit_3()
    network = HopfieldNetwork(64)
    network.train(digit)
    
    if step == "1. 元のパターンを見る":
        st.write("""
        これは数字の「3」のパターンです。
        8×8のグリッドで表現されています。
        このパターンをホップフィールドネットワークに「記憶」させます。
        """)
        st.pyplot(plot_pattern(digit, "元のパターン"))
        
    elif step == "2. ノイズを加える":
        noise_level = st.slider("ノイズの量を選択してください（%）:", 0, 100, 30) / 100.0
        noisy_digit = add_noise(digit, noise_level)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("元のパターン")
            st.pyplot(plot_pattern(digit))
        with col2:
            st.write("ノイズを加えたパターン")
            st.pyplot(plot_pattern(noisy_digit))
            
        st.write(f"""
        ノイズレベル: {noise_level*100:.1f}%
        画像にノイズ（汚れ）を加えると、元のパターンが分かりにくくなります。
        """)
        
    else:  # step 3
        noise_level = st.slider("ノイズの量を選択してください（%）:", 0, 100, 30) / 100.0
        noisy_digit = add_noise(digit, noise_level)
        
        if st.button("パターンを復元"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("元のパターン")
                st.pyplot(plot_pattern(digit))
                
            with col2:
                st.write("ノイズを加えたパターン")
                st.pyplot(plot_pattern(noisy_digit))
            
            with st.spinner("復元中..."):
                time.sleep(1)  # アニメーション効果のため
                restored_digit = network.update(noisy_digit)
                
            with col3:
                st.write("復元されたパターン")
                st.pyplot(plot_pattern(restored_digit))
                
            st.success("""
            復元完了！
            
            ホップフィールドネットワークは、記憶したパターンを使って
            ノイズの多い画像から元のパターンを「思い出す」ことができました。
            
            これは人間の脳の「連想記憶」の仕組みを模倣しています。
            """)

    # 追加の説明
    st.markdown("""
    ### 解説
    
    ホップフィールドネットワークは、以下のような特徴を持っています：
    
    1. **パターンの記憶**: ネットワークは与えられたパターンを重み行列として記憶します
    2. **エネルギー最小化**: ノイズの多いパターンから、記憶したパターンを想起する際に、システムのエネルギーが最小になる状態を探します
    3. **並列処理**: すべてのニューロンが同時に情報を処理します（人間の脳と同じように）
    
    このデモでは簡単な8×8のパターンを使用していますが、
    同じ原理で、より複雑なパターンの記憶と想起も可能です。
    """)

if __name__ == "__main__":
    main()
