# hopfield3.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from PIL import Image
import time

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))
    
    def train(self, pattern):
        pattern_1d = pattern.flatten() * 2 - 1
        self.weights += np.outer(pattern_1d, pattern_1d)
        np.fill_diagonal(self.weights, 0)
    
    def update_single_step(self, current_pattern):
        # 1ステップの更新
        pattern_1d = current_pattern.flatten() * 2 - 1
        idx = np.random.randint(0, self.size)
        pattern_1d[idx] = np.sign(self.weights[idx] @ pattern_1d)
        return ((pattern_1d + 1) / 2).reshape(int(np.sqrt(self.size)), -1)

def create_digit_3():
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
    plt.close()
    return fig

def main():
    st.set_page_config(page_title="ホップフィールドネットワークデモ", layout="wide")
    
    st.title("ホップフィールドネットワークによる画像の復元")
    st.write("2024年ノーベル物理学賞受賞 - John Hopfield 氏の研究を体験してみよう")

    if 'network' not in st.session_state:
        st.session_state.network = HopfieldNetwork(64)
        st.session_state.network.train(create_digit_3())
    
    if 'current_pattern' not in st.session_state:
        st.session_state.current_pattern = None
    
    if 'animation_running' not in st.session_state:
        st.session_state.animation_running = False

    # サイドバーに説明を追加
    st.sidebar.title("学習ステップ")
    st.sidebar.write("""
    1. まず、数字の「3」のパターンを見てみましょう
    2. ノイズ（汚れ）を加えてみましょう
    3. ホップフィールドネットワークで復元してみましょう
    """)
    
    step = st.radio(
        "学習ステップを選択してください：",
        ["1. 元のパターンを見る", "2. ノイズを加える", "3. パターンを復元する"]
    )
    
    digit = create_digit_3()
    
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
        st.session_state.noisy_pattern = noisy_digit  # ノイズのあるパターンを保存

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
        # ノイズレベルとノイズを加えたパターンを取得
        noise_level = st.slider("ノイズの量を選択してください（%）:", 0, 100, 30) / 100.0
        noisy_digit = st.session_state.get("noisy_pattern", add_noise(digit, noise_level))  # 保存済みパターンを取得

        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("元のパターン")
            st.pyplot(plot_pattern(digit))
            
        with col2:
            st.write("ノイズを加えたパターン")
            st.pyplot(plot_pattern(noisy_digit))
        
        with col3:
            st.write("復元過程")
            restore_placeholder = st.empty()
            progress_bar = st.progress(0)

        start_button = st.button("パターンを復元")
        stop_button = st.button("停止")

        if start_button:
            st.session_state.animation_running = True
            st.session_state.current_pattern = noisy_digit.copy()
            
        if stop_button:
            st.session_state.animation_running = False

        # アニメーション処理
        if st.session_state.animation_running:
            iteration = 0
            max_iterations = 200  # 最大イテレーション数

            while iteration < max_iterations and st.session_state.animation_running:
                # パターンの更新
                st.session_state.current_pattern = st.session_state.network.update_single_step(
                    st.session_state.current_pattern
                )
                
                # 表示の更新
                restore_placeholder.pyplot(
                    plot_pattern(st.session_state.current_pattern, "復元中のパターン")
                )
                
                # プログレスバーの更新
                progress = (iteration + 1) / max_iterations
                progress_bar.progress(progress, f"復元中... {int(progress * 100)}%")
                
                # アニメーションの速度調整
                time.sleep(0.1)  # 100ミリ秒待機
                
                iteration += 1

                # 最後のイテレーションで完了メッセージを表示
                if iteration == max_iterations:
                    progress_bar.progress(1.0, "復元完了!")
                    st.session_state.animation_running = False
                    st.success("""
                    復元完了！
                    
                    ホップフィールドネットワークは、記憶したパターンを使って
                    ノイズの多い画像から元のパターンを「思い出す」ことができました。
                    
                    これは人間の脳の「連想記憶」の仕組みを模倣しています。
                    """)

    # 追加の解説
    with st.expander("ホップフィールドネットワークの解説"):
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
