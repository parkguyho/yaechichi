import numpy as np
import matplotlib.pyplot as plt

def generate_audio_signal(duration=1.0, fs=1000):
    """
    1초 동안 재생되는 깨끗한 오디오 신호(5Hz 주파수의 사인파)를 생성합니다.
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    # 순수한 음성 신호 모델링
    clean_signal = np.sin(2 * np.pi * 5 * t)
    return t, clean_signal

def add_noise(signal):
    """
    오디오 신호에 화이트 노이즈(에어컨/선풍기 소리 등)와 
    순간적인 팝 노이즈(틱 소리)를 추가합니다.
    """
    np.random.seed(42) # 재현 가능한 결과를 위한 시드 고정
    
    # 1. 화이트 노이즈 추가
    white_noise = np.random.normal(0, 0.4, len(signal))
    noisy_signal = signal + white_noise
    
    # 2. 순간적인 임펄스(팝) 노이즈 몇 개 추가 (특정 지점에 강한 튀는 값 부여)
    impulse_indices = [200, 450, 780]
    for idx in impulse_indices:
        noisy_signal[idx] += 2.5
        
    return noisy_signal

def moving_average_filter(data, window_size=5):
    """
    이동 평균 필터: 주변 값들의 평균을 구해 급격한 변동(노이즈)을 부드럽게 만듭니다.
    """
    filtered = np.copy(data)
    half_window = window_size // 2
    for i in range(half_window, len(data) - half_window):
        filtered[i] = np.mean(data[i - half_window : i + half_window + 1])
    return filtered

def moving_median_filter(data, window_size=5):
    """
    이동 중앙값 필터: 주변 값들을 정렬하여 중앙값을 취하므로, 
    순간적으로 튀는 팝 노이즈(Impulse Noise) 제거에 탁월합니다.
    """
    filtered = np.copy(data)
    half_window = window_size // 2
    for i in range(half_window, len(data) - half_window):
        window_values = data[i - half_window : i + half_window + 1]
        filtered[i] = np.median(window_values)
    return filtered

def main():
    print(">>> 오디오 신호 생성 및 노이즈 제거 시뮬레이션 시작...")
    
    # 1. 신호 및 노이즈 생성
    t, clean_signal = generate_audio_signal(duration=1.0, fs=1000)
    noisy_signal = add_noise(clean_signal)
    
    # 2. 필터 적용 (윈도우 크기 7)
    window_size = 7
    ma_filtered = moving_average_filter(noisy_signal, window_size=window_size)
    median_filtered = moving_median_filter(noisy_signal, window_size=window_size)
    
    # 3. 결과 시각화
    plt.figure(figsize=(14, 10))
    
    # 원본 vs 노이즈 낀 신호
    plt.subplot(3, 1, 1)
    plt.plot(t, noisy_signal, color='red', alpha=0.6, label='Noisy Audio (White + Impulse Noise)')
    plt.plot(t, clean_signal, color='blue', linestyle='--', linewidth=2, label='Original Clean Signal')
    plt.title('1. Before Filtering: Input Signal with Noise')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    # 이동 평균 필터 결과
    plt.subplot(3, 1, 2)
    plt.plot(t, ma_filtered, color='orange', label=f'Moving Average Filter (Window={window_size})')
    plt.plot(t, clean_signal, color='blue', linestyle='--', linewidth=2, label='Original Clean Signal')
    plt.title('2. After Moving Average Filter (Good for smooth noise)')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    # 이동 중앙값 필터 결과
    plt.subplot(3, 1, 3)
    plt.plot(t, median_filtered, color='green', label=f'Moving Median Filter (Window={window_size})')
    plt.plot(t, clean_signal, color='blue', linestyle='--', linewidth=2, label='Original Clean Signal')
    plt.title('3. After Moving Median Filter (Excellent for impulse/pop noise)')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    plt.tight_layout()
    print(">>> 시뮬레이션 완료! 그래프 창을 확인하세요.")
    plt.show()

if __name__ == "__main__":
    main()
