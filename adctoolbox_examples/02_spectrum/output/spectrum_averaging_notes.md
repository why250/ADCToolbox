# 频谱平均方法：Power Averaging vs Coherent Averaging

> 源码参考：`python/src/adctoolbox/spectrum/_spectrum_averaging.py`
> 示例脚本：`adctoolbox_examples/02_spectrum/exp_s07_spectrum_averaging.py`

---

## 1. Power Spectrum Averaging（功率谱平均）

**做法：** 对每次 run 分别做 FFT，取模平方得功率谱，再对多次 run 的功率谱取均值。

```python
fft_matrix = np.fft.rfft(data_processed, axis=1)        # 每行一次 run 的 FFT
spectrum_power = np.mean(np.abs(fft_matrix)**2, axis=0)  # 功率谱均值
```

**关键特点：**

- 相位信息在取模平方后**完全丢失**，不同 run 之间无需相位对齐
- 噪声的**期望值（均值）不变** → 噪声底高度不变
- 噪声的**方差降低**（∝ 1/M）→ 视觉上噪声底更平滑
- **SNR 不随 run 数增加而提升**（0 dB 增益）

> ⚠️ 常见误解：Power Averaging 能降低噪声底。实际上它只是让噪声底"看起来更平"，并没有真正降低噪声功率。

---

## 2. Coherent Spectrum Averaging（相干谱平均）

**做法：** 对每次 run 做完整复数 FFT，先将每次 run 的**基频相位对齐到第一次 run**，再对复数谱累加平均，最后取模平方。

```python
# 记录第一次 run 的基频相位
original_fundamental_phase = np.angle(fft_data[fundamental_bin])

# 每次 run 做相位对齐后累加
fft_aligned = _align_spectrum_phase(fft_data, fundamental_bin, ...)
spec_coherent_sum += fft_aligned
```

**关键特点：**

- 信号各 run 相位一致 → 复数叠加，幅度 ∝ M，功率 ∝ M²
- 噪声各 run 随机相位 → 部分抵消，幅度 ∝ √M，功率 ∝ M
- SNR 增益 = M² / M = M → 真实 SNR 提升

$$\Delta \text{SNR} = 10 \cdot \log_{10}(M) \quad [\text{dB}]$$

| 次数 M | SNR 理论增益 |
|--------|------------|
| 1      | 0 dB       |
| 10     | +10 dB     |
| 100    | +20 dB     |

---

## 3. 两者对比

| | Power Averaging | Coherent Averaging |
|---|---|---|
| FFT 操作 | `\|FFT\|²` 再取均值 | 复数 FFT 对齐后取均值，再取 `\|·\|²` |
| 相位信息 | 丢弃 | 保留并对齐 |
| 噪声底均值 | **不变** | **真实下降**（∝ 1/M） |
| 噪声底方差 | 降低（∝ 1/M）| 降低（∝ 1/M） |
| SNR 增益 | **0 dB（无增益）** | **+10·log₁₀(M) dB** |
| 对相位的要求 | 无要求 | 需要相干采样，能定位基频做相位对齐 |
| 适用场景 | 视觉平滑，快速粗略分析 | 需要真实 SNR 提升的精确测量 |

---

## 4. 直觉类比

| 方法 | 类比 |
|---|---|
| Power Averaging | 拍多张照片后**逐像素取平均亮度** → 画面颗粒感轻微降低，整体曝光不变 |
| Coherent Averaging | 拍多张照片后**精确对齐再叠加** → 信号累积增强，随机噪点相消，信噪比真正提升 |

---

## 5. 示例脚本结论（exp_s07 输出）

```
1. Power Averaging:
   - SNR remains constant regardless of number of runs
   - Only smoothens the noise floor visually (reduces variance)
   - Does NOT provide true processing gain

2. Coherent Averaging:
   - SNR improves by ~10 dB for 10 runs, ~20 dB for 100 runs
   - Theoretical gain: 10*log10(N_runs) dB
   - Provides true processing gain through phase coherence
```
