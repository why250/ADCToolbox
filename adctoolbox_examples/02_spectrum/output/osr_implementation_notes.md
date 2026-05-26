# OSR 的实现方式

> 源码参考：`python/src/adctoolbox/spectrum/compute_spectrum.py` 和 `_bin_ranges.py`

## 核心结论

`osr` 参数**不改变信号、不做插值、不做重采样**。它只做一件事：

> **把 FFT 的"有效分析区间"从 `0 ~ Nyquist` 缩小为 `0 ~ Nyquist/OSR`。**

---

## 关键代码三处

### 1. 计算带内 bin 数

```python
# compute_spectrum.py, line 68
n_inband = rfft_inband_bin_count(N, osr)
```

```python
# _bin_ranges.py
def rfft_inband_bin_count(n_fft: int, osr: float = 1) -> int:
    rfft_len = n_fft // 2 + 1        # 全部 bin 数
    edge_bin = n_fft / (2 * osr)     # 截止 bin = Nyquist / OSR
    count = int(np.floor(edge_bin)) + 1
    return max(1, min(count, rfft_len))
```

**示例**（N=65536, Fs=100 MHz）：

| OSR | 截止 bin | 分析带宽 |
|-----|---------|---------|
| 1   | 32768   | 50 MHz（全带）|
| 2   | 16384   | 25 MHz |
| 4   | 8192    | 12.5 MHz |
| 10  | 3276    | 5 MHz |

---

### 2. 噪声积分只在带内做

```python
# compute_spectrum.py, lines 145-152
noi_sndr = float(np.sum(spec_sndr[:n_inband]))  # ← 只积分到 n_inband
sndr_dbc = 10 * np.log10(sig_linear / (noi_sndr + 1e-20))
```

带外（`n_inband` 以上）的 bin 直接不参与 SNR / SNDR 计算。

---

### 3. NSD 计算带宽也随 OSR 缩小

```python
# compute_spectrum.py, lines 174-178
nsd_dbfs_hz = noise_floor_dbfs - 10 * np.log10(fs / (2 * osr))
```

`fs / (2 * osr)` 是带内带宽，OSR 越大，带宽越窄，NSD 的参考带宽基准也跟着变。

---

## 直觉理解

```
OSR=1:  分析全部带宽 [0 ~ Fs/2]，带内噪声全部算进去
OSR=4:  只看 [0 ~ Fs/8]，带外噪声直接忽略不计

                  信号功率（不变）
SNR = ─────────────────────────────────────────────
        带内噪声功率（OSR 越大，积分区间越窄，噪声越少）
```

OSR 提升 SNR 的本质是：**用数学方法"假装"你只关心低频带内的噪声**，带外的噪声 bin 直接被切掉，不参与计算——就像在真实系统中用低通滤波器滤掉带外噪声一样，只是这里是在 FFT 后的功率谱上做截断。

### 理论增益

$$\Delta \text{SNR} = 10 \cdot \log_{10}(\text{OSR}) \quad [\text{dB}]$$

| OSR | 理论 SNR 增益 |
|-----|-------------|
| 1   | 0 dB        |
| 2   | +3.0 dB     |
| 4   | +6.0 dB     |
| 10  | +10.0 dB    |

---

## 总结

| 参数 | 是否改变信号样本 | 作用机制 |
|------|--------------|---------|
| FFT 长度 N | 是（需要更多采样点） | 提升频率分辨率，降低每 bin 噪声底（dBFS），但 NSD 和 SNR 不变 |
| OSR | **否** | 缩小 FFT 功率谱的积分范围，排除带外噪声，提升带内 SNR |
