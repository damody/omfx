"""
生成三種遊戲音效的 WAV 檔案：
  button_click.wav   — 輕快的 UI 點擊音
  tower_place.wav    — 放置英雄/塔的落地聲
  cookie_crunch.wav  — 餅乾碎裂聲（敵人消滅）
"""
import wave, struct, math, random, os

RATE = 44100

def write_wav(filename, samples):
    with wave.open(filename, 'w') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(RATE)
        data = struct.pack(f'<{len(samples)}h', *[int(max(-32767, min(32767, s))) for s in samples])
        f.writeframes(data)
    print(f"OK: {filename} ({len(samples)/RATE*1000:.0f} ms)")

def envelope(samples, attack_s=0.005, decay_s=0.03, sustain=0.7, release_s=0.05):
    n = len(samples)
    att = int(attack_s * RATE)
    dec = int(decay_s * RATE)
    rel = int(release_s * RATE)
    out = []
    for i, s in enumerate(samples):
        if i < att:
            gain = i / att
        elif i < att + dec:
            gain = 1.0 - (1.0 - sustain) * (i - att) / dec
        elif i < n - rel:
            gain = sustain
        else:
            gain = sustain * (n - i) / rel
        out.append(s * gain)
    return out

# ── 1. button_click.wav ─ 輕快 UI 點擊（50ms 高頻 sine）──────────────────────
dur = 0.055
t = [i / RATE for i in range(int(dur * RATE))]
samples = [20000 * math.sin(2 * math.pi * 1200 * ti) for ti in t]
samples = envelope(samples, attack_s=0.002, decay_s=0.015, sustain=0.0, release_s=0.035)
write_wav(os.path.join(os.path.dirname(__file__), "button_click.wav"), samples)

# ── 2. tower_place.wav ─ 放置英雄/塔（200ms 低頻衝擊 + harmonic）─────────────
dur = 0.22
n = int(dur * RATE)
samples = []
for i in range(n):
    ti = i / RATE
    # 低頻基音 + 倍頻（模擬「咚」聲）
    v  = 0.7 * math.sin(2 * math.pi * 180 * ti)
    v += 0.3 * math.sin(2 * math.pi * 360 * ti)
    v += 0.15 * math.sin(2 * math.pi * 540 * ti)
    # 快速指數衰減
    v *= math.exp(-ti * 18)
    samples.append(28000 * v)
write_wav(os.path.join(os.path.dirname(__file__), "tower_place.wav"), samples)

# ── 3. cookie_crunch.wav ─ 餅乾碎裂（多層次 crispy 碎裂聲）─────────────────
# 設計：初始清脆一聲 + 碎屑翻滾 + 溫潤低頻底
random.seed(99)
dur = 0.22
n = int(dur * RATE)

# 簡單 one-pole lowpass（模擬帶通，去掉最刺耳的超高頻）
def lowpass(samples, coef=0.35):
    out, prev = [], 0.0
    for s in samples:
        prev = prev * (1 - coef) + s * coef
        out.append(prev)
    return out

# 五個獨立碎裂脈衝（時間偏移、個別衰減速率、個別音量）
bursts = [
    # (onset_s, decay_rate, gain)
    (0.000, 90,  1.00),   # 主裂縫：最響
    (0.012, 120, 0.70),   # 第一塊碎片
    (0.028, 150, 0.50),   # 第二塊
    (0.052, 180, 0.35),   # 細碎屑
    (0.085, 200, 0.20),   # 尾音消散
]

raw = [0.0] * n
for onset, decay, gain in bursts:
    onset_i = int(onset * RATE)
    for i in range(onset_i, n):
        ti = (i - onset_i) / RATE
        noise = random.uniform(-1.0, 1.0)
        raw[i] += gain * noise * math.exp(-ti * decay)

# 加一點低頻溫潤感（短促 thud，讓餅乾有「份量」）
for i in range(n):
    ti = i / RATE
    thud = 0.18 * math.sin(2 * math.pi * 320 * ti) * math.exp(-ti * 80)
    raw[i] += thud

# lowpass 磨掉最尖的數位刺耳感
raw = lowpass(raw, coef=0.55)

# 正規化到目標音量
peak = max(abs(s) for s in raw) or 1.0
samples = [s / peak * 26000 for s in raw]
write_wav(os.path.join(os.path.dirname(__file__), "cookie_crunch.wav"), samples)

print("\nAll SFX generated.")
