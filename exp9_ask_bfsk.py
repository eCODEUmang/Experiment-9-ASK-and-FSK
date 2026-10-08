# %% [markdown]
# # Experiment 9 — ASK and BFSK (Python / NumPy version)
# Toolbox-free simulation of binary ASK (OOK) and binary FSK with **coherent** (correlator)
# and **noncoherent** (envelope / energy) detection over AWGN.
# Everything (modulation, channel, correlators, BER, spectra) is written by hand with NumPy;
# SciPy is used only for `erfc` (Q-function) and the scaled Bessel function `ive`.
#
# Run in Colab (Runtime > Run all) or locally: `python exp9_ask_bfsk.py`
# Figures are saved as `fig01_...png` ... `fig11_...png`; the console log is the results table.

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc, ive

rng = np.random.default_rng(1)                      # reproducible (analogue of MATLAB rng(1))
plt.rcParams.update({"font.size": 11, "lines.linewidth": 1.2, "axes.grid": True, "figure.dpi": 100})
Q = lambda x: 0.5 * erfc(np.asarray(x, float) / np.sqrt(2))     # Gaussian Q-function


def finish(fig, name):
    fig.tight_layout()
    fig.savefig(name, dpi=200)
    plt.show()


# %% [markdown]
# ## 1. Parameters
# * `Tb = 1 ms`, `Fs = 80 kHz` -> 80 samples/bit
# * ASK carrier `fc = 5 kHz` (5 cycles/bit, 16 samples/cycle)
# * BFSK `f1 = 4 kHz`, `f2 = 5 kHz` -> `df*Tb = 1` (orthogonal for coherent *and* noncoherent detection)
# * Average bit energy: ASK `Eb = A^2 Tb/4` (bit 0 has zero energy), BFSK `Eb = A^2 Tb/2`.

# %%
Tb, Fs = 1e-3, 80e3
dt = 1 / Fs
spb = int(round(Fs * Tb))
t = np.arange(spb) * dt
A = 1.0
fc, f1, f2 = 5e3, 4e3, 5e3
Eb_ask = A**2 * Tb / 4
Eb_fsk = A**2 * Tb / 2

Nbits, Nbits_hist = 10_000, 40_000
EbN0_demo = 8                                       # dB
EbN0_vec = np.arange(0, 13, 2)                      # dB
Nbits_ber, Nbatch = 200_000, 20_000
tone_x = np.arange(0, 2.5001, 0.125)                # (f2-f1)*Tb
EbN0_tone, Nbits_tone = 8, 100_000

print("=== PARAMETERS ===")
print(f"Tb={Tb*1e3:.1f} ms, Fs={Fs/1e3:.0f} kHz, samples/bit={spb}")
print(f"ASK fc={fc/1e3:.1f} kHz ({fc*Tb:.0f} cycles/bit, {Fs/fc:.0f} samples/cycle)")
print(f"BFSK f1={f1/1e3:.1f} kHz, f2={f2/1e3:.1f} kHz, df*Tb={(f2-f1)*Tb:.2f}")
print(f"Eb_ask(avg)={Eb_ask:.3e} J, Eb_fsk={Eb_fsk:.3e} J (A={A} V, 1 ohm)\n")


# %% [markdown]
# ## 2. Transmitter, channel, detectors (all written by hand)

# %%
def _col(theta):
    theta = np.asarray(theta, float)
    return theta[:, None] if theta.ndim == 1 else theta


def generate_ask(bits, A, fc, t, theta=0.0):
    """One row per bit: bit 1 -> A cos(2 pi fc t + theta), bit 0 -> 0."""
    return A * bits[:, None] * np.cos(2 * np.pi * fc * t[None, :] + _col(theta))


def generate_bfsk(bits, A, f1, f2, t, theta=0.0):
    """One row per bit: bit 0 -> A cos(2 pi f1 t + theta), bit 1 -> A cos(2 pi f2 t + theta)."""
    f = (f1 + bits * (f2 - f1))[:, None]
    return A * np.cos(2 * np.pi * f * t[None, :] + _col(theta))


def add_awgn(S, EbN0_dB, Eb, Fs):
    """N0 = Eb/10^(EbN0/10); sampled noise variance sigma^2 = N0*Fs/2 (PSD N0/2 over +-Fs/2)."""
    N0 = Eb / 10 ** (EbN0_dB / 10)
    sigma = np.sqrt(N0 * Fs / 2)
    return S + sigma * rng.standard_normal(S.shape), sigma, N0


def coherent_ask_detector(R, theta_ref=0.0):
    """z = sum r[n] cos(2 pi fc t[n] + theta_ref) dt over exactly one bit.
    Threshold = midpoint of E[z|0]=0 and E[z|1]=A*Tb/2 (equal priors, equal noise variance)."""
    C = np.cos(2 * np.pi * fc * t + theta_ref)
    z = R @ C * dt
    thr = 0.5 * A * np.sum(np.cos(2 * np.pi * fc * t) ** 2) * dt        # = A*Tb/4
    return (z > thr).astype(float), z, thr


def noncoherent_ask_detector(R, thr):
    """Quadrature correlation -> envelope sqrt(X^2+Y^2); independent of carrier phase."""
    X = R @ np.cos(2 * np.pi * fc * t) * dt
    Y = R @ np.sin(2 * np.pi * fc * t) * dt
    env = np.hypot(X, Y)
    return (env > thr).astype(float), env


def ask_envelope_threshold(EbN0_dB):
    """Bit 0 -> Rayleigh(sg); bit 1 -> Rician(m, sg), m=A*Tb/2, sg^2=N0*Tb/4.
    Pe(T) = 0.5 exp(-T^2/2sg^2) + 0.5 P(env1<T); T minimising Pe is found on a fine grid."""
    N0 = Eb_ask / 10 ** (EbN0_dB / 10)
    sg, m = np.sqrt(N0 * Tb / 4), A * Tb / 2
    r = np.linspace(0, m + 10 * sg, 8000)
    pdf1 = (r / sg**2) * np.exp(-(r - m) ** 2 / (2 * sg**2)) * ive(0, r * m / sg**2)   # scaled Bessel
    cdf1 = np.concatenate([[0], np.cumsum(0.5 * (pdf1[1:] + pdf1[:-1]) * np.diff(r))])
    Pe = 0.5 * np.exp(-(r**2) / (2 * sg**2)) + 0.5 * cdf1
    i = int(np.argmin(Pe))
    return r[i], Pe[i]


def coherent_bfsk_detector(R, f1, f2, theta_ref=0.0):
    z1 = R @ np.cos(2 * np.pi * f1 * t + theta_ref) * dt
    z2 = R @ np.cos(2 * np.pi * f2 * t + theta_ref) * dt
    return (z2 > z1).astype(float), z1, z2          # equal energy -> ML threshold on z2-z1 is 0


def noncoherent_bfsk_detector(R, f1, f2):
    E = []
    for f in (f1, f2):
        X = R @ np.cos(2 * np.pi * f * t) * dt
        Y = R @ np.sin(2 * np.pi * f * t) * dt
        E.append(X**2 + Y**2)                       # phase-independent energy
    return (E[1] > E[0]).astype(float), E[0], E[1]


def calculate_ber(tx, rx):
    assert tx.size == rx.size, "tx and rx must have the same length"
    return float(np.mean(tx != rx))


def inner_product(x, y):
    return float(np.sum(x * y) * dt)                # rectangle rule over exactly one bit


def sinc_(x):                                       # sin(x)/x, equal to 1 at 0 (NOT numpy's normalised sinc)
    x = np.asarray(x, float)
    return np.where(x == 0, 1.0, np.sin(x) / np.where(x == 0, 1, x))


def rho_coherent_theory(f1, f2):
    """(2/Tb) * integral_0^Tb cos(2 pi f1 t) cos(2 pi f2 t) dt  (exact)."""
    return sinc_(2 * np.pi * (f2 - f1) * Tb) + sinc_(2 * np.pi * (f1 + f2) * Tb)


def analytic_envelope(x):
    """Analytic signal from the FFT (zero negative frequencies, double positive) - no Hilbert toolbox."""
    N = x.size
    X = np.fft.fft(x)
    h = np.zeros(N)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2
    return np.abs(np.fft.ifft(X * h))


def power_spectrum(w, seglen=8000):
    """Averaged Hann-windowed periodogram, one-sided, dB relative to peak."""
    nseg = w.size // seglen
    win = np.hanning(seglen + 1)[:-1]               # periodic Hann
    U = np.sum(win**2)
    P = np.zeros(seglen)
    for k in range(nseg):
        P += np.abs(np.fft.fft(w[k * seglen:(k + 1) * seglen] * win)) ** 2 / (U * Fs)
    P /= nseg
    half = seglen // 2 + 1
    f = np.arange(half) * Fs / seglen
    return f, 10 * np.log10(P[:half] / P[:half].max() + 1e-12)


def simulate_ask_ber(EbN0_dB, N, batch):
    ec = en = done = 0
    thr, _ = ask_envelope_threshold(EbN0_dB)
    while done < N:
        nb = min(batch, N - done)
        b = (rng.random(nb) > 0.5).astype(float)
        R, *_ = add_awgn(generate_ask(b, A, fc, t, 0.0), EbN0_dB, Eb_ask, Fs)
        ec += np.sum(coherent_ask_detector(R, 0.0)[0] != b)
        R, *_ = add_awgn(generate_ask(b, A, fc, t, 2 * np.pi * rng.random(nb)), EbN0_dB, Eb_ask, Fs)
        en += np.sum(noncoherent_ask_detector(R, thr)[0] != b)
        done += nb
    return ec / N, en / N


def simulate_bfsk_ber(EbN0_dB, fa, fb, N, batch):
    ec = en = done = 0
    while done < N:
        nb = min(batch, N - done)
        b = (rng.random(nb) > 0.5).astype(float)
        R, *_ = add_awgn(generate_bfsk(b, A, fa, fb, t, 0.0), EbN0_dB, Eb_fsk, Fs)
        ec += np.sum(coherent_bfsk_detector(R, fa, fb, 0.0)[0] != b)
        R, *_ = add_awgn(generate_bfsk(b, A, fa, fb, t, 2 * np.pi * rng.random(nb)), EbN0_dB, Eb_fsk, Fs)
        en += np.sum(noncoherent_bfsk_detector(R, fa, fb)[0] != b)
        done += nb
    return ec / N, en / N


# %% [markdown]
# ## 3. Demo run at Eb/N0 = 8 dB (ASK, then BFSK)

# %%
bits = (rng.random(Nbits) > 0.5).astype(float)
theta = 2 * np.pi * rng.random(Nbits)               # unknown carrier phase for the noncoherent receivers

# ---- ASK
S_ask = generate_ask(bits, A, fc, t, 0.0)
R_ask, sig_ask, N0_ask = add_awgn(S_ask, EbN0_demo, Eb_ask, Fs)
rx_ask_c, z_ask, thr_ask = coherent_ask_detector(R_ask)
m_ask = A * np.sum(np.cos(2 * np.pi * fc * t) ** 2) * dt
print(f"=== ASK coherent @ {EbN0_demo} dB ===")
print(f"E[z|0]=0, E[z|1]={m_ask:.4e}, threshold={thr_ask:.4e} (midpoint), BER={calculate_ber(bits, rx_ask_c):.4e}")

thr_ask_nc, Pe_ask_nc_th = ask_envelope_threshold(EbN0_demo)
R_ask_ph, *_ = add_awgn(generate_ask(bits, A, fc, t, theta), EbN0_demo, Eb_ask, Fs)
rx_ask_n, env_ask = noncoherent_ask_detector(R_ask_ph, thr_ask_nc)
print(f"=== ASK noncoherent @ {EbN0_demo} dB ===")
print(f"threshold={thr_ask_nc:.4e} (optimal, theory Pe={Pe_ask_nc_th:.4e}), BER={calculate_ber(bits, rx_ask_n):.4e}\n")

# ---- BFSK
S_fsk = generate_bfsk(bits, A, f1, f2, t, 0.0)
R_fsk, sig_fsk, N0_fsk = add_awgn(S_fsk, EbN0_demo, Eb_fsk, Fs)
rx_fsk_c, z1, z2 = coherent_bfsk_detector(R_fsk, f1, f2)
R_fsk_ph, *_ = add_awgn(generate_bfsk(bits, A, f1, f2, t, theta), EbN0_demo, Eb_fsk, Fs)
rx_fsk_n, E1, E2 = noncoherent_bfsk_detector(R_fsk_ph, f1, f2)
print(f"=== BFSK @ {EbN0_demo} dB: coherent BER={calculate_ber(bits, rx_fsk_c):.4e}, "
      f"noncoherent BER={calculate_ber(bits, rx_fsk_n):.4e} ===\n")


# %% [markdown]
# ## 4. MANDATORY: BFSK orthogonality — inner products over one bit interval

# %%
phi1 = np.sqrt(2 / Tb) * np.cos(2 * np.pi * f1 * t)      # unit-energy basis functions
phi2 = np.sqrt(2 / Tb) * np.cos(2 * np.pi * f2 * t)
G = np.array([[inner_product(phi1, phi1), inner_product(phi1, phi2)],
              [inner_product(phi2, phi1), inner_product(phi2, phi2)]])
print(f"=== BFSK ORTHOGONALITY: inner products over one bit (f1={f1/1e3:.1f}, f2={f2/1e3:.1f} kHz) ===")
print(f"<phi1,phi1> = {G[0,0]: .6e}\n<phi1,phi2> = {G[0,1]: .6e}\n<phi2,phi1> = {G[1,0]: .6e}\n<phi2,phi2> = {G[1,1]: .6e}")
s0u, s1u = A * np.cos(2 * np.pi * f1 * t), A * np.cos(2 * np.pi * f2 * t)
print(f"Unnormalised: <s0,s1> = {inner_product(s0u, s1u):.4e}, <s0,s0> = {inner_product(s0u, s0u):.4e} (A^2Tb/2 = {A**2*Tb/2:.4e})")
print(f"Theory rho = sin(2pi df Tb)/(2pi df Tb) + sin(2pi(f1+f2)Tb)/(2pi(f1+f2)Tb) = {float(rho_coherent_theory(f1, f2)):.3e}")
B = np.sqrt(2 / Tb) * np.array([np.cos(2*np.pi*f1*t), np.sin(2*np.pi*f1*t), np.cos(2*np.pi*f2*t), np.sin(2*np.pi*f2*t)])
G4 = B @ B.T * dt
np.set_printoptions(precision=4, suppress=True)
print("4x4 Gram matrix [cos f1; sin f1; cos f2; sin f2] (identity => orthonormal):\n", G4)
print(f"Max |G4 - I| = {np.max(np.abs(G4 - np.eye(4))):.3e}\n")


# %% [markdown]
# ## 5. Figures 1–4: waveforms and correlator outputs

# %%
# ---- Fig 1: ASK chain
bs = np.array([1, 0, 1, 1, 0, 0, 1, 0], float); Ns = bs.size
ts = np.arange(Ns * spb) * dt * 1e3
Ss = generate_ask(bs, A, fc, t, 0.0); ws = Ss.reshape(-1)
Rs, *_ = add_awgn(Ss, EbN0_demo, Eb_ask, Fs); wr = Rs.reshape(-1)
rxs, zs, thrs = coherent_ask_detector(Rs)
env_s = analytic_envelope(wr)
fig, ax = plt.subplots(4, 1, figsize=(9, 9))
ax[0].step(np.arange(Ns + 1) * Tb * 1e3, np.r_[bs, bs[-1]], where="post", color="k", label="Input")
ax[0].plot((np.arange(Ns) + .5) * Tb * 1e3, rxs, "ro", label="Detected")
ax[0].set(ylim=(-.2, 1.2), xlabel="Time (ms)", ylabel="Bit value", title="ASK: input bits (line) and detected bits (circles)"); ax[0].legend(loc="center left", bbox_to_anchor=(1, .5))
ax[1].plot(ts, ws, "b"); ax[1].set(xlabel="Time (ms)", ylabel="Amplitude (V)", title=f"ASK passband waveform, f_c = {fc/1e3:.1f} kHz, A = {A:g} V")
ax[2].plot(ts, wr, color=".6", label="Noisy r(t)"); ax[2].plot(ts, env_s, "r", label="Envelope |analytic|")
ax[2].set(xlabel="Time (ms)", ylabel="Amplitude (V)", title=f"Noisy ASK (Eb/N0 = {EbN0_demo} dB) and FFT analytic-signal envelope"); ax[2].legend(loc="center left", bbox_to_anchor=(1, .5))
ax[3].stem((np.arange(Ns) + .5) * Tb * 1e3, zs, basefmt=" "); ax[3].axhline(thrs, color="r", ls="--", label="Threshold A·Tb/4")
ax[3].set(xlabel="Time (ms)", ylabel="z (V·s)", title="Coherent correlator statistic per bit and threshold"); ax[3].legend(loc="center left", bbox_to_anchor=(1, .5))
finish(fig, "fig01_ask_waveform.png")

# ---- Fig 2: BFSK waveform
bf = np.array([0, 1, 1, 0, 1, 0, 0, 1], float)
Sf = generate_bfsk(bf, A, f1, f2, t, 0.0); wf = Sf.reshape(-1)
Rf, *_ = add_awgn(Sf, EbN0_demo, Eb_fsk, Fs); wrf = Rf.reshape(-1)
fig, ax = plt.subplots(3, 1, figsize=(9, 7))
ax[0].step(np.arange(Ns + 1) * Tb * 1e3, np.r_[bf, bf[-1]], where="post", color="k")
ax[0].set(ylim=(-.2, 1.2), xlabel="Time (ms)", ylabel="Bit value", title="BFSK: input bits (0 → f1, 1 → f2)")
for k in range(Ns):
    idx = slice(k * spb, (k + 1) * spb)
    ax[1].plot(ts[idx], wf[idx], "r" if bf[k] == 0 else "b")
ax[1].set(xlabel="Time (ms)", ylabel="Amplitude (V)", title=f"BFSK passband: red = f1 = {f1/1e3:.0f} kHz (4 cycles/bit), blue = f2 = {f2/1e3:.0f} kHz (5 cycles/bit)")
ax[2].plot(ts, wrf, color=".4"); ax[2].set(xlabel="Time (ms)", ylabel="Amplitude (V)", title=f"Noisy BFSK, Eb/N0 = {EbN0_demo} dB")
finish(fig, "fig02_bfsk_waveform.png")

# ---- Fig 3: ASK correlator outputs
kk = np.arange(40); one = kk[bits[kk] == 1]
fig, ax = plt.subplots(2, 1, figsize=(9, 6.5))
ax[0].stem(kk + 1, z_ask[kk], basefmt=" "); ax[0].axhline(thr_ask, color="r", ls="--", label="Threshold")
ax[0].plot(one + 1, z_ask[one], "go", mfc="none", ms=9, label="Transmitted 1")
ax[0].set(xlabel="Bit index", ylabel="z (V·s)", title="Coherent ASK correlator output  z = ∫ r(t) cos(2π f_c t) dt"); ax[0].legend(loc="center left", bbox_to_anchor=(1, .5))
ax[1].stem(kk + 1, env_ask[kk], basefmt=" "); ax[1].axhline(thr_ask_nc, color="r", ls="--", label="Threshold")
ax[1].plot(one + 1, env_ask[one], "go", mfc="none", ms=9, label="Transmitted 1")
ax[1].set(xlabel="Bit index", ylabel="Envelope (V·s)", title="Noncoherent ASK envelope statistic (random carrier phase)"); ax[1].legend(loc="center left", bbox_to_anchor=(1, .5))
finish(fig, "fig03_ask_correlator.png")

# ---- Fig 4: BFSK correlator outputs
fig, ax = plt.subplots(3, 1, figsize=(9, 8.5))
ax[0].stem(kk + 1 - .15, z1[kk], linefmt="C0-", markerfmt="C0o", basefmt=" ", label="z1 (f1)")
ax[0].stem(kk + 1 + .15, z2[kk], linefmt="C1-", markerfmt="C1o", basefmt=" ", label="z2 (f2)")
ax[0].set(xlabel="Bit index", ylabel="z (V·s)", title="Coherent BFSK correlator outputs"); ax[0].legend(loc="center left", bbox_to_anchor=(1, .5))
ax[1].stem(kk + 1, (z2 - z1)[kk], basefmt=" "); ax[1].axhline(0, color="r", ls="--", label="Threshold 0")
ax[1].plot(one + 1, (z2 - z1)[one], "go", mfc="none", ms=9, label="Transmitted 1")
ax[1].set(xlabel="Bit index", ylabel="z2 − z1 (V·s)", title="Decision variable (threshold 0 by symmetry)"); ax[1].legend(loc="center left", bbox_to_anchor=(1, .5))
ax[2].stem(kk + 1 - .15, E1[kk], linefmt="C0-", markerfmt="C0o", basefmt=" ", label="E1")
ax[2].stem(kk + 1 + .15, E2[kk], linefmt="C1-", markerfmt="C1o", basefmt=" ", label="E2")
ax[2].set(xlabel="Bit index", ylabel="Energy (V²·s²)", title="Noncoherent BFSK energies E1, E2 (random carrier phase)"); ax[2].legend(loc="center left", bbox_to_anchor=(1, .5))
finish(fig, "fig04_bfsk_correlator.png")


# %% [markdown]
# ## 6. Power spectra (FFT, Hann-windowed, averaged)

# %%
fa, Pa = power_spectrum(S_ask.reshape(-1))
ff, Pf = power_spectrum(S_fsk.reshape(-1))
fig, ax = plt.subplots(2, 1, figsize=(9, 7))
ax[0].plot(fa / 1e3, Pa, "b", label="ASK"); ax[0].axvline(fc / 1e3, color="r", ls="--", label="f_c")
ax[0].set(xlim=(0, 10), ylim=(-80, 5), xlabel="Frequency (kHz)", ylabel="PSD (dB rel. peak)", title="ASK power spectrum (carrier + sinc² sidebands, null spacing 1/Tb)"); ax[0].legend(loc="center left", bbox_to_anchor=(1, .5))
ax[1].plot(ff / 1e3, Pf, "b", label="BFSK"); ax[1].axvline(f1 / 1e3, color="r", ls="--", label="f1"); ax[1].axvline(f2 / 1e3, color="g", ls="--", label="f2")
ax[1].set(xlim=(0, 10), ylim=(-80, 5), xlabel="Frequency (kHz)", ylabel="PSD (dB rel. peak)", title="BFSK power spectrum: peaks at f1 and f2"); ax[1].legend(loc="center left", bbox_to_anchor=(1, .5))
finish(fig, "fig05_power_spectra.png")

fig, ax = plt.subplots(figsize=(9, 4.5))
for dfx, c in zip([0.5, 1, 2], ["r", "b", "k"]):
    bq = (rng.random(Nbits) > 0.5).astype(float)
    fq, Pq = power_spectrum(generate_bfsk(bq, A, f1, f1 + dfx / Tb, t, 0.0).reshape(-1))
    ax.plot(fq / 1e3, Pq, c, label=f"Δf·Tb = {dfx:g}")
ax.set(xlim=(2, 9), ylim=(-70, 5), xlabel="Frequency (kHz)", ylabel="PSD (dB rel. peak)", title="BFSK spectrum vs tone spacing (f1 = 4 kHz fixed)"); ax.legend()
finish(fig, "fig06_bfsk_spacing_spectra.png")


# %% [markdown]
# ## 7. Decision-statistic histograms

# %%
bh = (rng.random(Nbits_hist) > 0.5).astype(float)
th_h = 2 * np.pi * rng.random(Nbits_hist)
Ra, *_ = add_awgn(generate_ask(bh, A, fc, t, 0.0), EbN0_demo, Eb_ask, Fs);          _, zh, thh = coherent_ask_detector(Ra)
Rap, *_ = add_awgn(generate_ask(bh, A, fc, t, th_h), EbN0_demo, Eb_ask, Fs);        _, envh = noncoherent_ask_detector(Rap, thr_ask_nc)
Rf2, *_ = add_awgn(generate_bfsk(bh, A, f1, f2, t, 0.0), EbN0_demo, Eb_fsk, Fs);    _, z1h, z2h = coherent_bfsk_detector(Rf2, f1, f2)
Rfp, *_ = add_awgn(generate_bfsk(bh, A, f1, f2, t, th_h), EbN0_demo, Eb_fsk, Fs);   _, E1h, E2h = noncoherent_bfsk_detector(Rfp, f1, f2)
Dh, DEh = z2h - z1h, E2h - E1h
panels = [(zh, thh, "z (V·s)", "Coherent ASK statistic"),
          (Dh, 0.0, "z2 − z1 (V·s)", "Coherent BFSK statistic"),
          (envh, thr_ask_nc, "Envelope (V·s)", "Noncoherent ASK: Rayleigh (0) vs Rician (1)"),
          (DEh, 0.0, "E2 − E1 (V²·s²)", "Noncoherent BFSK statistic")]
fig, axs = plt.subplots(2, 2, figsize=(11, 8))
for ax, (x, thr, xl, ttl) in zip(axs.ravel(), panels):
    ax.hist(x[bh == 0], 60, density=True, color="r", alpha=.6, label="bit 0")
    ax.hist(x[bh == 1], 60, density=True, color="b", alpha=.6, label="bit 1")
    ax.axvline(thr, color="k", ls="--", lw=2, label="threshold"); ax.set(xlabel=xl, ylabel="PDF", title=ttl); ax.legend()
finish(fig, "fig07_histograms.png")
sg = np.sqrt(N0_ask * Tb / 4); sgf = np.sqrt(N0_fsk * Tb / 4)
print(f"=== HISTOGRAM STATISTICS @ {EbN0_demo} dB ({Nbits_hist} bits) ===")
print(f"ASK coh: mean0={zh[bh==0].mean():.3e} mean1={zh[bh==1].mean():.3e} (theory 0, {m_ask:.3e}); "
      f"std0={zh[bh==0].std():.3e} std1={zh[bh==1].std():.3e} (theory {sg:.3e}); thr={thh:.3e}")
print(f"BFSK coh: mean(D|0)={Dh[bh==0].mean():.3e} mean(D|1)={Dh[bh==1].mean():.3e} (theory -/+ {A*Tb/2:.3e}); "
      f"std0={Dh[bh==0].std():.3e} std1={Dh[bh==1].std():.3e} (theory sqrt2*{sgf:.3e}={np.sqrt(2)*sgf:.3e})\n")


# %% [markdown]
# ## 8. BER vs Eb/N0
# **Expected:** coherent ASK (average Eb) = coherent BFSK = Q(√(Eb/N0)); noncoherent BFSK = ½exp(−Eb/2N0);
# noncoherent ASK is worse than coherent ASK at low SNR and converges at high SNR.

# %%
print(f"=== BER vs Eb/N0 ({Nbits_ber} bits per point) ===")
nE = EbN0_vec.size
BER, TH = np.zeros((nE, 4)), np.zeros((nE, 4))
for i, db in enumerate(EbN0_vec):
    BER[i, 0], BER[i, 1] = simulate_ask_ber(db, Nbits_ber, Nbatch)
    BER[i, 2], BER[i, 3] = simulate_bfsk_ber(db, f1, f2, Nbits_ber, Nbatch)
    g = 10 ** (db / 10)
    TH[i] = [Q(np.sqrt(g)), ask_envelope_threshold(db)[1], Q(np.sqrt(g)), 0.5 * np.exp(-g / 2)]
    print(f"{db:5.1f} dB | simASKc {BER[i,0]:.3e} (th {TH[i,0]:.3e}) | simASKn {BER[i,1]:.3e} (th {TH[i,1]:.3e}) | "
          f"simFSKc {BER[i,2]:.3e} (th {TH[i,2]:.3e}) | simFSKn {BER[i,3]:.3e} (th {TH[i,3]:.3e})")
ed = np.linspace(0, 12, 100); gd = 10 ** (ed / 10)
B2 = np.where(BER == 0, np.nan, BER)
fig, ax = plt.subplots(figsize=(9, 6.5))
ax.semilogy(ed, Q(np.sqrt(gd)), "k-", label="Theory Q(√(Eb/N0)) [coh ASK = coh BFSK]")
ax.semilogy(EbN0_vec, B2[:, 0], "bo", label="Sim coherent ASK")
ax.semilogy(EbN0_vec, B2[:, 1], "bs", mfc="c", label="Sim noncoherent ASK")
ax.semilogy(EbN0_vec, B2[:, 2], "r^", label="Sim coherent BFSK")
ax.semilogy(EbN0_vec, B2[:, 3], "mv", label="Sim noncoherent BFSK")
ax.semilogy(ed, 0.5 * np.exp(-gd / 2), "m--", label="Theory noncoh BFSK ½e^(−Eb/2N0)")
ax.semilogy(EbN0_vec, TH[:, 1], "c--", label="Theory noncoh ASK (Rician/Rayleigh, numeric)")
ax.set(xlim=(0, 12), ylim=(1e-6, 1), xlabel="Eb/N0 (dB)", ylabel="Bit error rate", title="BER vs Eb/N0 (simulated markers, theory lines)")
ax.grid(True, which="both", alpha=.4); ax.legend(loc="lower left", fontsize=9)
finish(fig, "fig08_ber_curves.png")


# %% [markdown]
# ## 9. Task 35 — tone-spacing experiment (f1 fixed, f2 = f1 + Δf)
# **Expected:** BER ≈ 0.5 at Δf = 0; coherent BER reaches its orthogonal value at Δf·Tb = 0.5, 1, 1.5, 2 and dips near 0.7
# (negative correlation); noncoherent BER is best only at Δf·Tb = 1, 2 (envelope correlation |sinc(Δf·Tb)| = 0), so
# 0.5 is orthogonal for coherent but NOT for noncoherent detection.

# %%
print(f"\n=== TONE-SPACING EXPERIMENT @ Eb/N0 = {EbN0_tone} dB, f1 = {f1/1e3:.1f} kHz fixed ===")
nT = tone_x.size
ber_tc, ber_tn = np.zeros(nT), np.zeros(nT)
rho_c_num, rho_n_num, rho_c_th, rho_n_th = (np.zeros(nT) for _ in range(4))
for q, x in enumerate(tone_x):
    f2q = f1 + x / Tb
    c1 = np.sqrt(2 / Tb) * np.cos(2*np.pi*f1*t); c2 = np.sqrt(2 / Tb) * np.cos(2*np.pi*f2q*t); s2 = np.sqrt(2 / Tb) * np.sin(2*np.pi*f2q*t)
    rho_c_num[q] = inner_product(c1, c2)
    rho_n_num[q] = np.hypot(inner_product(c1, c2), inner_product(c1, s2))
    rho_c_th[q] = rho_coherent_theory(f1, f2q)
    rho_n_th[q] = abs(float(sinc_(np.pi * x)))
    ber_tc[q], ber_tn[q] = simulate_bfsk_ber(EbN0_tone, f1, f2q, Nbits_tone, Nbatch)
    print(f"df*Tb={x:5.3f} f2={f2q/1e3:6.3f} kHz | rho_coh num={rho_c_num[q]: .4f} th={rho_c_th[q]: .4f} | "
          f"rho_env num={rho_n_num[q]:.4f} th={rho_n_th[q]:.4f} | BER coh={ber_tc[q]:.4e} noncoh={ber_tn[q]:.4e}")
g = 10 ** (EbN0_tone / 10)
fig, ax = plt.subplots(figsize=(9, 6))
ax.semilogy(tone_x, np.where(ber_tc == 0, np.nan, ber_tc), "bo-", label="Sim coherent")
ax.semilogy(tone_x, np.where(ber_tn == 0, np.nan, ber_tn), "rs-", label="Sim noncoherent")
ax.semilogy(tone_x, Q(np.sqrt(g * (1 - rho_c_th))), "k--", label="Theory coherent Q(√(γ(1−ρ)))")
ax.semilogy(tone_x, 0.5 * np.exp(-g / 2) * np.ones(nT), "m:", label="Theory noncoh (orthogonal) ½e^(−γ/2)")
for xo in (1, 2): ax.axvline(xo, color="g", ls=":")
ax.set(ylim=(1e-4, 1), xlabel="Tone spacing (f2 − f1)·Tb", ylabel="Bit error rate", title=f"BFSK BER vs tone spacing, Eb/N0 = {EbN0_tone} dB")
ax.grid(True, which="both", alpha=.4); ax.legend(loc="lower left")
finish(fig, "fig09_ber_vs_tone_spacing.png")


# %% [markdown]
# ## 10. Orthogonality figure and phase-sensitivity figure

# %%
labels = ["cos f1", "sin f1", "cos f2", "sin f2"]
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
im = ax[0].imshow(G4, vmin=-1, vmax=1, cmap="jet"); fig.colorbar(im, ax=ax[0], fraction=.046)
ax[0].set(xticks=range(4), yticks=range(4), xticklabels=labels, yticklabels=labels,
          title=f"Gram matrix <φi,φj>, f1={f1/1e3:.0f} kHz, f2={f2/1e3:.0f} kHz"); ax[0].grid(False)
for a in range(4):
    for b in range(4):
        ax[0].text(b, a, f"{G4[a,b]:.3f}", ha="center", va="center", color="w", fontweight="bold")
xf = np.linspace(0, 2.5, 500)
ax[1].plot(xf, rho_coherent_theory(f1, f1 + xf / Tb), "b", label="Theory coherent ρ")
ax[1].plot(xf, np.abs(sinc_(np.pi * xf)), "r", label="Theory envelope |sinc|")
ax[1].plot(tone_x, rho_c_num, "bo", label="Numerical coherent"); ax[1].plot(tone_x, rho_n_num, "rs", label="Numerical envelope")
ax[1].axhline(0, color="k", lw=.8)
ax[1].set(xlabel="Tone spacing (f2 − f1)·Tb", ylabel="Normalised correlation ρ", title="Cross-correlation vs tone spacing"); ax[1].legend()
finish(fig, "fig10_orthogonality.png")

# ---- Fig 11: phase sensitivity
phi_deg = np.arange(0, 91, 15); EbN0_ph, Nph = 10, 100_000
g = 10 ** (EbN0_ph / 10)
sg_ph = np.sqrt((Eb_ask / g) * Tb / 4); m1 = A * Tb / 2
b_pa, b_pf = np.zeros(phi_deg.size), np.zeros(phi_deg.size)
for q, d in enumerate(phi_deg):
    ph = np.deg2rad(d); bp = (rng.random(Nph) > 0.5).astype(float)
    Rp, *_ = add_awgn(generate_ask(bp, A, fc, t, 0.0), EbN0_ph, Eb_ask, Fs);       b_pa[q] = calculate_ber(bp, coherent_ask_detector(Rp, ph)[0])
    Rp, *_ = add_awgn(generate_bfsk(bp, A, f1, f2, t, 0.0), EbN0_ph, Eb_fsk, Fs);  b_pf[q] = calculate_ber(bp, coherent_bfsk_detector(Rp, f1, f2, ph)[0])
th_pa = 0.5 * Q((m1 / 2) / sg_ph) + 0.5 * Q((m1 * np.cos(np.deg2rad(phi_deg)) - m1 / 2) / sg_ph)
th_pf = Q(np.sqrt(g) * np.cos(np.deg2rad(phi_deg)))
bq = (rng.random(Nph) > 0.5).astype(float); thq = 2 * np.pi * rng.random(Nph)
Rq, *_ = add_awgn(generate_ask(bq, A, fc, t, thq), EbN0_ph, Eb_ask, Fs);       bn_a = calculate_ber(bq, noncoherent_ask_detector(Rq, ask_envelope_threshold(EbN0_ph)[0])[0])
Rq, *_ = add_awgn(generate_bfsk(bq, A, f1, f2, t, thq), EbN0_ph, Eb_fsk, Fs);  bn_f = calculate_ber(bq, noncoherent_bfsk_detector(Rq, f1, f2)[0])
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(phi_deg, b_pa, "bo", label="Sim coh ASK"); ax.plot(phi_deg, th_pa, "b--", label="Theory coh ASK")
ax.plot(phi_deg, b_pf, "r^", label="Sim coh BFSK"); ax.plot(phi_deg, th_pf, "r--", label="Theory coh BFSK")
ax.axhline(bn_a, color="c", label="Noncoh ASK (phase-free)"); ax.axhline(bn_f, color="m", label="Noncoh BFSK (phase-free)")
ax.set(xlabel="Receiver carrier-phase error (degrees)", ylabel="Bit error rate", title=f"Coherent detector phase sensitivity, Eb/N0 = {EbN0_ph} dB"); ax.legend(loc="upper left", fontsize=9)
finish(fig, "fig11_phase_sensitivity.png")


# %% [markdown]
# ## 11. Self-checks (diagnostic tests)

# %%
print("\n=== SELF-CHECKS ===")
nz = R_ask - S_ask
print(f"[1] sample noise variance: measured {nz.var():.4e}, derived N0*Fs/2 = {N0_ask*Fs/2:.4e}")
zc = nz @ np.cos(2 * np.pi * fc * t) * dt
print(f"[2] correlator noise variance: measured {zc.var():.4e}, theory N0*Tb/4 = {N0_ask*Tb/4:.4e}")
bq = (rng.random(5000) > 0.5).astype(float); tq = 2 * np.pi * rng.random(5000)
r = [calculate_ber(bq, coherent_ask_detector(generate_ask(bq, A, fc, t, 0.0))[0]),
     calculate_ber(bq, noncoherent_ask_detector(generate_ask(bq, A, fc, t, tq), thr_ask_nc)[0]),
     calculate_ber(bq, coherent_bfsk_detector(generate_bfsk(bq, A, f1, f2, t, 0.0), f1, f2)[0]),
     calculate_ber(bq, noncoherent_bfsk_detector(generate_bfsk(bq, A, f1, f2, t, tq), f1, f2)[0])]
print(f"[3] noiseless BER (ASKc, ASKn, FSKc, FSKn) = {r} (must all be 0)")
print(f"[4] simulated vs theory (95% binomial CI, N={Nbits_ber}); only points OUTSIDE the CI are listed:")
names = ["coh ASK", "noncoh ASK", "coh BFSK", "noncoh BFSK"]
for i in range(nE):
    for c in range(4):
        ci = 1.96 * np.sqrt(max(TH[i, c] * (1 - TH[i, c]), 1e-12) / Nbits_ber)
        if abs(BER[i, c] - TH[i, c]) > ci:
            print(f"    CHECK {names[c]:12s} @ {EbN0_vec[i]:4.1f} dB: sim {BER[i,c]:.3e} vs th {TH[i,c]:.3e} (CI ±{ci:.1e})")
print("    (about 5% of points outside the 95% CI is expected by chance)")
bq = (rng.random(Nph) > 0.5).astype(float); bers = []
for ph in (0.0, np.pi / 2, 2 * np.pi * rng.random(Nph)):
    Rq, *_ = add_awgn(generate_bfsk(bq, A, f1, f2, t, ph), EbN0_ph, Eb_fsk, Fs)
    bers.append(calculate_ber(bq, noncoherent_bfsk_detector(Rq, f1, f2)[0]))
print(f"[5] noncoherent BFSK BER at phase 0 / 90deg / random = {bers[0]:.4e} {bers[1]:.4e} {bers[2]:.4e} (should be equal within noise)")
