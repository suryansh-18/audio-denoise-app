# Audio Denoise
# Save as Audio_Denoise_GUI.py

import streamlit as st
import numpy as np
import io, os, tempfile
# Defensive plotting imports: matplotlib preferred, fallback to plotly
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    MATPLOTLIB_AVAILABLE = True
except Exception:
    plt = None
    PdfPages = None
    MATPLOTLIB_AVAILABLE = False

# Try Plotly as a fallback for plotting in Streamlit
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:
    go = None
    px = None
    PLOTLY_AVAILABLE = False

from scipy import signal
import wave

# Optional backends
try:
    import soundfile as sf
    SOUND_FILE_AVAILABLE = True
except Exception:
    sf = None
    SOUND_FILE_AVAILABLE = False

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except Exception:
    AudioSegment = None
    PYDUB_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except Exception:
    librosa = None
    LIBROSA_AVAILABLE = False

# ---------------- helper I/O & conversion ----------------
def read_wav_fallback(path):
    with wave.open(path,'rb') as wf:
        nch = wf.getnchannels(); sampwidth = wf.getsampwidth(); fs = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    if sampwidth == 1:
        data = np.frombuffer(frames,dtype=np.uint8).astype(np.float32); data=(data-128.0)/128.0
    elif sampwidth == 2:
        data = np.frombuffer(frames,dtype=np.int16).astype(np.float32); data = data/32768.0
    else:
        raise ValueError("Unsupported sample width")
    if nch>1:
        data = data.reshape(-1,nch).mean(axis=1)
    return data, fs

def write_wav_fallback(path, data, fs):
    if np.max(np.abs(data))>1.0:
        data = data/(np.max(np.abs(data))+1e-12)
    int_data = (data*32767.0).astype(np.int16)
    with wave.open(path,'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(int(fs)); wf.writeframes(int_data.tobytes())

def write_wav_fallback_to_buffer(buf, data, fs):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav'); tmp.close()
    write_wav_fallback(tmp.name, data, fs)
    with open(tmp.name,'rb') as f: buf.write(f.read())
    os.remove(tmp.name)

def convert_to_wav_bytes(file_like, filename_hint=None):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename_hint or 'upload')[1] if filename_hint else '.tmp')
    try:
        tmp.write(file_like.read()); tmp.close(); path=tmp.name
        if SOUND_FILE_AVAILABLE:
            try:
                data, fs = sf.read(path)
                if getattr(data,'ndim',1)>1: data = data.mean(axis=1)
                buf = io.BytesIO(); sf.write(buf, data.astype(np.float32), fs, format='WAV'); buf.seek(0)
                return buf.getvalue(), fs
            except Exception:
                pass
        if PYDUB_AVAILABLE:
            try:
                seg = AudioSegment.from_file(path)
                out = io.BytesIO(); seg.export(out, format='wav'); out.seek(0)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as t2:
                    t2.write(out.getvalue()); t2.close()
                    data, fs = read_wav_fallback(t2.name); os.remove(t2.name)
                return out.getvalue(), fs
            except Exception:
                pass
        if LIBROSA_AVAILABLE:
            try:
                data, fs = librosa.load(path, sr=None, mono=True)
                data = data.astype(np.float32)
                if SOUND_FILE_AVAILABLE:
                    buf = io.BytesIO(); sf.write(buf, data, fs, format='WAV'); buf.seek(0); return buf.getvalue(), fs
                else:
                    tmp2 = tempfile.NamedTemporaryFile(delete=False, suffix='.wav'); tmp2.close()
                    write_wav_fallback(tmp2.name, data, fs)
                    with open(tmp2.name,'rb') as f: b=f.read(); os.remove(tmp2.name)
                    return b, fs
            except Exception:
                pass
        # fallback: if it's WAV already
        try:
            data, fs = read_wav_fallback(path)
            buf = io.BytesIO(); write_wav_fallback_to_buffer(buf, data, fs); buf.seek(0)
            return buf.getvalue(), fs
        except Exception:
            pass
        raise RuntimeError("No backend available to convert file. Install soundfile/pydub+ffmpeg/librosa.")
    finally:
        try: os.remove(tmp.name)
        except Exception: pass

def read_audio(path_or_fileobj):
    need_rm=False
    if hasattr(path_or_fileobj,'read'):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav'); tmp.write(path_or_fileobj.read()); tmp.close(); path=tmp.name; need_rm=True
    else:
        path=path_or_fileobj
    try:
        if SOUND_FILE_AVAILABLE:
            data, fs = sf.read(path)
            if getattr(data,'ndim',1)>1: data=data.mean(axis=1)
            data = data.astype(np.float32)
            if np.max(np.abs(data))>1.5: data = data/(np.max(np.abs(data))+1e-12)
        else:
            data, fs = read_wav_fallback(path)
    finally:
        if need_rm:
            try: os.remove(path)
            except: pass
    return data, fs

def write_audio_fileobj(fileobj, data, fs):
    buf = io.BytesIO()
    if SOUND_FILE_AVAILABLE:
        sf.write(buf, data.astype(np.float32), fs, format='WAV'); buf.seek(0); fileobj.write(buf.read())
    else:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav'); tmp.close()
        write_wav_fallback(tmp.name, data, fs)
        with open(tmp.name,'rb') as f: fileobj.write(f.read())
        os.remove(tmp.name)

# ---- Helper display functions (works with matplotlib or plotly) ----
def display_waveform(audio, fs, title="Waveform (first 5s)", max_seconds=5):
    """Return ('mpl', fig) or ('plotly', fig) or ('none', None)."""
    max_samples = min(len(audio), int(fs * max_seconds))
    t = np.arange(max_samples) / float(fs)
    y = audio[:max_samples]

    if MATPLOTLIB_AVAILABLE:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.plot(t, y)
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        return ("mpl", fig)
    elif PLOTLY_AVAILABLE:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=y, mode='lines', name=title))
        fig.update_layout(title=title, xaxis_title='Time (s)', height=240, margin=dict(l=20,r=20,t=40,b=20))
        return ("plotly", fig)
    else:
        return ("none", None)


def display_spectrogram(audio, fs, nperseg=2048, title="Spectrogram"):
    """Return ('mpl', fig) or ('plotly', fig) or ('none', None)."""
    f, t, Sxx = signal.spectrogram(audio, fs=fs, nperseg=nperseg)
    S_db = 10.0 * np.log10(Sxx + 1e-12)

    if MATPLOTLIB_AVAILABLE:
        fig, ax = plt.subplots(figsize=(8, 3))
        pcm = ax.pcolormesh(t, f, S_db, shading='gouraud')
        ax.set_ylabel('Frequency [Hz]')
        ax.set_xlabel('Time [s]')
        ax.set_title(title)
        fig.colorbar(pcm, ax=ax, format='%+2.0f dB')
        return ("mpl", fig)
    elif PLOTLY_AVAILABLE:
        fig = go.Figure(data=go.Heatmap(
            x=t,
            y=f,
            z=S_db,
            colorscale='Viridis',
            zmin=np.percentile(S_db, 1),
            zmax=np.percentile(S_db, 99),
            colorbar=dict(title='dB')
        ))
        fig.update_layout(title=title, xaxis_title='Time (s)', yaxis_title='Frequency (Hz)', height=360, margin=dict(l=40,r=20,t=40,b=20))
        return ("plotly", fig)
    else:
        return ("none", None)


def build_pdf_report_safe(pdf_buf, noisy_signal, processed_signal, fs, title="Auto Denoise Report"):
    """Create a PDF report only if matplotlib.PdfPages is available.
    Writes bytes into pdf_buf (BytesIO) and returns True on success, False if not available.
    """
    if not MATPLOTLIB_AVAILABLE or PdfPages is None:
        return False

    import matplotlib
    matplotlib.rcParams.update({'figure.max_open_warning': 0})
    with PdfPages(pdf_buf) as pdf:
        # cover
        fig_cover = plt.figure(figsize=(8.27, 11.69))
        plt.axis('off')
        plt.text(0.5, 0.9, title, ha='center', fontsize=16)
        pdf.savefig(fig_cover)
        plt.close(fig_cover)

        # Spectrogram page
        fig_s, ax_s = plt.subplots(2, 1, figsize=(8.27, 11.69))
        f1, t1, S1 = signal.spectrogram(noisy_signal, fs=fs, nperseg=2048)
        ax_s[0].pcolormesh(t1, f1, 10*np.log10(S1+1e-12), shading='gouraud')
        ax_s[0].set_title('Spectrogram - Noisy')
        f2, t2, S2 = signal.spectrogram(processed_signal, fs=fs, nperseg=2048)
        ax_s[1].pcolormesh(t2, f2, 10*np.log10(S2+1e-12), shading='gouraud')
        ax_s[1].set_title('Spectrogram - Processed')
        pdf.savefig(fig_s)
        plt.close(fig_s)
    return True

# ---------------- DSP helpers ----------------
def rms(x): return np.sqrt(np.mean(np.square(x)))
def compute_snr(clean, proc):
    noise = proc - clean
    return 20 * np.log10(rms(clean)/(rms(noise)+1e-12))
def _clamp_normalized_cutoff(cutoff_hz, fs):
    nyq = 0.5*fs
    if cutoff_hz is None: return None, None
    max_cut = nyq*0.98
    cut_hz = min(max(cutoff_hz, 10.0), max_cut)
    Wn = cut_hz/nyq
    Wn = max(min(Wn,0.999),1e-6)
    return Wn, cut_hz

def apply_filters(x, fs, low_cut=None, low_order=4, notch_freq=None, notch_Q=30):
    y = x.copy()
    if low_cut is not None:
        Wn,_ = _clamp_normalized_cutoff(low_cut, fs)
        b,a = signal.butter(low_order, Wn, btype='low')
        y = signal.filtfilt(b,a,y)
    if notch_freq is not None:
        Wn_notch,_ = _clamp_normalized_cutoff(notch_freq, fs)
        b_n,a_n = signal.iirnotch(Wn_notch, notch_Q)
        y = signal.filtfilt(b_n,a_n,y)
    return y

# ---------------- Spectral denoiser ----------------
def spectral_subtract_denoise(x, fs, n_fft=2048, hop=None, noise_frames=6, alpha=4.0, beta=0.002):
    if hop is None: hop = n_fft//4
    pad = n_fft
    x_padded = np.concatenate((np.zeros(pad), x, np.zeros(pad)))
    win = np.hanning(n_fft)
    frames = []
    for i in range(0, len(x_padded)-n_fft+1, hop):
        frames.append(x_padded[i:i+n_fft]*win)
    F = np.fft.rfft(np.stack(frames), axis=1)
    mag = np.abs(F); phase = np.angle(F)
    if mag.shape[0] <= noise_frames:
        noise_spec = np.median(mag, axis=0)
    else:
        noise_spec = np.mean(mag[:noise_frames,:], axis=0)
    mag_d = mag - alpha*noise_spec[np.newaxis,:]
    floor = beta*noise_spec[np.newaxis,:]
    mag_d = np.maximum(mag_d, floor)
    try:
        from scipy.ndimage import convolve
        kernel = np.ones((3,1))/3.0
        mag_d = convolve(mag_d, kernel, mode='reflect')
    except Exception:
        pass
    F_d = mag_d * np.exp(1j*phase)
    out = np.zeros(len(x_padded)); wsum = np.zeros(len(x_padded))
    for idx, frame in enumerate(F_d):
        i = idx*hop
        frame_time = np.fft.irfft(frame)
        frame_time = frame_time*win
        out[i:i+n_fft] += frame_time
        wsum[i:i+n_fft] += win**2
    nz = wsum>1e-8; out[nz] /= wsum[nz]
    out = out[pad:pad+len(x)]
    maxv = np.max(np.abs(out))+1e-12
    if maxv>1.0: out = out/maxv
    return out

def wiener_like_denoise(x, fs, n_fft=2048, hop=None):
    return spectral_subtract_denoise(x, fs, n_fft=n_fft, hop=hop, noise_frames=8, alpha=2.5, beta=0.001)

# low-pitch rumble/hum cleanup helper
def remove_low_pitch_rumble(x, fs, low_freq_thresh=120, hp_cut=60):
    N = len(x)
    if N < 1024: return x
    X = np.abs(np.fft.rfft(x * np.hanning(N)))
    freqs = np.fft.rfftfreq(N, 1/fs)
    low_idx = freqs <= low_freq_thresh
    low_energy = np.sum(X[low_idx]) + 1e-12
    total_energy = np.sum(X) + 1e-12
    rel_low = low_energy/total_energy
    y = x.copy()
    if rel_low > 0.45:
        Wn,_ = _clamp_normalized_cutoff(hp_cut, fs)
        b_hp,a_hp = signal.butter(2, Wn, btype='high')
        y = signal.filtfilt(b_hp,a_hp,y)
    low_spec = X[low_idx]
    med = np.median(low_spec)+1e-12
    peaks = np.where(low_spec > med*6)[0]
    if len(peaks)>0:
        peak_freqs = freqs[low_idx][peaks]; pf = peak_freqs[np.argmax(low_spec[peaks])]
        notches=[]
        if abs(pf-50)<6: notches.append(50)
        elif abs(pf-60)<6: notches.append(60)
        else: notches.append(int(round(pf)))
        h = notches[0]*2
        if h < fs/2: notches.append(h)
        for nf in notches:
            Wn_notch,_ = _clamp_normalized_cutoff(nf, fs)
            b_n,a_n = signal.iirnotch(Wn_notch, 30)
            y = signal.filtfilt(b_n,a_n,y)
    return y

# auto design (unchanged)
def auto_design_filters(x, fs):
    N = len(x)
    if N < 512: return 4000, None, False
    X = np.abs(np.fft.rfft(x * np.hanning(N)))
    freqs = np.fft.rfftfreq(N, 1/fs)
    cumsum = np.cumsum(X); total = cumsum[-1]+1e-12
    idx98 = np.searchsorted(cumsum, 0.98*total)
    raw_cut = freqs[idx98]*1.05
    nyq = 0.5*fs
    raw_cut = max(1000, min(raw_cut, nyq*0.98))
    low_cut = int(raw_cut)
    high_band = freqs>low_cut
    high_energy = np.sum(X[high_band])+1e-12
    rel_high = high_energy/total
    use_spectral = rel_high > 0.12
    low_band_idx = freqs<=200
    band_spec = X[low_band_idx]; med=np.median(band_spec)+1e-12
    peaks = np.where(band_spec > med*6)[0]; notch=None
    if len(peaks)>0:
        peak_freqs = freqs[low_band_idx][peaks]; pf = peak_freqs[np.argmax(band_spec[peaks])]
        if abs(pf-50)<6: notch=50
        elif abs(pf-60)<6: notch=60
        else: notch=int(round(pf))
    return low_cut, notch, use_spectral

# ---------------- UI ----------------
st.set_page_config(page_title="Audio Denoise", layout='wide')
st.title("Audio Denoise")

st.sidebar.header("Inputs & Controls")
st.sidebar.write("Upload audio (WAV/MP3/MP4/FLAC/OGG). MP3/MP4 will be converted to WAV where possible.")

st.sidebar.markdown("**Upload noisy audio (or leave empty to generate from a clean file)**")
noisy_upload = st.sidebar.file_uploader(
    "", 
    type=['wav','flac','ogg','mp3','mp4'], 
    key="noisy_file"
)

st.sidebar.markdown("**(Optional) Upload CLEAN audio to generate noisy examples**")
clean_upload = st.sidebar.file_uploader(
    "", 
    type=['wav','flac','ogg','mp3','mp4'], 
    key="clean_file"
)


st.sidebar.markdown("---")
st.sidebar.subheader("Mode")
mode = st.sidebar.radio("Choose processing mode", options=["Auto (aggressive-by-default)","Manual clean (you control filters)"])

# Aggressiveness slider (0..100) — single control that maps to many params
st.sidebar.markdown("**Aggressiveness (single slider)**")
aggr_val = st.sidebar.slider("Aggressiveness", min_value=0, max_value=100, value=75)
# map aggressiveness to algorithm params:
# alpha: 2.0 -> 8.0
alpha = 2.0 + (6.0 * (aggr_val/100.0))
# noise_frames: 4 -> 20
noise_frames = int(4 + (16 * (aggr_val/100.0)))
# n_fft: choose 2048 for low aggr, 4096 for high
n_fft = 4096 if aggr_val >= 60 else 2048
# restore_mix mapping: higher aggressiveness -> more restore mix to keep bass
restore_mix_auto = 0.25 + 0.75*(aggr_val/100.0)  # 0.25..1.0

# Show derived params for transparency
st.sidebar.markdown(f"alpha ≈ {alpha:.2f}  •  noise_frames = {noise_frames}  •  n_fft = {n_fft}")
st.sidebar.markdown("---")

# LF restore toggles
st.sidebar.subheader("Low-frequency restoration")
restore_enabled = st.sidebar.checkbox("Restore low frequencies from original", value=True)
# allow manual override of restore_mix and cutoff
restore_cutoff = st.sidebar.slider("Restore LF cutoff (Hz)", min_value=40, max_value=600, value=200)
restore_mix = st.sidebar.slider("Restore mix (override auto)", min_value=0.0, max_value=1.0, value=float(restore_mix_auto), step=0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("Manual cleaning controls (if Manual mode chosen)")
man_lowcut = st.sidebar.slider("Manual lowpass cutoff (Hz, 0=off)", min_value=0, max_value=20000, value=0)
man_loworder = st.sidebar.selectbox("Manual lowpass order", options=[2,4,6], index=1)
man_notch = st.sidebar.number_input("Manual notch frequency (Hz, 0=off)", min_value=0, value=0)
man_notchQ = st.sidebar.slider("Manual notch Q", min_value=1.0, max_value=60.0, value=30.0)
man_apply_highpass = st.sidebar.checkbox("Apply manual gentle high-pass (remove rumble)", value=False)
man_hp_cut = st.sidebar.slider("Manual HP cutoff (Hz)", min_value=10, max_value=400, value=40)

st.sidebar.markdown("---")
if st.sidebar.button("Run / Apply"):
    st.session_state['run_request'] = True
else:
    if 'run_request' not in st.session_state:
        st.session_state['run_request'] = False

# Auto-tune sweep (preview multiple aggressiveness choices)
st.sidebar.markdown("---")
st.sidebar.subheader("Auto-tune sweep (preview candidates)")
sweep_btn = st.sidebar.button("Run auto-tune preview")

# helper load uploader with conversion
def load_uploaded_audio(uploader):
    if uploader is None: return None, None
    name = getattr(uploader, 'name', '')
    ext = os.path.splitext(name)[1].lower()
    try:
        if ext in ['.mp3', '.mp4', '.m4a', '.aac']:
            wav_bytes, wav_fs = convert_to_wav_bytes(uploader, filename_hint=name)
            return read_audio(io.BytesIO(wav_bytes))
        else:
            return read_audio(io.BytesIO(uploader.read()))
    except Exception as e:
        st.sidebar.error(f"Error loading {name}: {e}")
        return None, None

# prepare input
noisy_data=None; fs=None; clean_ref=None
if clean_upload is not None and noisy_upload is None:
    clean_sig, fs = load_uploaded_audio(clean_upload)
    if clean_sig is None:
        st.sidebar.error("Failed reading clean upload")
    else:
        clean_sig = clean_sig/(np.max(np.abs(clean_sig))+1e-12)
        gen_choice = st.sidebar.selectbox("Generate noisy sample", ['White noise','50Hz hum'])
        if gen_choice=='White noise':
            noisy_data = clean_sig + (rms(clean_sig)/(10**(6/20))) * np.random.RandomState(0).normal(size=clean_sig.shape)
        else:
            t = np.arange(len(clean_sig))/fs
            noisy_data = clean_sig + 0.03*np.sin(2*np.pi*50*t)
        clean_ref = clean_sig

if noisy_upload is not None:
    noisy_data, fs = load_uploaded_audio(noisy_upload)

if noisy_data is None:
    st.info("Upload a noisy audio (or a clean audio to generate noisy examples) from the sidebar.")
    st.stop()

# normalize
if np.max(np.abs(noisy_data))>0: noisy_data = noisy_data/(np.max(np.abs(noisy_data))+1e-12)

# input preview
st.header("Input preview")
c1, c2 = st.columns([2,1])
with c1:
    kind, fig_or_none = display_waveform(noisy_data, fs, title='Noisy waveform (first 5s)', max_seconds=5)
    if kind == "mpl":
        st.pyplot(fig_or_none)
    elif kind == "plotly":
        st.plotly_chart(fig_or_none, use_container_width=True)
    else:
        st.info("Plot unavailable (matplotlib/plotly not installed).")
with c2:
    buf_in = io.BytesIO(); write_audio_fileobj(buf_in, noisy_data, fs); buf_in.seek(0)
    st.audio(buf_in.read())

# LF extract + RMS match helper
def restore_lowband_and_match(original, processed, fs, cutoff_hz, mix):
    # extract low bands
    Wn,_ = _clamp_normalized_cutoff(cutoff_hz, fs)
    b_lp,a_lp = signal.butter(4, Wn, btype='low')
    orig_low = signal.filtfilt(b_lp,a_lp,original)
    proc_low = signal.filtfilt(b_lp,a_lp,processed)
    proc_high = processed - proc_low
    # RMS match: scale orig_low to match proc_low RMS if needed (to avoid overpowering)
    rms_orig = rms(orig_low) + 1e-12
    rms_proc_low = rms(proc_low) + 1e-12
    # Gain factor to bring orig low closer to proc_low while preserving overall energy
    # We'll scale orig_low so that final low-band RMS = (mix * rms_orig) + ((1-mix) * rms_proc_low)
    target_rms = mix * rms_orig + (1.0 - mix) * rms_proc_low
    scale = target_rms / rms_orig if rms_orig>0 else 1.0
    orig_low_scaled = orig_low * scale
    final = proc_high + orig_low_scaled
    # gentle overall normalization to input peak
    if np.max(np.abs(final))>0:
        final = final / (np.max(np.abs(final))+1e-12) * min(1.0, np.max(np.abs(original)))
    return final

# processing
if st.session_state.get('run_request', False) or sweep_btn:
    st.session_state['run_request'] = False
    with st.spinner("Processing..."):
        # Pre-clean low rumble for Auto path
        precleaned = remove_low_pitch_rumble(noisy_data, fs) if mode.startswith("Auto") else noisy_data.copy()

        results = {}  # map label->(signal,desc)
        if mode == "Auto (aggressive-by-default)":
            # use aggressiveness-derived params
            denoised = spectral_subtract_denoise(precleaned, fs,
                                                 n_fft=n_fft,
                                                 noise_frames=noise_frames,
                                                 alpha=alpha,
                                                 beta=0.0008)
            # if restore enabled, mix back LF using chosen restore_mix (manual override allowed)
            mix_value = float(restore_mix) if restore_mix is not None else float(restore_mix_auto)
            if restore_enabled:
                final = restore_lowband_and_match(noisy_data, denoised, fs, restore_cutoff, mix_value)
            else:
                final = denoised
            results[f"Aggressive (A={aggr_val})"] = (final, f"alpha={alpha:.2f}, nfft={n_fft}, noise_frames={noise_frames}, restore_mix={mix_value:.2f}")
        else:
            # Manual cleaning pipeline
            y = precleaned.copy()
            if man_apply_highpass:
                Wn,_ = _clamp_normalized_cutoff(man_hp_cut, fs)
                b_hp,a_hp = signal.butter(2, Wn, btype='high')
                y = signal.filtfilt(b_hp, a_hp, y)
            if man_lowcut>0:
                y = apply_filters(y, fs, low_cut=man_lowcut, low_order=man_loworder, notch_freq=None)
            if man_notch>0:
                y = apply_filters(y, fs, low_cut=None, low_order=4, notch_freq=man_notch, notch_Q=man_notchQ)
            final = y
            results["Manual"] = (final, "manual filters applied")

        # If sweep requested, build candidates around current aggr_val
        if sweep_btn:
            sweep_vals = sorted(set([max(0,aggr_val-30), max(0,aggr_val-15), aggr_val, min(100,aggr_val+15), min(100,aggr_val+30)]))
            sweep_vals = [v for v in sweep_vals if v>=0 and v<=100]
            for v in sweep_vals:
                a = 2.0 + (6.0 * (v/100.0))
                nf = int(4 + (16 * (v/100.0)))
                nfft_v = 4096 if v>=60 else 2048
                den = spectral_subtract_denoise(precleaned, fs, n_fft=nfft_v, noise_frames=nf, alpha=a, beta=0.0008)
                mix_v = 0.25 + 0.75*(v/100.0)
                if restore_enabled:
                    cand = restore_lowband_and_match(noisy_data, den, fs, restore_cutoff, mix_v)
                else:
                    cand = den
                results[f"Sweep {v}"] = (cand, f"alpha={a:.2f}, nf={nf}, nfft={nfft_v}, mix={mix_v:.2f}")

        # Add a filtered baseline for comparison
        low_cut_auto, notch_auto, use_spectral = auto_design_filters(precleaned, fs)
        filtered_baseline = apply_filters(precleaned, fs, low_cut=low_cut_auto, low_order=4, notch_freq=notch_auto, notch_Q=30)
        results["Filter baseline"] = (filtered_baseline, f"low_cut={low_cut_auto}, notch={notch_auto}")

        # show results: waveform + spectrogram + playback buttons
        st.subheader("Result preview (pick one to listen/download)")
        select_label = st.selectbox("Choose result to preview", options=list(results.keys()))
        sel_sig, sel_desc = results[select_label]
        st.write(f"Params: {sel_desc}")

        # display noisy and selected waveforms using helper (supports mpl or plotly)
        k1, fig1 = display_waveform(noisy_data, fs, title="Noisy (first 5s)", max_seconds=5)
        k2, fig2o = display_waveform(sel_sig, fs, title=f"{select_label} (first 5s)", max_seconds=5)

        # show noisy
        if k1 == "mpl":
            st.pyplot(fig1)
        elif k1 == "plotly":
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.info("Noisy waveform plot unavailable.")

        # show selected
        if k2 == "mpl":
            st.pyplot(fig2o)
        elif k2 == "plotly":
            st.plotly_chart(fig2o, use_container_width=True)
        else:
            st.info(f"{select_label} waveform plot unavailable.")

        st.subheader("Spectrogram (Noisy vs Selected)")
        k1s, spec1 = display_spectrogram(noisy_data, fs, nperseg=2048, title='Spectrogram - Noisy')
        k2s, spec2 = display_spectrogram(sel_sig, fs, nperseg=2048, title=f"Spectrogram - {select_label}")

        if k1s == "mpl" and k2s == "mpl":
            # combine into one mpl figure (2 rows)
            fig_comb, ax_comb = plt.subplots(2,1, figsize=(12,6))
            f1, t1, S1 = signal.spectrogram(noisy_data, fs=fs, nperseg=2048)
            ax_comb[0].pcolormesh(t1, f1, 10*np.log10(S1+1e-12), shading='gouraud'); ax_comb[0].set_title('Spectrogram - Noisy')
            f2, t2, S2 = signal.spectrogram(sel_sig, fs=fs, nperseg=2048)
            ax_comb[1].pcolormesh(t2, f2, 10*np.log10(S2+1e-12), shading='gouraud'); ax_comb[1].set_title(f"Spectrogram - {select_label}")
            st.pyplot(fig_comb)
        else:
            # show whichever are available
            if k1s == "mpl":
                st.pyplot(spec1)
            elif k1s == "plotly":
                st.plotly_chart(spec1, use_container_width=True)
            if k2s == "mpl":
                st.pyplot(spec2)
            elif k2s == "plotly":
                st.plotly_chart(spec2, use_container_width=True)

        # playback/download UI for all candidates (quick A/B)
        st.subheader("Playback / Download candidates (A/B test)")
        cols = st.columns(2)
        i = 0
        for label, (sig, desc) in results.items():
            col = cols[i % 2]
            with col:
                st.write(label)
                buf = io.BytesIO(); write_audio_fileobj(buf, sig, fs); buf.seek(0)
                st.audio(buf.read())
                st.download_button(f"Download {label}", data=buf.getvalue(), file_name=f"{label.replace(' ','_')}.wav")
            i += 1

        # if clean ref exists show SNR improvement for selected
        if clean_ref is not None:
            L = min(len(clean_ref), len(noisy_data), len(sel_sig))
            s_before = compute_snr(clean_ref[:L], noisy_data[:L]); s_after = compute_snr(clean_ref[:L], sel_sig[:L])
            st.success(f"SNR before: {s_before:.2f} dB | after: {s_after:.2f} dB | improvement: {s_after-s_before:.2f} dB")

        # download report (safe)
        pdf_buf = io.BytesIO()
        if MATPLOTLIB_AVAILABLE and PdfPages is not None:
            ok = build_pdf_report_safe(pdf_buf, noisy_data, sel_sig, fs, title="Audio Denoise Report")
            if ok:
                pdf_buf.seek(0)
                st.download_button("Download report (PDF)", data=pdf_buf.getvalue(), file_name="denoise_report.pdf")
            else:
                st.warning("PDF report generation failed even though matplotlib appears available.")
        else:
            st.info("PDF report not available because matplotlib is not installed on the server. To enable reports, add matplotlib to requirements and rebuild the app.")

st.markdown("---")
st.write("Hints: If bass still sounds reduced, increase Aggressiveness slider (this also increases LF restoration mix) or use Auto-tune sweep and pick the candidate that sounds best. If a particular 50/60Hz hum remains, add a manual notch at ~50 or ~60 Hz in Manual mode.")
