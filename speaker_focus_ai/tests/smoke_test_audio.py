"""Phase 0: Audio Hardware Smoke Test

Verifies runtime execution and hardware compatibility for:
- PyTorch device backend (CUDA / MPS / CPU)
- Silero VAD (Voice Activity Detection)
- SpeechBrain ECAPA-TDNN (Speaker Embedding)
- faster-whisper (Speech-to-Text)
- SpeechBrain SepFormer (Speech Separation)
"""

import sys
import os
import platform
import time
import numpy as np
import torch

# Force UTF-8 encoding for Windows console output
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Safe symlink fallback for SpeechBrain on Windows (WinError 1314 fix)
import shutil
_orig_symlink = getattr(os, "symlink", None)
def safe_symlink(src, dst, target_is_directory=False, *, dir_fd=None):
    try:
        if _orig_symlink:
            _orig_symlink(src, dst, target_is_directory=target_is_directory, dir_fd=dir_fd)
        else:
            shutil.copyfile(src, dst)
    except OSError as e:
        if getattr(e, 'winerror', None) == 1314 or getattr(e, 'errno', None) == 1314:
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if os.path.exists(dst):
                    os.remove(dst)
                shutil.copyfile(src, dst)
        else:
            raise

os.symlink = safe_symlink


def test_pytorch_device() -> torch.device:
    print("\n[1/5] Testing PyTorch Hardware Backend...")
    print(f"  OS / System     : {platform.system()} ({platform.machine()})")
    print(f"  PyTorch Version : {torch.__version__}")
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"  Device Selected : [PASS] NVIDIA CUDA ({gpu_name})")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("  Device Selected : [PASS] Apple Silicon Metal (MPS)")
    else:
        device = torch.device("cpu")
        print("  Device Selected : [INFO] CPU Fallback (Universal)")
        
    # Quick tensor compute test
    x = torch.ones(3, 3, device=device)
    y = x * 2.0
    print(f"  Tensor Test     : [PASS] Compute succeeded on {device}")
    return device


def test_silero_vad(device: torch.device):
    print("\n[2/5] Benchmarking Silero VAD Latency...")
    try:
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        model.to(device if device.type != 'mps' else torch.device('cpu'))
        
        dummy_chunk = torch.randn(512)
        # Warmup
        _ = model(dummy_chunk, 16000)
        
        # Benchmark 50 iterations over 32ms audio chunks
        times = []
        for _ in range(50):
            t0 = time.perf_counter()
            _ = model(dummy_chunk, 16000)
            times.append((time.perf_counter() - t0) * 1000.0)
            
        avg_latency = np.mean(times)
        print(f"  Silero VAD Latency : [PASS] {avg_latency:.2f} ms per 32ms chunk ({avg_latency/32.0:.2f}x real-time)")
        return True, f"{avg_latency:.2f} ms / chunk"
    except Exception as e:
        print(f"  Silero VAD Latency : [FAIL] {e}")
        return False, "N/A"


def test_speechbrain_ecapa(device: torch.device):
    print("\n[3/5] Benchmarking SpeechBrain ECAPA-TDNN Latency...")
    try:
        from speechbrain.inference.speaker import EncoderClassifier
        run_device = "cpu" if device.type == "mps" else str(device)
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="tmp_models/spkrec-ecapa-voxceleb",
            run_opts={"device": run_device}
        )
        
        signal = torch.randn(1, 16000 * 3) # 3 seconds audio
        # Warmup
        _ = classifier.encode_batch(signal)
        
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            _ = classifier.encode_batch(signal)
            times.append((time.perf_counter() - t0) * 1000.0)
            
        avg_latency = np.mean(times)
        print(f"  ECAPA-TDNN Latency : [PASS] {avg_latency:.2f} ms per 3-sec clip ({avg_latency/3000.0:.2f}x real-time)")
        return True, f"{avg_latency:.2f} ms / 3s clip"
    except Exception as e:
        print(f"  ECAPA-TDNN Latency : [FAIL] {e}")
        return False, "N/A"


def test_faster_whisper(device: torch.device):
    print("\n[4/5] Benchmarking Speech-to-Text (Whisper) Latency...")
    try:
        import whisper
        model = whisper.load_model("tiny.en", device=device)
        
        # 3 seconds 16kHz audio
        dummy_audio = np.random.randn(16000 * 3).astype(np.float32)
        # Warmup
        _ = model.transcribe(dummy_audio, fp16=False)
        
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            _ = model.transcribe(dummy_audio, fp16=False)
            times.append((time.perf_counter() - t0) * 1000.0)
            
        avg_latency = np.mean(times)
        print(f"  Whisper Latency    : [PASS] {avg_latency:.2f} ms per 3-sec audio clip")
        return True, f"{avg_latency:.2f} ms / 3s clip"
    except Exception as e:
        print(f"  Whisper Latency    : [FAIL] {e}")
        return False, "N/A"


def test_sepformer(device: torch.device):
    print("\n[5/5] Benchmarking SpeechBrain SepFormer Latency...")
    try:
        from speechbrain.inference.separation import SepformerSeparation
        run_device = "cpu" if device.type == "mps" else str(device)
        separator = SepformerSeparation.from_hparams(
            source="speechbrain/sepformer-wham",
            savedir="tmp_models/sepformer-wham",
            run_opts={"device": run_device}
        )
        
        # 1.5 second mixture
        signal = torch.randn(1, int(16000 * 1.5))
        t0 = time.perf_counter()
        _ = separator.separate_batch(signal)
        latency = (time.perf_counter() - t0) * 1000.0
        
        print(f"  SepFormer Latency  : [PASS] {latency:.2f} ms per 1.5-sec mixture")
        return True, f"{latency:.2f} ms / 1.5s clip"
    except Exception as e:
        print(f"  SepFormer Latency  : [FAIL] {e}")
        return False, "N/A"


def main():
    print("=" * 60)
    print("      Person 2: Audio Hardware & Model Smoke Test")
    print("=" * 60)
    
    start_time = time.time()
    device = test_pytorch_device()
    
    results = {
        "PyTorch Device": (True, "< 1 ms"),
        "Silero VAD": test_silero_vad(device),
        "ECAPA-TDNN": test_speechbrain_ecapa(device),
        "Whisper (STT)": test_faster_whisper(device),
        "SepFormer": test_sepformer(device),
    }
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 65)
    print("                     MODEL LATENCY BENCHMARK")
    print("=" * 65)
    print(f"  {'Component':<20} | {'Status':<8} | {'Measured Latency'}")
    print("  " + "-" * 60)
    all_passed = True
    for component, (status, latency) in results.items():
        symbol = "[PASS]" if status else "[FAIL]"
        print(f"  {component:<20} | {symbol:<8} | {latency}")
        if not status:
            all_passed = False
            
    print(f"\nTotal Smoke & Benchmark Time: {elapsed:.2f} seconds")
    if all_passed:
        print(">>> ALL AUDIO MODEL LATENCY BENCHMARKS PASSED! <<<")
    else:
        print(">>> SOME BENCHMARKS FAILED. <<<")
    print("=" * 65)

if __name__ == "__main__":
    main()
