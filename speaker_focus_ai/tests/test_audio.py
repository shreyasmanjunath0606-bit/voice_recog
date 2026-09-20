"""Unit tests for Audio Extraction and Voice Activity Detection (VAD)."""

import os
import tempfile
import unittest
import numpy as np

from speaker_focus_ai.audio.extractor import AudioExtractor
from speaker_focus_ai.audio.vad import VoiceActivityDetector
from speaker_focus_ai.audio.speaker_embedding import SpeakerEmbeddingExtractor
from speaker_focus_ai.audio.enrollment import VoiceprintEnrollmentManager
from speaker_focus_ai.core.types import EnrollmentPath, SpeechSegment


class TestAudioExtractor(unittest.TestCase):
    """Test suite for AudioExtractor."""

    def setUp(self):
        self.extractor = AudioExtractor(target_sample_rate=16000)
        self.temp_dir = tempfile.mkdtemp()

    def test_save_and_extract_wav(self):
        # Create 1 second of synthetic 440Hz sine wave audio
        sample_rate = 16000
        t = np.linspace(0, 1.0, sample_rate, endpoint=False)
        sine_wave = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

        wav_path = os.path.join(self.temp_dir, "test_synth.wav")
        saved_path = self.extractor.save_wav(sine_wave, sample_rate, wav_path)

        self.assertTrue(os.path.exists(saved_path))

        # Extract waveform back
        waveform, sr = self.extractor.extract_waveform(saved_path)

        self.assertEqual(sr, 16000)
        self.assertEqual(waveform.dtype, np.float32)
        self.assertEqual(waveform.ndim, 1)
        self.assertLessEqual(np.max(np.abs(waveform)), 1.0)
        self.assertGreater(len(waveform), 0)


class TestVoiceActivityDetector(unittest.TestCase):
    """Test suite for VoiceActivityDetector."""

    def setUp(self):
        self.vad = VoiceActivityDetector(threshold=0.5)

    def test_silence_detection(self):
        # 2 seconds of complete silence
        silence = np.zeros(16000 * 2, dtype=np.float32)
        segments = self.vad.get_speech_timestamps(silence, sample_rate=16000)
        self.assertEqual(len(segments), 0)

    def test_speech_segment_detection(self):
        # Generate 1 sec silence + 1 sec tone + 1 sec silence
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        speech_sim = 0.8 * np.sin(2 * np.pi * 300 * t).astype(np.float32)
        
        # Add random harmonics to simulate voice spectrum
        speech_sim += 0.3 * np.sin(2 * np.pi * 600 * t).astype(np.float32)
        
        audio = np.concatenate([np.zeros(sr, dtype=np.float32), speech_sim, np.zeros(sr, dtype=np.float32)])
        segments = self.vad.get_speech_timestamps(audio, sample_rate=sr)

        self.assertIsInstance(segments, list)
        for seg in segments:
            self.assertIsInstance(seg, SpeechSegment)
            self.assertGreaterEqual(seg.start_sec, 0.0)
            self.assertGreater(seg.end_sec, seg.start_sec)


class TestSpeakerEmbeddingExtractor(unittest.TestCase):
    """Test suite for SpeakerEmbeddingExtractor."""

    def setUp(self):
        self.extractor = SpeakerEmbeddingExtractor()

    def test_embedding_shape_and_norm(self):
        # Generate 1.5 seconds of synthetic audio
        sr = 16000
        t = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False)
        audio = (0.5 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.random.randn(len(t))).astype(np.float32)

        emb = self.extractor.compute_embedding(audio, sample_rate=sr)

        self.assertEqual(emb.shape, (192,))
        self.assertEqual(emb.dtype, np.float32)
        self.assertAlmostEqual(float(np.linalg.norm(emb)), 1.0, places=4)

    def test_similarity(self):
        # Test cosine similarity calculation
        emb1 = np.ones((192,), dtype=np.float32) / np.sqrt(192)
        emb2 = np.ones((192,), dtype=np.float32) / np.sqrt(192)
        sim_identical = self.extractor.compute_similarity(emb1, emb2)
        self.assertAlmostEqual(sim_identical, 1.0, places=4)

        emb3 = -emb1
        sim_opposite = self.extractor.compute_similarity(emb1, emb3)
        self.assertAlmostEqual(sim_opposite, -1.0, places=4)


class TestVoiceprintEnrollmentManager(unittest.TestCase):
    """Test suite for VoiceprintEnrollmentManager."""

    def setUp(self):
        self.manager = VoiceprintEnrollmentManager(min_solo_duration_sec=1.5)

    def test_find_clean_segments(self):
        asd_timelines = {
            "person_1": [(0.0, 2.0, 0.95), (2.0, 4.0, 0.90)],
            "person_2": [(0.0, 2.0, 0.02), (2.0, 4.0, 0.88)] # overlaps during 2.0-4.0
        }
        clean_segs = self.manager.find_clean_segments("person_1", asd_timelines)
        
        self.assertEqual(len(clean_segs), 1)
        self.assertEqual(clean_segs[0], (0.0, 2.0))

    def test_clean_enrollment(self):
        sr = 16000
        t = np.linspace(0, 3.0, sr * 3, endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        asd_timelines = {
            "target": [(0.0, 2.0, 0.95)],
            "other": [(0.0, 2.0, 0.02)]
        }
        
        emb, path, conf = self.manager.enroll_target("target", audio, asd_timelines, sample_rate=sr)
        
        self.assertEqual(path, EnrollmentPath.CLEAN)
        self.assertEqual(emb.shape, (192,))
        self.assertGreater(conf, 0.80)

    def test_provisional_fallback_enrollment(self):
        sr = 16000
        t = np.linspace(0, 2.0, sr * 2, endpoint=False)
        mixed_audio = np.random.randn(sr * 2).astype(np.float32)
        
        # Target speaks during 0.0-1.0s, but other person also speaks (no solo clean segment)
        asd_timelines = {
            "target": [(0.0, 1.0, 0.95), (1.0, 2.0, 0.05)],
            "other": [(0.0, 1.0, 0.85), (1.0, 2.0, 0.90)]
        }
        
        # Stream 0 has high energy during 0.0-1.0s (matches target ASD)
        stream0 = np.zeros(sr * 2, dtype=np.float32)
        stream0[0:sr] = 0.8 * np.sin(2 * np.pi * 300 * t[:sr])
        
        # Stream 1 has high energy during 1.0-2.0s
        stream1 = np.zeros(sr * 2, dtype=np.float32)
        stream1[sr:sr*2] = 0.8 * np.sin(2 * np.pi * 300 * t[:sr])
        
        emb, path, conf = self.manager.enroll_target(
            "target", mixed_audio, asd_timelines, separated_streams_fallback=[stream0, stream1], sample_rate=sr
        )
        
        self.assertEqual(path, EnrollmentPath.PROVISIONAL)
        self.assertEqual(emb.shape, (192,))
        self.assertGreater(conf, 0.40)


from speaker_focus_ai.audio.separator import AudioSeparator
from speaker_focus_ai.audio.enhancer import SpeechEnhancer


class TestAudioSeparator(unittest.TestCase):
    """Test suite for AudioSeparator."""

    def setUp(self):
        self.separator = AudioSeparator()

    def test_blind_separate(self):
        sr = 16000
        t = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False)
        mixed = (0.5 * np.sin(2 * np.pi * 300 * t) + 0.5 * np.sin(2 * np.pi * 800 * t)).astype(np.float32)

        streams = self.separator.blind_separate(mixed, sample_rate=sr)

        self.assertIsInstance(streams, list)
        self.assertGreaterEqual(len(streams), 1)
        for s in streams:
            self.assertEqual(s.dtype, np.float32)

    def test_separate_target(self):
        sr = 16000
        t = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False)
        mixed = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        voiceprint = np.ones((192,), dtype=np.float32) / np.sqrt(192)

        target_audio, conf = self.separator.separate_target(mixed, sample_rate=sr, voiceprint=voiceprint)

        self.assertEqual(target_audio.dtype, np.float32)
        self.assertGreaterEqual(conf, 0.0)


class TestSpeechEnhancer(unittest.TestCase):
    """Test suite for SpeechEnhancer."""

    def setUp(self):
        self.enhancer = SpeechEnhancer()

    def test_enhance(self):
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        noisy_audio = (0.5 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(sr)).astype(np.float32)

        enhanced = self.enhancer.enhance(noisy_audio, sample_rate=sr)

        self.assertEqual(enhanced.dtype, np.float32)
        self.assertEqual(len(enhanced), len(noisy_audio))
        self.assertLessEqual(np.max(np.abs(enhanced)), 1.0)


from speaker_focus_ai.speech.transcription import SpeechTranscriber
from speaker_focus_ai.core.types import TranscriptSegment


class TestSpeechTranscriber(unittest.TestCase):
    """Test suite for SpeechTranscriber."""

    def setUp(self):
        self.transcriber = SpeechTranscriber(model_size="tiny.en")

    def test_transcribe_empty_waveform(self):
        segments = self.transcriber.transcribe(np.array([], dtype=np.float32), speaker_id="target_01")
        self.assertEqual(len(segments), 0)

    def test_transcribe_synthetic_audio(self):
        sr = 16000
        # 1.5 second synthetic audio
        t = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False)
        audio = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

        segments = self.transcriber.transcribe(audio, sample_rate=sr, speaker_id="target_01")
        self.assertIsInstance(segments, list)
        for seg in segments:
            self.assertIsInstance(seg, TranscriptSegment)
            self.assertEqual(seg.speaker, "target_01")


if __name__ == "__main__":
    unittest.main()
