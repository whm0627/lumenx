import os
import time
from typing import Dict, Any, List
from .models import StoryboardFrame, Character, GenerationStatus
from ...utils import get_logger
from ...audio.tts import TTSProcessor

logger = get_logger(__name__)

class AudioGenerator:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.output_dir = self.config.get('output_dir', 'output/audio')

        # TTS provider is environment-driven so the user can flip between
        # cloud DashScope CosyVoice and the local CosyVoice2-0.5B runtime
        # without code changes (mirrors IMAGE_PROVIDER for image gen).
        try:
            self.tts = self._build_tts(self.config.get('tts', {}))
            logger.info(f"TTS Processor initialized ({type(self.tts).__name__})")
        except ValueError:
            # Unknown TTS_PROVIDER — surface as a hard error rather than
            # silently falling back, so misconfiguration is loud.
            raise
        except Exception as e:
            logger.warning(f"Failed to initialize TTS Processor: {e}. Using mock mode.")
            self.tts = None

    @staticmethod
    def _build_tts(tts_config: Dict[str, Any]):
        """Pick TTS backend based on TTS_PROVIDER env var.

        Mirrors AssetGenerator._build_model / StoryboardGenerator._build_model:
        a single env var flips every dialogue render between cloud and local.
        Local provider routes through AudioModelManager (singleton) so all
        AudioGenerator instances share one loaded CosyVoice2 model."""
        provider = os.getenv("TTS_PROVIDER", "dashscope").lower()
        if provider == "local":
            from ...audio_local.manager import AudioModelManager
            return AudioModelManager.get(tts_config)
        if provider == "dashscope":
            return TTSProcessor()
        raise ValueError(
            f"Unknown TTS_PROVIDER={provider!r}; expected 'dashscope' or 'local'"
        )

    def get_available_voices(self) -> List[Dict[str, str]]:
        """Returns a list of available voices for the *active* TTS provider.

        Cloud and local CosyVoice expose disjoint voice ID spaces (cloud
        = Alibaba's longxxxx_v2 series, local = CosyVoice2-0.5B's 中文女/
        中文男 presets) — characters bind voice_id at assignment time, and
        a voice from the wrong provider just fails at synthesis. Surface
        only voices that actually work right now."""
        provider = os.getenv("TTS_PROVIDER", "dashscope").lower()
        if provider == "local":
            from ...audio_local.manager import AudioModelManager
            voices = AudioModelManager.get().list_voices()
            return [
                {
                    "id": v["id"],
                    "name": f"{v.get('name', v['id'])} - CosyVoice2 (本地)",
                    "gender": v.get("gender", "Unknown"),
                    "model": "cosyvoice2-0.5b",
                }
                for v in voices
            ]
        # Cloud DashScope branch (default)
        if self.tts:
            voices_dict = TTSProcessor.list_voices()
            return [
                {"id": key, "name": f"{meta['name']} - CosyVoice", "gender": meta.get('gender', 'Unknown'), "model": meta.get('model', 'cosyvoice-v2')}
                for key, meta in voices_dict.items()
            ]
        else:
            return [
                {"id": "longxiaochun", "name": "龙小淳 (知性女) - CosyVoice", "gender": "Female"},
                {"id": "longyue", "name": "龙悦 (温柔女) - CosyVoice", "gender": "Female"},
                {"id": "longcheng", "name": "龙诚 (睿智青年) - CosyVoice", "gender": "Male"},
                {"id": "longshu", "name": "龙书 (播报男) - CosyVoice", "gender": "Male"},
            ]

    def generate_dialogue(self, frame: StoryboardFrame, character: Character, speed: float = 1.0, pitch: float = 1.0, volume: int = 50) -> StoryboardFrame:
        """Generates TTS audio for the dialogue."""
        if not frame.dialogue:
            return frame

        frame.status = GenerationStatus.PROCESSING

        text = frame.dialogue

        logger.info(f"Generating dialogue for {character.name}: {text} (Speed: {speed}, Pitch: {pitch}, Volume: {volume})")

        if not self.tts:
            frame.status = GenerationStatus.FAILED
            frame.audio_error = "TTS service not available. Check DASHSCOPE_API_KEY configuration."
            logger.warning(f"TTS not initialized, cannot generate audio for frame {frame.id}")
            return frame

        if not character.voice_id:
            frame.status = GenerationStatus.FAILED
            frame.audio_error = f"No voice assigned to character '{character.name}'. Please assign a voice first."
            logger.warning(f"No voice_id for character {character.name}, cannot generate audio")
            return frame

        return self._real_generate_dialogue(frame, character, text, speed, pitch, volume)

    def _real_generate_dialogue(self, frame: StoryboardFrame, character: Character, text: str, speed: float, pitch: float, volume: int) -> StoryboardFrame:
        """Generate dialogue using real TTS."""
        try:
            output_path = os.path.join(self.output_dir, 'dialogue', f"{frame.id}.mp3")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Use character's assigned voice
            voice = character.voice_id
            
            # Call TTSProcessor with speed/pitch
            self.tts.synthesize(text, output_path, voice=voice, speech_rate=speed, pitch_rate=pitch, volume=volume)
            
            # Store relative path for frontend serving
            rel_path = os.path.relpath(output_path, "output")
            frame.audio_url = rel_path
            frame.status = GenerationStatus.COMPLETED
            
        except Exception as e:
            logger.error(f"TTS generation failed for frame {frame.id}: {e}")
            frame.status = GenerationStatus.FAILED
            frame.audio_error = f"TTS generation failed: {str(e)}"
            
        return frame

    def _mock_generate_dialogue(self, frame: StoryboardFrame, character: Character, text: str, speed: float, pitch: float, volume: int) -> StoryboardFrame:
        """Mock fallback — marks frame as FAILED instead of writing dummy bytes."""
        frame.status = GenerationStatus.FAILED
        frame.audio_error = "TTS service unavailable (mock mode)"
        logger.warning(f"Mock generate_dialogue called for frame {frame.id} — marking as FAILED")
        return frame

    def generate_sfx(self, frame: StoryboardFrame) -> StoryboardFrame:
        """Generates sound effects for the frame."""
        frame.status = GenerationStatus.PROCESSING
        
        try:
            # TODO: Implement actual SFX call (e.g., MMAudio)
            # For now, we mock it.
            logger.info(f"Generating SFX for: {frame.action_description}")
            
            output_path = os.path.join(self.output_dir, 'sfx', f"{frame.id}.mp3")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Create a dummy file
            with open(output_path, 'wb') as f:
                f.write(b'dummy sfx content')
                
            # Store relative path for frontend serving
            rel_path = os.path.relpath(output_path, "output")
            frame.sfx_url = rel_path
            frame.status = GenerationStatus.COMPLETED
            
        except Exception as e:
            logger.error(f"Failed to generate SFX for frame {frame.id}: {e}")
            frame.status = GenerationStatus.FAILED
            
        return frame

    def generate_sfx_from_video(self, frame: StoryboardFrame) -> StoryboardFrame:
        """Generates SFX based on video content (Video-to-Audio)."""
        if not frame.video_url:
            return frame
            
        logger.info(f"Generating SFX from video for frame {frame.id}")
        # Mock V2A Logic
        time.sleep(1)
        
        output_path = os.path.join(self.output_dir, 'sfx', f"{frame.id}_v2a.mp3")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'wb') as f:
            f.write(b'dummy v2a sfx content')
            
        frame.sfx_url = os.path.relpath(output_path, "output")
        return frame

    def generate_bgm(self, frame: StoryboardFrame) -> StoryboardFrame:
        """Generates BGM based on frame context."""
        logger.info(f"Generating BGM for frame {frame.id}")
        # Mock MusicGen Logic
        time.sleep(1)
        
        output_path = os.path.join(self.output_dir, 'bgm', f"{frame.id}.mp3")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'wb') as f:
            f.write(b'dummy bgm content')
            
        frame.bgm_url = os.path.relpath(output_path, "output")
        return frame
