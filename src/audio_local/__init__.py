"""Local TTS runtime — CosyVoice-300M-SFT (FunAudioLLM open weights) via
the official cosyvoice inference package.

Mirrors src/img_local in shape: a manager singleton with an explicit
state machine (UNLOADED / LOADING / READY / SYNTHESIZING / ERROR) wraps
a runtime class that does the actual work. AudioGenerator routes to it
when TTS_PROVIDER=local.

This module is import-cheap: the cosyvoice package is only imported on
first load(), so the rest of the app keeps starting even if the
optional TTS deps aren't installed yet."""
