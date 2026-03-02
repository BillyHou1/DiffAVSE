# Billy
# AV inference, takes noisy audio + optional video, outputs enhanced audio.
#
# TODO implement inference pipeline
#
# Two modes: single file with --input_audio + --input_video, or folder mode
# with --input_folder which expects paired .wav and .mp4/.mpg with matching
# names. If video is missing just run audio-only with video=None, don't crash.
#
# Pipeline: load audio -> RMS normalize -> STFT -> model forward -> iSTFT ->
# undo normalize -> save wav. Reuse load_video_frames from dataloader_av.py.
