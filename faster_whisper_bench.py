#adapted from faster_whisper README

from faster_whisper import WhisperModel
import timeit

def run_test():
    segments, info = model.transcribe("tests/data/physicsworks.wav", beam_size=5, language="en")
    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

#don't include model load in bench
model_size = "medium"
model = WhisperModel(model_size, device="cuda", compute_type="float16")
print(timeit.timeit("run_test()", globals=locals(), number=1))
