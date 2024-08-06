# Adapted from whisperx readme
import whisperx
import gc 
import timeit

device = "cuda" 
audio_file = "tests/data/physicsworks.wav"
batch_size = 16 # reduce if low on GPU mem
compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)
model_size = "medium"

def run_test():
    #1. Transcribe with original whisper (batched)
    result = model.transcribe(audio, batch_size=batch_size, language="en")
    print(result["segments"]) # before alignment

# don't include model load in bench
model = whisperx.load_model(model_size, device, compute_type=compute_type)
audio = whisperx.load_audio(audio_file)
print(timeit.timeit("run_test()", globals=locals(), number=1))
