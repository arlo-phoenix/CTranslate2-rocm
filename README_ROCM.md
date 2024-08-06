# ROCm CT2

## Install Guide

These install instructions are for https://hub.docker.com/r/rocm/pytorch. They should mostly work for system installs as well, but then you'll have to change install directories and make sure all dependencies are installed (in the image they are already present in the conda env)

after following the guide in https://hub.docker.com/r/rocm/pytorch (tested for latest 95ac9ef9b5ec (**ROCm 6.1**))

```bash
#init conda
conda init
bash
conda activate py_3.9
```

```bash
git clone https://github.com/arlo-phoenix/CTranslate2-rocm.git
cd CTranslate2-rocm
git checkout rocm
CLANG_CMAKE_CXX_COMPILER=clang++ CXX=clang++ HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)"     cmake -S . -B build -DWITH_MKL=OFF -DWITH_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1030 -DBUILD_TESTS=ON -DWITH_CUDNN=ON
cmake --build build -- -j16
cd build
cmake --install . --prefix $CONDA_PREFIX #or just sudo make install if not using conda env
sudo ldconfig
cd ../python
pip install -r install_requirements.txt
python setup.py bdist_wheel
pip install dist/*.whl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
```

## Running tests / debugging issues

### Tests

In CT2 project root folder:

```bash
./build/tests/ctranslate2_test ./tests/data/ --gtest_filter=*CUDA*:-*bfloat16*
```
for me only some int8 test failed (I think that test shouldn't even be run for CUDA, but didn't check too deeply. The guard is from CT2 itself so it's supposed to fail)

### Checking that all libraries are found

`ld -lctranslate2 --verbose` (ignore warnings, only important thing is that it doesn't find link errors)

### BF16 issues

This fork just commented out everything related to bf16. I think an implicit conversion operator from 
`__hip_bfloat16` to `float` is missing

example error with bf16 enabled:
```cpp
CTranslate2/src/cuda/primitives.cu:284:19: error: no viable conversion from 'const __hip_bfloat16' to 'const float'
  284 |       const float score = previous_scores[i];
```

Other than that I **won't** be adding FA2 or AWQ support. It's written with assembly for cuda and it isn't helpful at all for my use case (whisper). Otherwise on this older commit (besides bf16) this fork is feature complete, so I might look into cleaning it up and possibilities of disabling these for ROCm on master for upstreaming. But I'll only do that **after BF16 gets proper support** since this discrepancy adds way too many different code paths between ROCm and CUDA. Other than that conversion worked quite well, I only had to change a couple defines, otherwise it hipified well. Only the `conv1d OP` required a custom implementation for MIOpen (hipDNN isn't maintained anymore).

## Tested libraries

### faster_whisper
```bash
pip install faster_whisper 

#1.0.3 was the most recent version when I made this, so try testing that one first if a newer one doesn't work
#pip install faster_whisper==1.0.3 
```

I included a small benchmark script in this CT2 fork. You need to download a test file from the faster whisper repo
```bash
wget -P "./tests/data" https://github.com/SYSTRAN/faster-whisper/raw/master/tests/data/physicsworks.wav 
```

Then you should be able to run. This per default does just one testrun with the medium model

```bash
python faster_whisper_bench.py
```

I'm getting around `11s-12s` on my RX6800 (with model loading included `14s-15s`). 


### whisperx

System dependency is just ffmpeg. Either use your system package mangager or with conda `conda install conda-forge::ffmpeg`

```bash
pip install whisperx
```
Python dependencies are a mess here since versions aren't really pinned. I'd recommend just installing faster_whisper from master so you don't run into a bunch of version conflicts. I personally did get it running by pinning numpy==1.23 and then trial and error rolling back stuff till it worked (albeit with a bunch of warnings).

For running you can use its great cli-tool by just using `whisperx path/to/audio` or running my little bench script for the `medium` model.

```bash
python whisperx_bench.py
```

this took `4.4s` with language detection and around `4.2s` without.

If you do get it running it's pretty fast. I excluded model load since that one takes quite a while. With model load it was only slightly faster than faster_whisper, but I think that's connected with the bunch of version conflicts I had. The main advantage of `whisperx` is its great feature set (Forced Alignment, VAD, Speaker Diarization) and the cli-tool (lots of output options), so do try and get it running it's worth it.