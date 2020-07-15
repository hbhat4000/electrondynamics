# check GPUs
# nvidia-smi
# cat /usr/local/cuda/version.txt

# install GPU-enabled JAX
# pip install --upgrade https://storage.googleapis.com/jax-releases/cuda101/jaxlib-0.1.48-cp37-none-linux_x86_64.whl
# pip install --upgrade jax

# pip install --upgrade jaxlib
# pip install --upgrade jax

# train
python hamtrainexp.py > ./lihLINEAR/lihresults1.txt

# test
python hamfieldpaper.py > ./lihLINEAR/lihresults2.txt

