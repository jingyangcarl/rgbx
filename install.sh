conda env create -n rgbx -f environment.yml
conda activate rgbx

conda install -c conda-forge git-lfs
git lfs install
git clone https://huggingface.co/zheng95z/x-to-rgb
git clone https://huggingface.co/zheng95z/rgb-to-x
pip install accelerate scipy imageio[ffmpeg] opencv-python

python rgb2x/gradio_demo_rgb2x.py