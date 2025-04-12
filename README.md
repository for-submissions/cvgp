## Instructions

Using anacoda;

1. Create a new conda environment:

`conda create -n "cvgp_env" python=3.9.0 anaconda`

2. Iniate the environment:

`conda activate cvgp_env`

3. Install requirements:

`pip install -r ./requirements.txt`

4. Install torch (see: https://pytorch.org/):

__windows__: `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117`


__linux__: `pip3 install torch torchvision torchaudio`


__mac__: `pip3 install torch torchvision torchaudio`

5. Install UCI datasets:

`python -m pip install git+https://github.com/treforevans/uci_datasets.git`

6. Run:

`python main.py --epochs 1000 --ip_size 50 --loss CVTGP --cv_folds 5 --dataset parkinsons --check_stop 150 --early_stop 20 --save_metric rmse --device cuda`

The code is preliminary and will change. E.g., CVTGP in code -> CVGP in paper.