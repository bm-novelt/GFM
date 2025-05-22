# Setting up
1. Clone GFM.
2. Create uv venv by running
    ```bash
    uv init
    ```
3. download dependencies:
    ```bash
    uv add torch pyyaml scipy termcolor timm yacs torchmetrics rasterio torchgeo opencv-python
    ```
4. *[OPTIONAL: only for speeding up, you can ommit the dependency to apex by using the parameter `--amp-opt-level O0` when running scripts]* 
    
    Clone [apex](https://github.com/NVIDIA/apex) and run commands:
    ```bash
    git clone https://github.com/NVIDIA/apex
    ```
    ```bash
    cd apex
    ```
    ```bash
    rm pyproject.toml
    ```
    ```bash
    uv run setup.py install
    ```
    ```bash
    uv run pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
    ```
5. Download weights from imagenet
    ```
    wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth
    ```
    and place it under the following folder structure.
    ```
    output
    |- simmim_finetune
        |- swin_base_patch4_window7_224_22k.pth
    ```
6. Download GeoPile from NAS-3: `"\\NAS-3\Imagery\ai-internship-2025\geopile\GeoPile.zip"`, place the file in `data` and unzip it to obtain the GeoPileV0 folder (might take several minutes to unzip).
7. To train their fundation model, you can run the `main_teacher.py` script as follows:
    ```bash
    uv run -m torch.distributed.run --nproc_per_node 1 main_teacher.py --cfg configs/simmim_pretrain__swin_base__img192_window6__100ep.yaml --batch-size 1 --data-path data/GeoPileV0 --tag gfm --pretrained output/simmim_finetune/swin_base_patch4_window7_224_22k.pth --amp-opt-level O0
    ```
    This is slightly different from GFM instruction because the `torch.distributed` library evolved. Note that we use a single GPU and a batch size that is much smaller because of current RAM issues. Depending on your set, adapt those parameters.