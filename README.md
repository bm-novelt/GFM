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
4. Clone [apex](https://github.com/NVIDIA/apex) and run commands:
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
5. Install