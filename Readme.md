How to install

Option A - Using requirements.txt
# 1. Create & activate venv
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows

# 2. Install PyTorch FIRST (needs special index for +cu118)
pip install torch==2.1.0+cu118 torchaudio==2.1.0+cu118 torchvision==0.16.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# 3. Install everything else
pip install -r requirements.txt


Option B - Using pyproject.toml
# Same step 1 & 2 as above, then:
pip install .