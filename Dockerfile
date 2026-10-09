FROM ajay/ms-swift-official:4.5.3-cu130

# Official base config SHA256:
# 81d1d940c5b76e10a5426a60d522ba8d72dd1b54bf063d2b5348ce8b09ba2f29
# Repair dependency conflicts in the upstream all-in-one image.
# These optional packages are not required for text SFT/GRPO.
COPY wheels/ /tmp/swift-wheels/
COPY requirements-fixes.txt /tmp/requirements-fixes.txt
RUN python -m pip uninstall -y ms-agent ms-opencompass decord gradio \
    && python -m pip install --no-cache-dir --no-index --find-links=/tmp/swift-wheels -r /tmp/requirements-fixes.txt \
    && python -m pip check \
    && rm -rf /tmp/swift-wheels /tmp/requirements-fixes.txt

ENV VLLM_NO_USAGE_STATS=1 \
    DO_NOT_TRACK=1 \
    TOKENIZERS_PARALLELISM=false
