#!/bin/bash
# 在 uv venv 中安装 deepep 和 deepgemm 的完整脚本

set -e

CSRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Installing DeepEP (commit: 73b6ea4)"
echo "=========================================="
cd "${CSRC_DIR}/deepep"

# 清理旧构建
rm -rf build dist *.egg-info

# 构建 wheel
uv run python setup.py bdist_wheel

# 用 uv pip 安装
uv pip install dist/*.whl

echo ""
echo "=========================================="
echo "DeepEP installed successfully!"
echo "=========================================="

echo ""
echo "=========================================="
echo "Installing DeepGEMM (commit: 594953a)"
echo "=========================================="
cd "${CSRC_DIR}/deepgemm"

# 初始化子模块（如果没有的话）
if [ ! -f "third-party/cutlass/README.md" ]; then
    echo "Initializing git submodules..."
    git submodule update --init --recursive
fi

# 清理旧构建
rm -rf build dist *.egg-info

# 构建 wheel
uv run python setup.py bdist_wheel

# 用 uv pip 安装
uv pip install dist/*.whl

echo ""
echo "=========================================="
echo "DeepGEMM installed successfully!"
echo "=========================================="

echo ""
echo "Verifying installations..."
uv run python -c "import deep_ep; print(f'deep_ep version: {deep_ep.__version__}')" 2>/dev/null || echo "deep_ep import failed"
uv run python -c "import deep_gemm; print(f'deep_gemm version: {deep_gemm.__version__}')" 2>/dev/null || echo "deep_gemm import failed"

echo ""
echo "All done!"
