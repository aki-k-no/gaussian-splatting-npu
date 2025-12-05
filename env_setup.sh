# before run this .sh file, you should install NPU drivers

# setup xrt
source /opt/xilinx/xrt/setup.sh

# setup venv for mlir-aie project
cd mlir-aie
python3 -m venv ironenv
source ironenv/bin/activate

# Install IRON library and mlir-aie from a wheel
python3 -m pip install mlir_aie -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/v1.1.0

# Install Peano from a llvm-aie wheel
python3 -m pip install https://github.com/Xilinx/llvm-aie/releases/download/nightly/llvm_aie-20.0.0.2025090701+8c084497-py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl

# Install MLIR Python Extras
HOST_MLIR_PYTHON_PACKAGE_PREFIX=aie python3 -m pip install -r python/requirements_extras.txt

# setup mlir-aie with .sh
source utils/env_setup.sh

