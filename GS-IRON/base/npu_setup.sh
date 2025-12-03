echo start setup
echo set PATH to Xilinx tools
source /opt/xilinx/xrt/setup.sh
echo activate ironenv
cd ../../mlir-aie
source ironenv/bin/activate
source utils/env_setup.sh
cd ../GS-IRON/npu-1
echo setup done