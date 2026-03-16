mount -t tracefs nodev /sys/kernel/tracing
mount -t debugfs nodev /sys/kernel/debug
echo 0 > /sys/kernel/debug/tracing/tracing_on
#ixsys -o bevfusion.trace -t cuda,cudnn,osrt,power,mem  bash train.sh
ixsys -o bevfusion.trace -t cuda,cudnn,osrt,power,mem --print-gpu-summary bash train.sh
