# 1. 屏蔽 0 号卡，让当前终端只看到 1-7 号卡
#    这样对 Ray 来说，物理上的 1 号卡就变成了逻辑上的 "0 号卡"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RAY_TMPDIR="$HOME/my_ray_tmp"
# 2. 启动 Ray，并将 GPU 数量设为 7
ray start --head \
    --port=6380 \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8266 \
    --dashboard-agent-listen-port=52366 \
    --dashboard-agent-grpc-port=52367 \
    --include-dashboard=True \
    --num-gpus=8