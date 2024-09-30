__conda_setup="$('/data/input/claran/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/data/input/claran/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/data/input/claran/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/data/input/claran/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup

conda activate inf-carbon

huggingface-cli login --token $HF_TOKEN
wandb login --relogin $WANDB_KEY

cp /data/input/claran/.codecarbon.config-$4 ~/.codecarbon.config
ipnport=$(shuf -i30000-31999 -n1)

MODEL=$1
python -m sglang.launch_server --model-path $MODEL --enable-torch-compile --disable-radix-cache --port $ipnport & server=localhost:$ipnport/health
timeout=1200   # 20 minutes in seconds
interval=10    # Interval between pings

# Start the timer
elapsed=0

while ! curl "$server" &> /dev/null; do
    echo "Waiting for server to respond..."
    sleep $interval
    elapsed=$((elapsed + interval))

    if [ "$elapsed" -ge "$timeout" ]; then
        echo "Timed out waiting for server after $timeout seconds."
        exit 1
    fi
done

python3 -m sglang.bench_serving --backend sglang --dataset-name sharegpt --num-prompts $2 --request-rate $3 --port $ipnport

HOST_NICKNAME=$(python -c "print('$(hostname)'.split('.')[0])")
cp emissions.csv /results/emissions.csv
cp *.jsonl /results/
mkdir -p /data/input/claran/results/batch/${MODEL////__}/$2_$3_$4_${HOST_NICKNAME}
cp /results/* /data/input/claran/results/batch/${MODEL////__}/$2_$3_$4_${HOST_NICKNAME}
