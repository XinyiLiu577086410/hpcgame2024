#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=C064M0256G
#SBATCH --qos=normal
#SBATCH -J rayHead
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8

module load python/hpcgame

export ADDR=$(ip -f inet addr show ens1np0 | sed -En -e 's/.*inet ([0-9.]+).*/\1/p')
export PORT=$(comm -23 <(seq 10000 65535) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | grep "[0-9]\{1,5\}" | sort | uniq) | shuf | head -n 1)
export TMP_DIR=/tmp/$(tr -dc A-Za-z0-9 </dev/urandom | head -c 16; echo)

INCREMENT=1

PORT=$PORT
isfree=$(netstat -taln | grep $PORT)

while [[ -n "$isfree" ]]; do
    PORT=$[port+INCREMENT]
    isfree=$(netstat -taln | grep $PORT)
done

echo "Usable Port: $PORT"

MIN_WORKER_PORT=$[PORT+1]
MAX_WORKER_PORT=$[PORT+500]

# if MAX_WORKER_PORT > 65535, then MAX_WORKER_PORT = 65535
if [ $MAX_WORKER_PORT -gt 65535 ]; then
    MAX_WORKER_PORT=65535
    MIN_WORKER_PORT=$[MAX_WORKER_PORT-500]
fi

echo "$ADDR:$PORT" > ray-addr.txt
ray start --node-ip-address $ADDR --head --port $PORT --num-cpus 4 --include-dashboard=false --temp-dir $TMP_DIR --min-worker-port $MIN_WORKER_PORT --max-worker-port $MAX_WORKER_PORT --block 