#!/usr/bin/bash

# start head node

rm -rf ray-addr.txt

export HEAD_JID=$(sbatch start-head.sbatch | awk '{print $4}')

# wait for head node to start, time out 300s
for i in {1..300}; do
    if [ -f ray-addr.txt ]; then
        break
    fi
    sleep 1
done

sleep 5 # wait for head node to start

if [ ! -f ray-addr.txt ]; then
    echo "Head node failed to start"
    exit 1
fi

echo "Head node started"

export SLAVE_JID=$(sbatch --dependency=after:$HEAD_JID start-slave.sbatch | awk '{print $4}')

RUNNING_STATE="State: RUNNING"

echo "Waiting for slave to start. Job not found. is expected"

# wait for slave node to start, time out 300s
for i in {1..300}; do
    export LD_PRELOAD=/usr/lib64/slurm/libslurmfull.so 
    SLAVE_STATE=$(seff $SLAVE_JID | grep State)
    if [[ $SLAVE_STATE == $RUNNING_STATE ]]; then
        break
    fi
    sleep 1
done

if [[ $SLAVE_STATE != $RUNNING_STATE ]]; then
    echo "Slave node failed to start"
    exit 1
fi

sleep 5 # wait for ray to start

echo "Slave node started"

echo "Waiting for job to start. Job not found. is expected"

# get node of head node

RAY_HEAD_NODE=$(sacct -j $HEAD_JID --starttime 2014-07-01 --format=nodelist | grep "l08c" | head -n 1)
# run job on head node
PYTHON_JID=$(sbatch start-job.sbatch --dependency=after:$SLAVE_JID --nodelist=$RAY_HEAD_NODE | awk '{print $4}')

# wait for job to finish, time out 700s
for i in {1..700}; do
    JOB_STATE=$(sacct -j $PYTHON_JID --starttime 2014-07-01 --format=state | grep "COMPLETED")
    if [[ $JOB_STATE == "COMPLETED" ]]; then
        break
    fi
    sleep 1
done

if [[ $JOB_STATE != "COMPLETED" ]]; then
    echo "Job failed to finish"
    exit 1
fi

# stop ray
scancel $HEAD_JID
scancel $SLAVE_JID
scancel $PYHTON_JID