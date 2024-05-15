import numpy as np
import os
from ray.util.placement_group import (
    placement_group,
    placement_group_table,
    remove_placement_group,
)
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import ray

ray.init(address=f"{os.environ['RAY_CLUSTER_ADDR']}")
# ray.init(num_cpus=16)


# Create directories if they don't exist
os.makedirs('outputs', exist_ok=True)

pg=placement_group([{"CPU":4}] * 4, strategy="PACK")

try:
    ray.get(pg.ready(), timeout=10)
except TimeoutError as e:
    print("Timed out waiting for placement group")
else:
    print("Placement group ready")

@ray.remote(num_cpus=4)
class Actor:
    def __init__(self, weight_file):
        self.weight = np.load(weight_file)
    def forward(self, input):
        return np.maximum(0.0, np.matmul(input, self.weight))
    

# Load the weights
actors = [Actor.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg,)).remote(f'weights/weight_{i}.npy',) for i in range(4)]

    # Run the forward pass
handles = []
for i in range(100):
    input = ray.put(np.load(f'inputs/input_{i}.npy'))
    handles.append(input)
    for actor in actors:
        handles[i] = actor.forward.remote(handles[i])


for i in range(100):
    # Save the output
    np.save(f'outputs/output_{i}.npy', ray.get(handles[i]))