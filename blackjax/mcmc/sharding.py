import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils
from typing import Tuple

def create_thermodynamic_mesh() -> Tuple[Mesh, NamedSharding]:
    """
    Detects local physical hardware (GPUs, TPUs, or CPUs) and builds 
    a distribution mesh to shard temperature rungs across them.
    """
    # 1. Detect physical devices available to the JAX runtime
    devices = jax.local_devices()
    num_devices = len(devices)
    
    # 2. Create a 1D grid of devices
    # E.g., if you have 2 ASUS GPUs, this creates a 1x2 hardware grid.
    device_mesh = mesh_utils.create_device_mesh((num_devices,))
    
    # 3. Name the physical axis "device_axis" so we can reference it
    mesh = Mesh(device_mesh, axis_names=('device_axis',))
    
    # 4. Define the Sharding Specification:
    # P('device_axis', None) means: "Cut the FIRST dimension (the rungs) into chunks 
    # and distribute them across the 'device_axis'. Keep the SECOND dimension intact."
    sharding_spec = NamedSharding(mesh, P('device_axis'))
    
    return mesh, sharding_spec


def shard_pytree(pytree, sharding_spec: NamedSharding):
    """
    Physically transfers the arrays inside any Pytree (like your ParallelTemperingState) 
    onto the distributed hardware lanes.
    """
    # jax.device_put moves the memory from the host (CPU) to the target devices (GPUs)
    # according to the sharding rule.
    return jax.tree_util.tree_map(
        lambda leaf: jax.device_put(leaf, sharding_spec) if isinstance(leaf, jax.Array) else leaf,
        pytree
    )