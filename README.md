# LOB Snapshot Backtesting  
Use this package to do backtesting using reinforcement learning on a Limit Order Book with 1 second snapshots. Must have data already pre processed

## How to use  
1. Instal using pip: `pip install git+https://github.com/GrantCanty/LOB-snapshot-backtesting.git`  
1. Import package: `from lob_env.environment import LOBEnv`  
1. Create custom environment: `env = environment.LOBEnv(example_LOB, device, 3, False)`  
1. Takes the following arguments: `data: torch tensor`, `device`, `order book depth: int`, `optional: Normalization=True`, `optional: fee percentage=0.0005`, `optional: reward cap step=25`

## Example
```
from lob_env.environment import LOBEnv
import torch

example_LOB = torch.tensor([
    [1772202600, 55.8, 2, 55.76, 6, 55.7, 4, 55.82, 8, 55.88, 2, 55.89, 4, 0.8, 55.78, 0],
    [1772202601, 55.78, 10, 55.73, 12, 55.71, 8, 55.81, 8, 55.84, 12, 55.87, 8, 0.65, 55.83, 1],
    [1772202602, 55.78, 10, 55.74, 3, 55.73, 12, 55.81, 8, 55.84, 12, 55.89, 4, 0.6, 55.82, 1],
    [1772202603, 55.76, 4, 55.75, 2, 55.74, 8, 55.81, 8, 55.82, 2, 55.84, 11, 0.82, 55.77, 0],
    [1772202604, 55.81, 10, 55.79, 23, 55.78, 15, 55.83, 8, 55.85, 6, 55.93, 8, 0.44, 55.79, 0],
])

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
env = environment.LOBEnv(example_LOB, device, 3, False)
obs, info = env.reset()
```
