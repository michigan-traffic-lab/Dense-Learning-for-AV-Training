# Installation
> The overall installation process should take 1~2 minutes on a recommended computer.

## 1. Create a virtual environment and activate it
To ensure high flexibility, it is recommended to use a virtual environment when running the code. To set up the virtual environment, please follow the commands provided below:
```bash
virtualenv venv_dlav
source venv_dlav/bin/activate
# if you prefer Anaconda3, please use the following commands instead
# conda create -n venv_dlav python=3.9.12
# conda activate venv_dlav
```

## 2. Install Torch
Please install the torch package using the following command:
```bash
pip install torch==1.12.1+cpu torchvision==0.13.1+cpu torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cpu
```

## 3. Install all required packages
To install the packages required for this repository, execute the command provided below:
```bash
pip install -r requirements.txt
```

## 4. Update RLlib
The specific RLlib has [one known bug related to the model restoring](https://github.com/ray-project/ray/issues/26557). In order to fix this issue, please do the following modification to the function named "set_state" in the file "torch_policy.py" within the folder "venv_dlav/lib/python3.9/site-packages/ray/rllib/policy" or "~/anaconda3/envs/venv_dlav/lib/python3.9/site-packages/ray/rllib/policy":
```
    @override(Policy)
    @DeveloperAPI
    def set_state(self, state: dict) -> None:
        # Set optimizer vars first.
        optimizer_vars = state.get("_optimizer_variables", None)
        if optimizer_vars:
            assert len(optimizer_vars) == len(self._optimizers)
            for o, s in zip(self._optimizers, optimizer_vars):
                
                # Fix RLlib bugs
                for v in s["param_groups"]:
                    if "foreach" in v.keys():
                        v["foreach"] = False if v["foreach"] is None else v["foreach"]
                    if "fused" in v.keys():
                        v["fused"] = False if v["fused"] is None else v["fused"]
                for v in s["state"].values():
                    if "momentum_buffer" in v.keys():
                        v["momentum_buffer"] = False if v["momentum_buffer"] is None else v["momentum_buffer"]

                optim_state_dict = convert_to_torch_tensor(
                    s, device=self.device)
                o.load_state_dict(optim_state_dict)
        # Set exploration's state.
        if hasattr(self, "exploration") and "_exploration_state" in state:
            self.exploration.set_state(state=state["_exploration_state"])
        # Then the Policy's (NN) weights.
        super().set_state(state)
```

-> Next Page: [Data Collection/Evaluation](data_collection_evaluation.md)