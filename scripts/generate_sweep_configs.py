"""Generate config files for the noise sweep experiment."""
import yaml
from pathlib import Path

base_config = yaml.safe_load(open('config/simple.yaml'))
noise_levels = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]

sweep_dir = Path('config/sweep')
sweep_dir.mkdir(exist_ok=True)

for noise in noise_levels:
    cfg = yaml.safe_load(yaml.dump(base_config))  # deep copy
    cfg['training']['stimulus_noise'] = noise
    name = f'noise_{noise:.2f}'.replace('.', 'p')
    with open(sweep_dir / f'{name}.yaml', 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f'Generated {name}.yaml with stimulus_noise={noise}')
