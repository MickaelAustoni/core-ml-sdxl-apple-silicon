# Training ML DL SDXL

### Create env and activate
```bash
  python -m venv env_DL_SDXL_training
  source env_DL_SDXL_training/bin/activate
```

### Install dependencies
```bash
  pip install -r requirements_sdxl.txt
```

### And initialize accelerate environment with:
```bash
  accelerate config
```
Or for a default accelerate configuration without answering questions about your environment

```bash
  accelerate config default
```

### Run training
```bash
  chmod +x scripts/shell/train_dreambooth_lora_sdxl.sh
  ./scripts/shell/train_dreambooth_lora_sdxl.sh
```
