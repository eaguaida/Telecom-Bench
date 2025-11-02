# Installing Inspect AI

You need to install `inspect-ai` to run evaluations. Choose ONE of the following methods:

## Option 1: Install globally with pip
```bash
pip install inspect-ai
```

## Option 2: Install in your conda environment
```bash
conda activate your_environment_name
pip install inspect-ai
```

## Option 3: Set custom inspect command
If you have `inspect` installed in a specific location, set the environment variable:
```bash
export INSPECT_CMD=/path/to/your/inspect
python app.py
```

## Verify Installation
After installing, verify it works:
```bash
inspect --version
```

You should see something like: `Inspect AI v0.x.x`

## Then restart the Flask app
```bash
cd ui
python app.py
```

The app will automatically detect the `inspect` command when it starts.

