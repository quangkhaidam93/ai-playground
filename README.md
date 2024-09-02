# AI Playground

## Virtual Environment

Package shipped with Python3 for creating virtual environment (inspired by popular package `virtualenv`)

### Commands

Create virtual environment

```bash
python -m venv .venv
```

Activate virtual environment

```bash
source .env/bin/activate
```

Deactivate current session of venv

```bash
deactivate
```

### Resolve missing module

There is an error when using installed packages in venv -> manually enter binary of venv python to resolve this error

1. Run `which python` in terminal with activated `venv`
2. Cmd + shift + P > Python: Select Interpreter
3. Choose: Enter interpreter path...
4. Copy path returned from `which python` command -> Enter

## Generate requirements.txt

### Commands

Install package `pipreqs`

```bash
pip install pipreqs
```

Generate `requirements.txt` in specified directory

```bash
// Current directory
pipreqs

// (Recommended) Ignore .venv
pipreqs --ignore .venv

// Custom directory
pipreqs {path_to_directory}
```
