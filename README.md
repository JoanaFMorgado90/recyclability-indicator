# Recyclability indicator tester

Supplementary code for Framework for Recyclability Indicator Development: Bridging Product Design and Recycling Technology - Resources,Conservation & Recycling, 2026 .  
DOI: [add once published]

## Contents

| Path | Description |
|------|-------------|
| `notebooks/recyclability_indicator_tester.ipynb` | Notebook to test the recyclability indicator principles|
| `scripts/recyclability_ua_sa.py` | Uncertainty and sensitivity analysis for the recyclability indicator |

## How to run

### With Docker (recommended)

```bash
docker build -t recyclability-indicator .
docker run -p 8888:8888 recyclability-indicator
```

Open the URL printed in the terminal (starting with `http://127.0.0.1:8888/...`) 
and navigate to `notebooks/`.

### Without Docker

```bash
pip install -r requirements.txt
jupyter notebook notebooks/
```

## How to cite


