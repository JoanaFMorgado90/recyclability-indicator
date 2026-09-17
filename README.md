# Recyclability indicator

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

## Acknowledgements

- The authors were supported by Swiss State Secretariat for Education, Research and Innovation (SERI) under contract number 22.00489 in the frame of Horizon Europe [“CE-RISE: Circular Economy Resource Information System”](https://ce-rise.eu) co-funded by the [European Union project 101092281](https://cordis.europa.eu/project/id/101092281/reporting). 

- The authors acknowledge the use of Claude (Anthropic) via Abacus.ai for providing code-generation support during the development of the uncertainty and sensitivity analysis framework. The final code execution, data interpretation and scientific conclusions remain solely the responsibility of the authors.


