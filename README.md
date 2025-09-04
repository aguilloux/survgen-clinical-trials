# SurvGen-Clinical-Trials

Code accompanying the paper *"Toward Valid Generative Clinical Trial Data with Survival Endpoints"*.  
This repository provides tools to **generate and analyze synthetic clinical trial data with survival endpoints** using a generative modeling approach.

It includes our extension of the **Heterogeneous Variational Autoencoder (HI-VAE)** supporting survival data, enabling the model to handle **time-to-event analysis** with incomplete and heterogeneous data types.  
The code is written in **Python** and uses **PyTorch**.

---

<!-- ## Table of Contents
1. [Database Description](#database-description)
2. [Files Description](#files-description)
3. [Code Prerequisites](#code-prerequisites)
4. [Citation](#citation)
5. [License](#license)
<!-- --- -->

## Database Description

Our experiments use a mix of **real clinical trial datasets** and **simulated datasets**.  

### Real Clinical Trial Datasets

We used **phase III clinical trial datasets**:  

* **ACTG 320** – publicly available via the [Stanford University HIV Drug Resistance Database](https://hivdb.stanford.edu/pages/clinicalStudyData/ACTG320.html).  
* **NCT00119613, NCT00113763, NCT00339183** – accessible after registration via [Project Data Sphere](https://data.projectdatasphere.org).  

> **Important:** Real datasets are **not included** in this repository due to privacy restrictions. Users need to download them separately. Once downloaded, the **preprocessing scripts provided** can generate the required files (`data.csv`, `data_types.csv`, `Missingxx_y.csv`) for each real dataset.

---

### Simulated Datasets

Simulated datasets are provided in the repository for demonstration and reproducibility purposes.  

Each dataset (real or simulated) should have its own folder and contain the following files:  
* **data.csv** – the dataset itself  
* **data_types.csv** – describes the types of that particular dataset. Each line is a different attribute and contains:  
    * `name` – variable name  
    * `type` – data type: `real`, `pos` (positive), `cat` (categorical), `ord` (ordinal), `count`, `surv` (log-normal), or `surv_weibull` (Weibull)  
    * `dim` – dimension of the variable in the original dataset  
    * `nclass` – number of categories (for categorical and ordinal variables)  
* **Missingxx_y.csv** – (optional) contains positions of missing values. Each mask `"y"` corresponds to `"xx"%` missing values. Leave blank if not needed.  

> You can add your own datasets as long as they follow this folder structure.  

---

## Files Description

The repository is organized as follows:

* **execute/** – Scripts to **train, generate data, and optimize hyperparameters** for each model: `'surv-GAN'`, `'surv-VAE'`, `'HI-VAE_piecewise'`, `'HI-VAE_weibull'`.  
* **preprocessing/*.ipynb** – Notebooks for **preprocessing real datasets**. Can generate required `data.csv`, `data_types.csv`, and `Missingxx_y.csv` files dor real datasets.  
* **script/** – Scripts defining experiments (hyperparameter optimization, Monte Carlo experiments, type I error and power estimation).  
* **batch/** – Batch files to **run simulations and experiments** defined in the `script/` folder. Useful for automated or parallel execution on servers.
* **tutorial/*.ipynb** – Tutorial notebooks demonstrating **how to run models and evaluate results**.  
* **utils/** – Helper scripts, including:  
	- `src.py` – Implementation of the **HI-VAE models** (factorized encoder or input dropout encoder).  
	- Additional utilities for **loading data, computing likelihoods, computing errors**, and other supporting functions.  
* **visualization_notebook/*.ipynb** – Notebooks to **visualize experiment results** and **reproduce figures from the paper** (e.g., `notebook_figures_paper.ipynb`).

---

## Code Prerequisites

Create a Conda environment compatible with [synthcity](https://github.com/vanderschaarlab/synthcity):

```bash
cd HI-VAE
conda create --name hivae python=3.12.9
conda activate hivae
pip install synthcity
pip install -r pip_requirements.txt
```

---

## License 

---




