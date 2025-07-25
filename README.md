# Transforming Cognitive Modeling Workflow with BayesFlow and HSSM

Welcome to the tutorial repo on supercharging cognitive modeling workflow with simulation-based inference (SBI).
For the simulation part, we will use the [Sequential Sampling Model Simulators (SSMS)](https://lnccbrown.github.io/ssm-simulators/api/ssms/) Package.

We will illustrate posterior amortization, likelihood amortization and focus on aspects of scientific workflows that can benefit from each.
You will learn about the powerful [BayesFlow](\https://bayesflow.org) and [HSSM](https://github.com/lnccbrown/HSSM) toolboxes and how to leverage both respectively to supercharge you cognitive modeling.

Check the official [course website](https://sites.google.com/view/sbi-workshop-mathpsych2025/home) for the schedule and more general information about the workshop.

## Colab setup

All notebooks should be runnable directly via **colab** and this is the preferred mode of interaction with the tutorial material.

1. [HSSM/PyMC tutorial link](https://githubtocolab.com/bayesflow-org/hssm-bayesflow-workshop/blob/main/notebooks/rv_to_hssm.ipynb)

## Local setup

If you rather have a local setup, the instructions below should help you get started.

### 1. Get the Workshop Materials

You can either **clone the repository** (if you're familiar with Git) or **download it as a ZIP**.

#### 🔧 Option 1: Clone with Git (recommended)

```bash
git clone https://github.com/your-username/hssm-bayesflow-workshop.git
cd hssm-bayesflow-workshop
```

#### 📦 Option 2: Download ZIP
1. Click the green "Code" button
2. Select "Download ZIP"
3. Extract the ZIP file and navigate into the folder

### 2. Install the necessary packages

#### Option A: Using uv (recommended)

This repository includes a `pyproject.toml` file for easy dependency management with `uv`. If you have `uv` installed, you can set up the environment with:

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate a virtual environment with all dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -e .
```

#### Option B: Using pip

Make sure to install BayesFlow and SSMS for the Python version you use for this workshop:

```bash
pip install bayesflow ssm-simulators
```

You can also install conda and install the packages from an environment you create for the workshop materials.

### 3. Let's GO!

