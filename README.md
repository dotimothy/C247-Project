# C247-Project
Final Project Repository of C147/C247 for Winter Quarter 2024. Task: Classification of BCI EEG Actions on Different Subjects.

Authors: [Timothy Do](https://timothydo.me), Brandon Kam, Josh McDermott, Steve Zang

## Setup
1. Run <code>pip install -r requirements.txt</code> to check and install all Python dependencies.
2. Run <code>jupyter notebook</code> and open the <code>Project.ipynb</code> to perform the task progressively! This main notebook covers data exploration, architecture definitions and sample training procedures. It also investigates the experinment of individual subject optimization and model ensembling.
3. To evaluate our best pretrained models for each architecture stored in <code>models</code>, in Jupyter open <code>BestModelTesting.ipynb</code> and follow each cell output.
4. To investigate classification as a function of time, in Jupyter open <code>TestAccuracyVSTimeStep.ipynb</code> and follow each cell output. Note that this notebook takes a long time as it performs extensive grid search through different learning rates, batch sizes, and time steps for each model (more than 3 hours per model).


