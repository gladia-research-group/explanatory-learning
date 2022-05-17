# Explanatory Learning
This is the official repository for *"Explanatory Learning: Beyond Empiricism in Neural Networks"* [[arXiv](https://arxiv.org/abs/2201.10222)][[PDF](https://arxiv.org/pdf/2201.10222.pdf)].

## Datasets
Download the datasets with the following links:

 * [Training Dataset - 1438 rules](https://drive.google.com/file/d/16Z0mlO_ynFBZVG0Gpx3oCqr-MP8IJu_f/view?usp=sharing)
 * [Training Dataset - 500 rules](https://drive.google.com/file/d/1WD36fUBw1QbxLdEThxeaV6SyAwl1lm0I/view?usp=sharing)
 * [Validation Dataset](https://drive.google.com/file/d/1IyUKQrpD6tQRcp_4NgCevWqsoypObU7Y/view?usp=sharing)
 * [Test Dataset](https://drive.google.com/file/d/1BpVp2gO9Uy5pkdMdSXsVY90Oqxp26b00/view?usp=sharing)

After downloading, move the datasets inside the directory `data`, located in the repository root folder.

## Usage
First, clone the github repository
```bash
git clone https://github.com/gladia-research-group/explanatory-learning.git
```
and move the current directory int the repository root:
```bash
cd explanatory-learning
```

### Dependencies
This repository has the following dependencies:
 * Python 3.8
 * PyTorch 1.7 (look [here](https://pytorch.org/get-started/locally/)) 

All other dependencies can be easily installed by running:
```bash
pip install -r requirements.txt
```

### Training
It is possible to train the models described in the paper with the following commands.
First, while inside the root directory, you can start the training procedure for the models described in the paper with the following command:
```bash
python explanatory_learning/learning/training.py --config-file "config.json"
```
This will create a folder in the repository root directory named `training_results`. This folder is necessary in order to run the experiments in the paper.

**NOTE:** It is possible to change the configuration file `config.json` (located in the repository root folder) to modify the training procedure.

### Experiments
In order to execute the experiments reported in the paper you can use the following commands.

* **Rule Induction Task:** For the results presented in Table 1, run:
	```bash
	python explanatory_learning/experiments/rule_induction.py --config-file "configs/config.json"
	```
 * **World Labeling Task:** For the results presented in Table 2, run:
	```bash
	python explanatory_learning/experiments/world_labeling.py --config-file "configs/config.json"
	```
 * **Rank Analysis:** For the results presented in Table 3, run:
	```bash
	python explanatory_learning/experiments/rank_analysis.py --config-file "configs/config.json"
	```
* **Interpreter Evaluation:** For the results presented in Figure 4, run:
	```bash
	python explanatory_learning/experiments/interpreter_analysis.py --config-file "configs/config.json"
	```

**NOTE:** Each command will generate a respective folder in the repository root directory containing the results in json format. In particular, the file 
```bash
python explanatory_learning/experiments/<X>.py --config-file "configs/config.json"
```
will generate the directory `<X>_results`.
