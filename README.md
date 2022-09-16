# IAMAX

This repository contains the supplemental materials released together with the paper 
"[Using Constraint Programming and Graph Representation Learning for Generating Interpretable Cloud Security Policies](https://www.ijcai.org/proceedings/2022/0257.pdf)" 
published at IJCAI-ECAI'22 in Vienna, Austria.

### Abstract
Modern software systems rely on mining insights from business sensitive data stored in public clouds.
A data breach usually incurs significant (monetary) loss for a commercial organization.
Conceptually, cloud security heavily relies on Identity Access Management (IAM) policies that IT admins need to properly configure and periodically update.
Security negligence and human errors often lead to misconfiguring IAM policies which may open a backdoor for attackers. 
To address these challenges, _first_, we develop a novel framework that encodes generating _optimal_ IAM policies 
using constraint programming (CP). We identify reducing _dark permissions_ of cloud users as an optimality criterion, 
which intuitively implies minimizing unnecessary datastore access permissions. _Second_, to make IAM policies interpretable, 
we use graph representation learning applied to historical access patterns of users to augment our CP model with 
_similarity_ constraints: similar users should be grouped together and share common IAM policies.
_Third_, we describe multiple attack models, and show that our optimized IAM policies significantly reduce the impact 
of security attacks using real data from 8 commercial organizations, and synthetic instances.


## Prerequisites

#### Required Ubuntu packages
```
sudo apt-get install build-essential libssl-dev libboost-all-dev libgtk-3-dev libpoppler-cpp-dev cmake poppler-utils
```

#### Install [IBM ILOG CPLEX Optimization Studio](https://www.ibm.com/products/ilog-cplex-optimization-studio) and its prerequisites

```
install Python 3, e.g. sudo apt install python3.8
install JDK, e.g. sudo apt install default-jdk
chmod u+x <installname>.bin
# install IBM ILOG CPLEX Optimization Studio to "/home/ubuntu/ibm/ILOG/CPLEX_Studio201"
./<installname>.bin
```
You may have to update the configuration variable **CPO_OPTIMIZER_PATH** in config.py if the CPO Optimizer's location 
is different from the one in the file:
```
/home/ubuntu/ibm/ILOG/CPLEX_Studio201/cpoptimizer/bin/x86-64_linux/cpoptimizer
```

#### Run
```
pip3 install -r requirements.txt
```

#### Prepare synthetic data
```
cd data
unzip synthetic_graphs.zip
```

## Repository Structure

```
├── data
│   ├── synthetic_graphs                <- Input synthetic graphs
│   ├── synthetic_results_all_groups    <- Output directory
│   ├── figs                            <- Output directory for figures
├── src                                     <- Source code
│   ├── run_iam_experiments.py                  <- Runs IAM optimization across all synthetic datasets
│   ├── iam_optimization_cp.py                  <- Configures Docplex CPOmodel object
│   ├── callbacks.py                            <- Callback that monitors reduction of dark permissions
│   ├── data_loader.py                          <- Parses and loads data
│   ├── visualize.py                            <- Runs visualization code and outputs figures used in the paper
│   ├── generate_synthetic_data.py              <- Generator of synthetic graphs 
├── requirements.txt                        <- required Python libraries
```


## Running the code

### Run IAMAX

```
$ python src/run_iam_experiments.py
```

### Run visualization code

```
$ python src/visualize.py
```

### Bibtex
```
@inproceedings{ijcai2022p257,
  title     = {Using Constraint Programming and Graph Representation Learning for Generating Interpretable Cloud Security Policies},
  author    = {Kazdagli, Mikhail and Tiwari, Mohit and Kumar, Akshat},
  booktitle = {Proceedings of the Thirty-First International Joint Conference on
               Artificial Intelligence, {IJCAI-22}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Lud De Raedt},
  pages     = {1850--1858},
  year      = {2022},
  month     = {7},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2022/257},
  url       = {https://doi.org/10.24963/ijcai.2022/257},
}
```
