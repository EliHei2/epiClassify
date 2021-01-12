


scPotter: Structural Inductive Bias in Supervised Learning for Single-cell Data
=========================================================================
.. raw:: html

scPotter takes feature interactions (e.g., gene-gene interactions) into account to classify single-cells and interpret the marker genes. 

scPotter is an end-to-end pipeline for 1) annotation 2) finding the most important features for each annotation category. The backbone is adopted from two of the packages, previously developed by us, GNNProject [GitHub.com/e-sollier/DL2020](https://github.com/e-sollier/DL2020) in python and scGCNUtiles [GitHub.com/EliHei2/scGCN](https://github.com/EliHei2/scGCN) in R. Many models from synthetic data generation to multiple neural network classifiers are implemented along with tools for visualization and exploratory data analysis. The modular structure of the pipeline enables extensions to new models and analyses. The pipeline is designed to meet state of the art reproducibility standards and criteria of open science, such as coherence, integrity, documentation, and readability.

The models are implemented with `pytorch <https://pytorch.org/docs/stable/index.html>`_ and `pytorch-geometric <https://pytorch-geometric.readthedocs.io/en/latest/>`_ in python. Some tools for preprocessing, visualization and exploratory data analysis are implemented in R.

**NOTE:** Here, we talk about graph classification, where each observation to be classified is represented by a graph, with nodes representing features. The graph structure for all observations is the same, the only difference comes from node values.


Modules
-------------------------------

Synthetic data generation
**********************
We generate graph structured, labeled, synthetic data inspired by gene regulatory networks. It consists of two steps:

1. Random Graph Generation:

- **Erdős–Rényi model:** To generate plane random networks based on `Erdős & Rényi, 1959 <https://en.wikipedia.org/wiki/Barabási–Albert_model>`_)

- **Barabási–Albert model:** To generate "scale-free" networks based on `Albert & Barabási, 2002 <https://en.wikipedia.org/wiki/Barabási–Albert_model>`_).

- **Stochastic Block model:** To generate "modular" networks based on `Stochastic Block Models <https://en.wikipedia.org/wiki/Stochastic_block_model>`_)

2. Message Passing on Graph (Data Generation):
This happens once the graph is generated and the nodes are initialized by a Gaussian white noise. For each class a subset of nodes (features) are chosen randomly as 'characteristics'. After message passing, another Gaussian white noise is added to all nodes.  

- **Information diffusion:** A Gaussian signal with a non-zero mean is added initially to the characteristic nodes. Each edge passes information based on the (global) edge capacity (diffusion coefficient) and the difference between its end nodes. Such a diffusion can be performed multiple times.

- **Activation:** Characteristic nodes are 'activated' with a Gaussian signal weighted by average of their neighbors. 

- **Sign:** Sign of characteristic nodes is set based on the sign of average of their neighbors. 

PBMC data
**********************
The dataset contains two separate experiments, hereafter called PBMC 1 and PBMC 2, generated in a single center, but employing seven different scRNA-seq methods. Both datasets are annotated by cell types which can be used to train and validate our classifiers. This suits our goal to verify our framework on a simulated task of real transfer learning. As in potential applications we expect our methods to learn annotation priorly on the complete (training) dataset and transfer the learned model to annotate the unseen (test) one. The raw dataset is available on the Gene Expression Omnibus with accession number [GSE132044](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE132044). The preprocessed dataset is available [here](https://polybox.ethz.ch/index.php/s/8cm5WSz70Wq5vmQ).

Graph inference
**********************
We infer the underlying graph structure based on the sparse high-dimensional correlation matrices estimated using the graphical lasso (`Friedman et al., 2008 <https://en.wikipedia.org/wiki/Graphical_lasso>`_) or the Ledoit-Wolf shrinkage estimator (`Ledoit & Wolf, 2004 <https://en.wikipedia.org/wiki/Graphical_lasso>`_). The adjacency matrix for the observed dataset and correspondingly the graph structure is reconstructed based on estimation of the inverse of a sparse high-dimensional correlation matrix. 

Classifiers
**********************
We use GNNs and compare them with FCNNs as baselines. We define GNNs based on `message passing <https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html>`_: each node generates a message based on its features, and sends it to its neighbors. Then each node aggregates the messages of its neighbors and uses this aggregate to updates its features. The aggregation is done with a permutation-invariant function, like mean, sum or max. Typically, we use one round of message passing and we use 8 hidden features per node. Several types of graph convolutional layers are implemented: 

 - `GraphSAGE <https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.SAGEConv>`_ 
 - `Chebyshev convolutions <https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.ChebConv>`_
 - `MFConv <https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.MFConv>`_
 - `GraphConv <https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GraphConv>`_ 
 - `GIN <https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GINConv>`_ 
 - `GATConv (graph attention) <https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GATConv>`_
 - `TransformerConv <https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.TransformerConv>`_
 
 Also a 1-d Convolutional Neural Network is implemented for the applications where the structure is linear, e.g., genomic locations.


Usage and installation
-------------------------------
You can get the latest development version of our toolkit from `Github <https://github.com/e-sollier/DL2020/>`_ using the following steps:
First, clone the repository using ``git``::

    git clone https://github.com/EliHei2/scPotter

Then, ``cd`` to the scArches folder and run the install command::

    cd DL2020
    python setup.py install

If you have problems with the torch-geometric dependency, look at `this <https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html>`_. 

Dependencies on ETH's Leonhard Cluster
**********************
In order to solve dependencies on Leonhard one should take the following steps:
1. Import the following modules::

    module load gcc/6.3.0 python_gpu/3.7.4 hdf5/1.10.1

2. Install relevant packages for torch-geometric::
    
    pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.5.1+cu101.html
    pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.5.1+cu101.html
    pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.5.1+cu101.html
    pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.5.1+cu101.html
    pip install torch-geometric

Notebooks
-------------------------------
To see some examples of our pipeline's capability, look at the `notbooks` directory.

Final Report
-------------------------------
The extended version of the report for this project can be found `here <https://github.com/EliHei2/scPotter/tree/main/notebooks/report>`_.

Reproducing the report figures/tables
**********************
- Preprocessing: notebooks/GNN_input_prep_pbmc.rmd
- Traning classifiers and finding important featuers: notebooks/PBMC_captum.ipynb
- Post analysis and visualization: notebooks/final-report-GCN-2020-01-11-pbmc.rmd


Support and contribute
-------------------------------
If you have a question or new architecture or a model that could be integrated into our pipeline, you can
post an `issue <https://github.com/EliHei2/scPotter/issues/new>`__ or reach us by `email <mailto:eheidari@student.ethz.ch>`_.



