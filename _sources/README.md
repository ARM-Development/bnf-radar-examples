# AMF3 Radar Examples
This is a place to collaborate on data related to the Atmospheric Radiation Measurement user facility Advanced Mobile Facility (3) deployment to the Southeastern United States.

## Motivation

This repository is for collaborating on workflows and visualizations related to the ARM AMF deployment in the Southeastern United States. We use a variety of datasets (mostly radar) to investigate different events, and provide open workflows for others to use.

## Authors

[Max Grover](@mgrover1)

### Contributors

<a href="https://github.com/ARM-Development/amf3-radar-examples/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ARM-Development/amf3-radar-examples" />
</a>

## Structure

### Site Exploration
Initial data exploration, looking at sample cases and associated datasets such as the University of Alabama Huntsville operated ARMOR radar.

## Running the Notebooks
You can either run the notebook using [Binder](https://mybinder.org/) or on your local machine.

### Running on Binder

The simplest way to interact with a Jupyter Notebook is through
[Binder](https://mybinder.org/), which enables the execution of a
[Jupyter Book](https://jupyterbook.org) in the cloud. The details of how this works are not
important for now. All you need to know is how to launch a Pythia
Cookbooks chapter via Binder. Simply navigate your mouse to
the top right corner of the book chapter you are viewing and click
on the rocket ship icon, (see figure below), and be sure to select
“launch Binder”. After a moment you should be presented with a
notebook that you can interact with. I.e. you’ll be able to execute
and even change the example programs. You’ll see that the code cells
have no output at first, until you execute them by pressing
{kbd}`Shift`\+{kbd}`Enter`. Complete details on how to interact with
a live Jupyter notebook are described in [Getting Started with
Jupyter](https://foundations.projectpythia.org/foundations/getting-started-jupyter.html).

### Running on Your Own Machine
If you are interested in running this material locally on your computer, you will need to follow this workflow:

1. Clone the `https://github.com/ARM-Development/amf3-radar-examples` repository:

   ```bash
    git clone https://github.com/ARM-Development/amf3-radar-examples.git
    ```  
1. Move into the `cookbook-example` directory
    ```bash
    cd amf3-radar-examples
    ```  
1. Create and activate your conda environment from the `environment.yml` file
    ```bash
    conda env create -f environment.yml
    conda activate amf3-radar-examples-dev
    ```  
1.  Move into the `notebooks` directory and start up Jupyterlab
    ```bash
    cd notebooks/
    jupyter lab
    ```
