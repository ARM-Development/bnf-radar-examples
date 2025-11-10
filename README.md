# BNF Radar Examples
This is a place to collaborate on data related to the Atmospheric Radiation Measurement user facility Advanced Mobile Facility (3) deployment to the Bankhead National Forest (BNF) site in the Southeastern United States.

## Motivation

This repository is for collaborating on workflows and visualizations related to the ARM AMF deployment to the BNF site. We use a variety of datasets (mostly radar) to investigate different events, and provide open workflows for others to use.

## Authors

[Joe O'Brien](@jrobrien91), [Bobby Jackson](@rcjackson), [Bhupendra Raut](@RBhupi), [Zach Sherman](@zssherman), [Max Grover](@mgrover1)

### Contributors

<a href="https://github.com/ARM-Development/bnf-radar-examples/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ARM-Development/bnf-radar-examples" />
</a>

## Structure

### Site Exploration
Initial data exploration, looking at sample cases and associated datasets such as the University of Alabama Huntsville operated ARMOR radar.

### Extracted Radar Columns and In-SItu Sensors
Exploration of the RadCLss Value Added Product design, implentation and initial case studies. 

### Surface Quantitative Precipitation Estimates
Exploration of the SQUIRE Value Added Product design, implentation, and initial case studies. 

## Running the Notebooks
You can either run the notebook using [ARM Jupyter Hub](https://jupyterhub.arm.gov/) or on your local machine.

### Running on Your Own Machine
If you are interested in running this material locally on your computer, you will need to follow this workflow:

1. Clone the `https://github.com/ARM-Development/bnf-radar-examples` repository:

   ```bash
    git clone https://github.com/ARM-Development/bnf-radar-examples.git
    ```  
1. Move into the `bnf-radar-examples` directory
    ```bash
    cd bnf-radar-examples
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
