# ideas-toolbox-caiman


**Table of Contents**
- [Toolbox Description](#toolbox-description)
- [How to Get Help](#how-to-get-help)
- [Navigating the Project Repository](#navigating-the-project-repository)


## Toolbox Description
A toolbox for running [CaImAn-based](https://github.com/flatironinstitute/CaImAn) tools on the IDEAS platform.

This toolbox is designed to run as a Docker image, which can be run on the IDEAS platform. This toolbox consists of the following tools:

- `CaImAn Cell Extraction Workflow`: The CaImAn cell identification workflow performs motion correction using the NoRMCorre algorithm, cell identification using the CNMF/CNMF-E algorithm, and automated component evaluation.
- `CaImAn NoRMCorre`: NoRMCorre (Non-Rigid Motion Correction) is a motion correction algorithm that can be used to perform rigid or piecewise rigid motion correction based on template matching.
- `CaImAn Source Extraction`: Extract spatial footprints and temporal activity of cells using the CaImAn CNMF/CNMF-E algorithm.
- `CaImAn Spike Extraction`: Infer and extract the neural activity underlying fluorescence traces using a constrained deconvolution approach.
- `CaImAn Multi-Session Registration`: Register cells across sessions spanning days, weeks or even months.

## How to Get Help
- [IDEAS documentation](https://inscopix.github.io/ideas-docs/tools/caiman/caiman_isx_academic__caiman_workflow/caiman_isx_academic__caiman_workflow.html) contains detailed information on how to use the toolbox within the IDEAS platform, the parameters that can be used, and the expected output.
- If you have found a bug, we recommend searching the [issues page](https://github.com/inscopix/ideas-toolbox-caiman/issues) to see if it has already been reported. If not, please open a new issue.
- If you have a feature request, please open a new issue with the label `enhancement`.

## Executing the Toolbox

One dependency is required for building the toolbox and running tools locally.
- Download the [TensorFlow 2.8.0 built distribution for Linux](https://tf.novaal.de/barcelona/tensorflow-2.8.0-cp310-cp310-linux_x86_64.whl) into the `ideas-toolbox-caiman/resources/` directory

To run the toolbox, you can use the following command:

`make run TOOL=<tool_name>`

Available tools are:

- `caiman_isx_academic__caiman_workflow`
- `caiman_isx_academic__motion_correction`
- `caiman_isx_academic__source_extraction`
- `caiman_isx_academic__spike_extraction`
- `caiman_msr__caiman_msr`

The command will excute the tool with inputs specified in the `inputs` folder. The output will be saved in the `outputs` folder.

## Navigating the Project Repository

```
├── commands                # Standardized scripts to execute tools on the cloud
├── data                    # Small data files used for testing
├── info                    # Information about the toolbox and its tools
├── inputs                  # Example input files for testing the tools
│── toolbox                 # Contains all code for running and testing the tools
│   ├── tools               # Contains the individual analysis tools
│   ├── utils               # General utilities used by the tools
│   ├── tests               # Unit tests for the individual tools
│── Makefile                 # To automated and standardize toolbox usage
│── Dockerfile               # Commands to assemble the Docker imageons
|── LICENSE                 # License file
|── user_deps.txt           # Python dependencies for the toolbox
└── .gitignore              # Tells Git which files & folders to ignore
```

## License

This project is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.en). Note that the software is provided "as is", without warranty of any kind.
