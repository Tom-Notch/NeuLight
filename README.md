# NeuLight

## Author

- [Mukai (Tom Notch) Yu](https://tomnotch.com): <mukaiy@andrew.cmu.edu>

## Environment Setup

- Create and activate a conda environment with python 3.10

```Shell
conda create -n neulight python=3.10
conda activate neulight
```

- Install dependencies

```Shell
pip install -e .
```

## Dataset

- [Free Viewpoint](https://repo-sam.inria.fr/fungraph/deep-indoor-relight/)

### Folder Structure

```Shell
❯ tree -dh data/neulight
data/neulight
└── [4.0K]  Salon2
    └── [4.0K]  scene_share
        ├── [4.0K]  cameras
        ├── [ 36K]  images
        ├── [4.0K]  lightings
        └── [4.0K]  meshes
```
