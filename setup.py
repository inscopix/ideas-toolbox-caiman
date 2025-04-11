import os
from setuptools import setup

token = os.environ["IDEAS_GITHUB_TOKEN"]

install_requires = []
if os.path.isfile("user_deps.txt"):
    with open("user_deps.txt") as f:
        install_requires = f.read().splitlines()

setup(
    name="ideas-toolbox-caiman",
    python_requires=">=3.9",
    version="1.0.0",
    packages=[],
    description="",
    url="https://github.com/inscopix/ideas-toolbox-caiman",
    install_requires=[
        f"ideas-python-utils@git+https://{token}@github.com/inscopix/ideas-python-utils.git@10.14.1",
        f"ideas-commons@git+https://{token}@github.com/inscopix/ideas-commons.git@1.25.2",
        f"ideas-schemas@git+https://{token}@github.com/inscopix/ideas-schemas.git@2.4.2",
        f"ideas-tools-profiler@git+https://{token}@github.com/inscopix/ideas-tools-profiler.git@0.2.0",
        "caiman@git+https://github.com/inscopix/CaImAn.git@v0.0.8",
        "isx==2.0.1",
        "requests==2.27.1",
        "urllib3==1.26.16",
        "configobj==5.0.8",
        "pytest==7.4.2",
        "tabulate==0.9.0",
        "hdmf==3.14.6",
        # ideas-python-utils dependencies
        "figrid==0.1.6",
        "tabulate==0.9.0",
        # tensorflow dependencies
        "protobuf==3.20.3",
        # CaImAn dependencies
        "opencv-python==4.10.0.84",
        "h5py==3.11.0",
        "ipython==8.26.0",
        "ipywidgets==8.1.3",
        "matplotlib==3.9.1",
        "pims==0.7",
        "scipy==1.14.0",
        "scikit-image==0.24.0",
        "scikit-learn==1.5.1",
        "zarr==2.18.2",
        "pynwb==2.8.1",
        "ipyparallel==8.8.0",
        "peakutils==1.3.5",
        "bokeh==3.4.2",
        "holoviews==1.19.1",
        # movie previews
        "imageio==2.35.0",
        "imageio-ffmpeg==0.4.7",
    ]
    + install_requires,
)
