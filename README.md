***TFDWT: Fast Discrete Wavelet Transform TensorFlow Layers***

https://github.com/kkt-ee/TFDWT 

<!-- ***Batched 1D, 2D and 3D fast discrete wavelet transform (DWT) and inverse DWT*** -->

---


# Developer 

        Python 3.9.18
        TensorFlow 2.16.0
        Keras 2.14.0
        Numpy 1.26.0
        CUDA 12.3

**Tree**

        .
        |-- LICENSE
        |-- README.md
        |-- dbFBimpulseResponses.json
        |-- pyproject.toml
        |-- requirements.txt
        |-- resetpypi
        |-- src
        |   `-- TFDWT3D
        |       |-- DWTIDWT1Dtfv1.py
        |       |-- DWTIDWT2Dtfv1.py
        |       |-- DWTIDWT3Dtfv1.py
        |       |-- GETDWTFiltersOrtho.py
        |       |-- __init__.py
        |       |-- dbFBimpulseResp.json
        |       |-- dbFBimpulseResponse.py
        |       |-- get_A_matrix_dwt_analysisFB_unit.py
        |       `-- setup.py
        |-- tests
        `-- updaterepo




**To build a pypi package**

> python3 -m pip install --upgrade build

> python3 -m build


**To upload a pypi package online** (Registration required)

> python3 -m pip install --upgrade twine

> python3 -m twine upload --repository TFDWT3D dist/*

Enter username and password...



### Github push troubleshoot ssh

> ls -artl ~/.ssh  

> eval "$(ssh-agent -s)" 

> ssh-add ~/.ssh/id_ed25519githubnew



# User manual
    
        Works with square, cube dyadic scale batched data eg. 512x512x512
        Upgrade reqd. for rectangular and cubiod shaped data... 
    




**Install locally**

> pip install .

Check installation e.g.

> ls /opt/anaconda3/envs/tf2/lib/python3.9/site-packages/TFDWT3D 

**Uninstall**

> pip uninstall TFDWT3D





**Import forward and inverse transforms**


> from TFDWT3D.DWTIDWT1Dtfv1 import DWT1D, IDWT1D

> from TFDWT3D.DWTIDWT2Dtfv1 import DWT2D, IDWT2D

> from TFDWT3D.DWTIDWT3Dtfv1 import DWT3D, IDWT3D




END

---

***TFDWT3D (c) 2024 Kishore Kumar Tarafdar, Research Fellow, EE, IITB, India/ भारत***

