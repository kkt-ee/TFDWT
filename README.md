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

        ├── dist
        │   ├── tfdwt-0.2-py3-none-any.whl
        │   └── tfdwt-0.2.tar.gz
        ├── LICENSE
        ├── pyproject.toml
        ├── README.md
        ├── requirements.txt
        ├── resetpypi
        ├── src
        │   └── TFDWT
        │       ├── dbFBimpulseResponse.py
        │       ├── DWTIDWT1Dv1.py
        │       ├── DWTIDWT2Dv1.py
        │       ├── get_A_matrix_dwt_analysisFB_unit.py
        │       ├── GETDWTFiltersOrtho.py
        │       └── __init__.py
        ├── tests
        ├── Tutorials
        │   ├── brain.png
        │   └── DWT_Tutorial.ipynb
        └── updaterepo





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


```from TFDWT3D.DWTIDWT1Dv1 import DWT1D, IDWT1D```

```from TFDWT3D.DWTIDWT2Dv1 import DWT2D, IDWT2D```





END

---

***TFDWT (C) 2025 Kishore Kumar Tarafdar, Prime Minister's Research Fellow, EE, IITB, India/ भारत***

