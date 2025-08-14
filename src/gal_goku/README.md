# HETDEX-Comology


```
python -m pip install emukit
conda create -n gal_goku python=3.12 numpy scipy=1.15.1 matplotlib
python -m pip install scikit-learn

```

```
python -m pip install -e .
```


## Flowchart

```
flowchart TD
    A[HMF_emu] -->|ϕ(M) dM| B
    B[Xihh_emu] -->|ξ_hh(M_th1, M_th2)| C[Hankel]
    C --> D[P_hh(M_th1, M_th2)]
    D -->|Spline| D

    D --> E[I_d = ∫ dM1 dM2 S(P_hh(M_th1, M_th2))]
    E --> F[HOD]
    F --> G[P_gg]
    G -->|Inverse Hankel| H[ξ_gg(r)]
```