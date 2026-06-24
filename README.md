
Setup environment:

```sh
conda env create -f environment.yaml
conda activate hwo_subneptune
```

Install PICASO and get reference data:

```sh
wget https://github.com/Nicholaswogan/picaso/archive/77365d37772f25de6d9e0e2e045cdd393647c3f8.zip
unzip 77365d37772f25de6d9e0e2e045cdd393647c3f8.zip
cd picaso-77365d37772f25de6d9e0e2e045cdd393647c3f8
python -m pip install . -v --no-deps --no-build-isolation
cd ../
cp -r picaso-77365d37772f25de6d9e0e2e045cdd393647c3f8/reference picasofiles/
rm -rf picaso-77365d37772f25de6d9e0e2e045cdd393647c3f8
rm 77365d37772f25de6d9e0e2e045cdd393647c3f8.zip
```

Get input files:

```sh
python input_files.py
```


```sh
mkdir -p codex_reference && cd codex_reference
for u in \
  https://github.com/natashabatalha/picaso/archive/77365d37772f25de6d9e0e2e045cdd393647c3f8.zip \
  https://github.com/Nicholaswogan/photochem/archive/refs/tags/v0.8.4.zip \
  https://github.com/Nicholaswogan/clima/archive/refs/tags/v0.7.4.zip
do
  f="$(basename "$u")"
  wget -O "$f" "$u"
  unzip -q "$f"
  rm -f "$f"
done
cd ..
```

cmake .. -DCMAKE_INSTALL_PREFIX=/nobackup/nwogan/MiniNeptuneGrid26_PostBac_retrievals/lib -DCMAKE_Fortran_COMPILER=/nasa/hpe/mpt/2.30_rhel810/bin/mpif90 -DCMAKE_C_COMPILER=/nasa/hpe/mpt/2.30_rhel810/bin/mpicc -DCMAKE_CXX_COMPILER=/nasa/hpe/mpt/2.30_rhel810/bin/mpicxx -DCMAKE_PREFIX_PATH=$CONDA_PREFIX