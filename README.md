
Setup environment:

```sh
conda env create -f environment.yaml
conda activate hwo_subneptune
```

Install PICASO and get reference data:

```sh
wget https://github.com/Nicholaswogan/picaso/archive/0f2a21e574f2583786b31ebe09f95b0d372d92d4.zip
unzip 0f2a21e574f2583786b31ebe09f95b0d372d92d4.zip
cd picaso-0f2a21e574f2583786b31ebe09f95b0d372d92d4
python -m pip install . -v --no-deps --no-build-isolation
cd ../
cp -r picaso-0f2a21e574f2583786b31ebe09f95b0d372d92d4/reference picasofiles/
rm -rf picaso-0f2a21e574f2583786b31ebe09f95b0d372d92d4
rm 0f2a21e574f2583786b31ebe09f95b0d372d92d4.zip
```

Get input files:

```sh
python input_files.py
```


```sh
mkdir -p codex_reference && cd codex_reference
for u in \
  https://github.com/natashabatalha/picaso/archive/0f2a21e574f2583786b31ebe09f95b0d372d92d4.zip \
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