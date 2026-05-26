
Setup environment:

```sh
conda env create -f environment.yaml
conda activate hwo_subneptune
```

Install PICASO and get reference data:

```sh
wget https://github.com/natashabatalha/picaso/archive/c3ce0d85973bc3b7eb6af117666c041d9a4966ab.zip
unzip c3ce0d85973bc3b7eb6af117666c041d9a4966ab.zip
cd picaso-c3ce0d85973bc3b7eb6af117666c041d9a4966ab
python -m pip install . -v --no-deps --no-build-isolation
cd ../
cp -r picaso-c3ce0d85973bc3b7eb6af117666c041d9a4966ab/reference picasofiles/
rm -rf picaso-c3ce0d85973bc3b7eb6af117666c041d9a4966ab
rm c3ce0d85973bc3b7eb6af117666c041d9a4966ab.zip
```

Get input files:

```sh
python input_files.py
```




```sh
mkdir -p codex_reference && cd codex_reference
for u in \
  https://github.com/natashabatalha/picaso/archive/c3ce0d85973bc3b7eb6af117666c041d9a4966ab.zip
do
  f="$(basename "$u")"
  wget -O "$f" "$u"
  unzip -q "$f"
  rm -f "$f"
done
cd ..
```