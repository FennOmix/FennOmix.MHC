rm -rf build
conda env remove -n fennomixdocs
conda create -n fennomixdocs python=3.11 -y
conda activate fennomixdocs
pip install '../.[development]'
make html
conda deactivate
