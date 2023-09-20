mkdir -p external_dependencies
cd external_dependencies
git clone https://github.com/veichta/day-night-classification.git
cd day-night-classification
pip install -e .
cd ../..
