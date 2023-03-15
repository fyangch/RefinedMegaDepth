cd external_dependencies
brew install qt5
conda install cmake boost eigen freeimage gflags metis suitesparse ceres-solver glew cgal glog=0.5.0
git clone -b dev git@github.com:colmap/colmap.git
cd colmap
sed -i "" 's/OPENMP_ENABLED "Whether to enable OpenMP parallelization" ON/OPENMP_ENABLED "Whether to enable OpenMP parallelization" OFF/g' CMakeLists.txt
mkdir build
cd build
cmake .. -DBOOST_STATIC=OFF -DQt5_DIR="/opt/homebrew/opt/qt@5/lib/cmake/Qt5" -DCGAL_DATA_DIR="/opt/homebrew/Caskroom/miniforge/base/envs/${CONDA_DEFAULT_ENV}/lib/cmake/CGAL"
make
sudo make install
sudo install_name_tool -add_rpath /opt/homebrew/Caskroom/miniforge/base/envs/${CONDA_DEFAULT_ENV}/lib /usr/local/bin/colmap
cd ../../..
