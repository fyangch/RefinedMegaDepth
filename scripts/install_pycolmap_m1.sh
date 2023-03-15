git clone --recursive git@github.com:colmap/pycolmap.git
cd pycolmap
cd pipeline
sed -i "" 's/\.def_readwrite("ba_global_use_pba"/\/\/\.def_readwrite("ba_global_use_pba"/g' sfm.cc
sed -i "" 's/\.def_readwrite("ba_global_pba_gpu_index"/\/\/\.def_readwrite("ba_global_pba_gpu_index"/g' sfm.cc
sed -i "" 's/\&Opts::ba_global_pba_gpu_index)/\/\/\&Opts::ba_global_pba_gpu_index)/g' sfm.cc
cd ..
export CMAKE_PREFIX_PATH="/opt/homebrew/opt/qt@5/lib/cmake/Qt5"
export CGAL_DATA_DIR="/opt/homebrew/Caskroom/miniforge/base/envs/${CONDA_DEFAULT_ENV}/lib/cmake/CGAL"
pip install -v .
