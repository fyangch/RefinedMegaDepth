export data="/Volumes/Extreme_SSD/MegaDepth/scenes"

for scene in 0015 0016 0022 0047 0058 0064 0121 0133 0229; do # 5014 5015 5016 5018; do
    python megadepth/visualization/view_sparse_model.py --data_path ${data} --scene ${scene} --model_name baseline
    python megadepth/visualization/view_sparse_model.py --data_path ${data} --scene ${scene} --model_name superpoint_max-superglue-netvlad-50
    python megadepth/visualization/view_dense_model.py --data_path ${data} --scene ${scene} --model_name baseline-superglue-netvlad-50
    python megadepth/visualization/view_dense_model.py --data_path ${data} --scene ${scene} --model_name superpoint_max-superglue-netvlad-50
    # python megadepth/visualization/view_sparse_model.py --data_path ${data} --scene ${scene} --model_name superpoint_max-superglue-exhaustive
done
