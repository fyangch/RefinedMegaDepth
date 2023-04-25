export data="/Volumes/Extreme_SSD/MegaDepth/scenes"

for scene in 5014 5015 5016 5018; do
    python megadepth/visualization/create_movie.py \
        --data_path ${data} \
        --scene ${scene} \
        --cameras \
        --zoom_in \
        --visible \
        --model_type sfm \
        --model_name superpoint_max-superglue-netvlad-50-KA+BA
    python megadepth/visualization/create_movie.py \
        --data_path ${data} \
        --scene ${scene} \
        --cameras \
        --zoom_in \
        --visible \
        --model_type mvs \
        --model_name superpoint_max-superglue-netvlad-50-KA+BA
    python megadepth/visualization/create_movie.py \
        --data_path ${data} \
        --scene ${scene} \
        --cameras \
        --zoom_in \
        --visible \
        --model_type sfm \
        --model_name superpoint_max-superglue-exhaustive-KA+BA
    python megadepth/visualization/create_movie.py \
        --data_path ${data} \
        --scene ${scene} \
        --cameras \
        --zoom_in \
        --visible \
        --model_type mvs \
        --model_name superpoint_max-superglue-exhaustive-KA+BA
done
