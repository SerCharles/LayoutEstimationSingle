

base_source=/data/sgl/matterport/v1/scans
base_target=/home/shenguanlin/geolayout_pretrain
mkdir $base_target
cd $base_target
mkdir norm 
mkdir depth 
mkdir image 
mkdir camera
mkdir camera_pre


cd $base_source
files=$(ls $folder)
for file in $files
do
    unzip $base_source/$file/matterport_color_images.zip -d $base_target
    unzip $base_source/$file/undistorted_camera_parameters.zip -d $base_target
    unzip $base_source/$file/matterport_depth_images.zip -d $base_target
    unzip $base_source/$file/undistorted_normal_images.zip -d $base_target

    chmod -R 777 $base_target
    cd $base_target/$file/matterport_color_images
    images=$(ls *)
    cd $base_target/$file/undistorted_camera_parameters
    cameras=$(ls *)
    cd $base_target/$file/matterport_depth_images
    depths=$(ls *)
    cd $base_target/$file/undistorted_normal_images
    norms=$(ls *)

    for image in $images
    do 
        mv $base_target/$file/matterport_color_images/$image $base_target/image/$image
    done

    for camera in $cameras
    do 
        mv $base_target/$file/undistorted_camera_parameters/$camera $base_target/camera_pre/$camera
    done

    for depth in $depths
    do 
        mv $base_target/$file/matterport_depth_images/$depth $base_target/depth/$depth
    done

    for norm in $norms
    do 
        mv $base_target/$file/undistorted_normal_images/$norm $base_target/norm/$norm
    done
    rm -rf $base_target/$file
done