#sudo apt-get install python3-rosdep python3-rosinstall-generator python3-vcstools python3-vcstool build-essential
#sudo python -m pip install --upgrade setuptools
#sudo python -m pip install -U rosdep rosinstall_generator vcstool
#sudo rosdep init
#rosdep update
#mkdir ~/ros_catkin_ws
cd ~/ros_catkin_ws
#rosinstall_generator ros_base --rosdistro noetic --deps --tar > noetic-ros_base.rosinstall
#mkdir ./src
#vcs import --input noetic-ros_base.rosinstall ./src
#rosdep install --from-paths ./src --ignore-packages-from-source --rosdistro noetic -y
./src/catkin/bin/catkin_make_isolated --install -DCMAKE_BUILD_TYPE=Release -DSETUPTOOLS_DEB_LAYOUT=OFF
