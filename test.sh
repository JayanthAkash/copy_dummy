rosdep install -y --from-paths src --ignore-src --rosdistro noetic -r --os=debian:buster
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
sudo src/catkin/bin/catkin_make_isolated --install -DCMAKE_BUILD_TYPE=Release --install-space /opt/ros/noetic -j1 -DPYTHON_EXECUTABLE=/usr/bin/python
