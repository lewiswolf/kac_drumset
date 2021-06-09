# install cpp dependencies
cd physical-modelling-lib/includes
	git clone -b v2.6.2 https://github.com/pybind/pybind11.git
	git clone -b 3.3.4 https://github.com/glfw/glfw.git
	git clone -b glew-2.2.0 https://github.com/nigels-com/glew.git
cd ../../

# build cpp and install python dependencies
sh bin/compile_cpp.sh
pipenv install
exit