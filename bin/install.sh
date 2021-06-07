# check if cmake exists
if ! command -v cmake >/dev/null
then
	echo "\033[0;91mWARNING\033[0m: please install cmake to use this code."
	exit
fi

# install cpp dependencies
cd physical-modelling-lib
	git clone https://github.com/pybind/pybind11.git
cd ../

# build cpp and install python dependencies
sh bin/compile_cpp.sh
pipenv install
exit