# check if cmake exists
if ! command -v cmake >/dev/null
then
	echo "\033[0;91mWARNING\033[0m: please install cmake to use this code."
	exit
fi

# clear build folder (used for testing)
# cd physical-modelling-lib/build
# 	shopt -s extglob
# 	rm -rf !(.gitignore)
# cd ../../

# build project
cmake -S ./physical-modelling-lib -B ./physical-modelling-lib/build
cd physical-modelling-lib/build
	make