# clear build folder (used for testing)
cd physical-modelling-lib/build
shopt -s extglob
rm -rf !(.gitignore)
cd ../../

# build project
cmake -S ./physical-modelling-lib -B ./physical-modelling-lib/build
cd physical-modelling-lib/build
make