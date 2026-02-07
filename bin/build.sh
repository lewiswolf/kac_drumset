#!/bin/bash
# Compile scikit-build backend.

# build dir
if [ ! -d ./_skbuild ]; then
	mkdir -p ./_skbuild
fi

# locate install dir
VENV_DIR=$(pipenv --venv)

# build
cmake -S . -B _skbuild \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_INSTALL_PREFIX="$VENV_DIR/lib/python3.13/site-packages"
cmake --build _skbuild -j4
cmake --install _skbuild