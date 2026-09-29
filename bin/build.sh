#!/bin/bash
# Compile scikit-build backend.

# build dir
if [ ! -d ./_skbuild ]; then
	mkdir -p ./_skbuild
fi

# locate install dir
SITE_PACKAGES=$(pipenv run python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")

# build
cmake -S . -B _skbuild -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$SITE_PACKAGES"
cmake --build _skbuild -j4
cmake --install _skbuild