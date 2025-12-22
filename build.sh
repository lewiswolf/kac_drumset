#!/bin/bash

# build dir
if [ ! -d ./_skbuild ]; then
	mkdir -p ./_skbuild
fi

# build
cmake -S . -B _skbuild
cmake --build _skbuild --config Release -j4
