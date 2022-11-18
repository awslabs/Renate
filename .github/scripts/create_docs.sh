#!/bin/bash

rm -rf doc/_apidoc/
rm -rf doc/build/
sphinx-build -b html doc/ doc/build
