#!/bin/bash

rm -rf doc/_apidoc/
rm -rf doc/build/
sphinx-build -W -b html doc/ doc/build
exit $?
