#!/bin/bash

exit_code=0

for folder in "examples" "src" "test"
do
	for i in $(find $folder -name '*.py');
    do
      if [ -s $i ] && ! grep -q "Copyright Amazon.com" $i;
      then
        sed -i "" "1s;^;# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.\n# SPDX-License-Identifier: Apache-2.0\n;" $i
        exit_code=1
      fi
    done
done

exit $exit_code
