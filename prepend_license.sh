#!/bin/bash

for folder in "src" "test"
do
	for i in $(find $folder -name '*.py');
    do
      if ! grep -q "Copyright Amazon.com" $i;
      then
        sed -i "" "1s;^;# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.\n# SPDX-License-Identifier: Apache-2.0\n;" $i
      fi
    done
done
