#!/bin/bash

## download alo 
echo "***** Download ALO !! *****"
git clone http://mod.lge.com/hub/dxadvtech/aicontents-framework/alo.git

echo ""
echo "***** Copy experimental_plan.yaml to ALO !! *****"
cp ./experimental_plan.yaml ./alo/config -rf 


