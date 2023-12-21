#!/bin/bash

wget -P ${1} https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth
wget -P ${1} https://download.pytorch.org/models/resnet101-cd907fc2.pth
wget -P ${1} https://download.pytorch.org/models/vgg19-dcbb9e9d.pth
