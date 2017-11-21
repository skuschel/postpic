#!/bin/bash

# update the apidoc folder, which is used 
# by sphinx autodoc to convert docstings
# into the manual pages.

# need to rerun this file whenever a new python file
# is created within postpic.

# Stephan Kuschel, 2017

rm source/apidoc/*
sphinx-apidoc -o source/apidoc ../postpic -M
