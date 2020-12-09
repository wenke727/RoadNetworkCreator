#!/bin/bash

rm test/*_output.png
rm test/*_blur_gray_output.png
rm test/*_edges_img_output.png
rm test/*_masked_edges_output.png



#for i in direct_line1/*; do python detection.py -f $i; done
for i in panos/*; do python3 detection.py -f $i; done

