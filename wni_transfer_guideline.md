---
# pandoc wni_transfer_guideline.md -o pdf/wni_transfer_guideline.pdf --from markdown --template /home/vmodev/.local/share/pandoc/templates/eisvogel.tex --listings --pdf-engine=xelatex --toc --number-sections

papersize: a4
lang: vi-VN
# geometry:
#     - top=30mm
#     - left=20mm
#     - right=20mm
#     - heightrounded
documentclass: article
title: Bijikon server guideline
author: VMO Holdings .Jsc
date: 20-01-2021
titlepage: true
toc-own-page: true
logo: images/vmo.png
header-includes: 
      - |
        ``` {=latex}
        \let\originAlParaGraph\paragraph
        \renewcommand{\paragraph}[1]{\originAlParaGraph{#1} \hfill}
        ```
...


# WNI script

## Diagolization result script
### Requirements

Given a table of prediction in different timestamps and localtion, export an diagolized table for each lclid (location name)

The columns in the given table include:

* context
* lclid
* t_0
* t_1
* ...
* t_36

![Data from customer](images/screen_1.png)

Important columns:

1. lclid: place name
2. t_1 -> t_36: accuracy

For each lclid, rearrange data as follow:

![Rearranged data](images/screen_2.png)

The script has been finished and can use immediately.

## Transpose accuracy result script


## Storm and map drawing script using matplotlib



# Improve accuracy of WNI nowcasting using deep learning instead of traditional machine learning method

## Applied deep learning on local optical flow

Local optical flow right now is using Hornchunk as a prediction method. Accuracy can be improve using deep learning. Implement the following method on the nowcasing code.

PWCNet


## Applied deep learning on global optical flow


## Applied deep learning on radar image prediction

