# Kinyarwanda: Analyser and Generator
This repository contains all the python files and data I wrote and used for my bachelor thesis on the analysis and generation of Kinyarwanda verb forms.

## Files

Most of the files are used for the analyser with the generator files named "gen_". However, some of the analyser files are used for the generator as well.
The complete data can be found in "kinyarwandaVerbsExtensionsSylJuliaTestCombo.csv", with the other three csv-files being the splitted files for training, development and testing.

## Usage

I used an anaconda environment with Python 3.7.6 (default, Jan  8 2020, 20:23:39) [MSC v.1916 64 bit (AMD64)]. In order to run the code I did the following:

```bash
ipython
```
and then

```bash
run main_total
```
```bash
run main_individual
```
```bash
run gen_main
```
in order to run the analyser with the total accuracy, the analyser with the individual accuracy and the generator.
