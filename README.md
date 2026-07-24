# P2 Checkout Incident Analysis
This is the code behind the charts and stats used in the P2 checkout incident review. To pick this up after my internship ends, these are the following steps:

## Packages
Ensure Python is of current version.
Use the following command to install the necessary packages.
```
pip install pandas numpy scipy matplotlib openpyxl lifelines
```
## Getting the data
You need XO_incidents_RCA_merchants.xlsx. The sheet the code read is called "Data - Updated." Place the code into the folder you have.

## Running
Use the following command:
```
python p2_incident_statistics.py
```
This loads the data and prints statistics from the analysis to your terminal. You will also receive the necessary charts as a PNG file, labelled sequentially, saved to the same folder you have the code/excel sheet in.

## Code Layout
Each function has a short comment above it saying what statistical test it uses (if any), what it assumes, and relevancy. Most of these tests are descriptive statistics and not complex tests (e.g., median/mode/mean/counts)

## Repo Layout
- p2_incident_statistics.py
- README_statistics.md
- Archive: Old BERT model code to read tickets

- NOTE: You will need to supply XO_incidents_RCA_merchants.xlsx and place it in the same folder

## Hardcoded Metrics notes
Nothing is hardcoded except the sample date window and the Feb-Apr 2025 spike window in spike(), both of which are tied to specific historical periods and shouldn't need to change unless a stakeholder asks for a different window to be analyzed. The excel workbook's name and sheet's name is also hardcoded.
