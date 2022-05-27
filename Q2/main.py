import fitz # install using: pip install PyMuPDF
import re
import pandas as pd

file = r"src/Q2.pdf"
export = r"src/Q2.csv"

# Regular Expression to Search Data
regexPage = r"Service Diagnosis\s*\n*(\d+)\s*\n*"
regexDescription = r"Description\s*\n*(.+?[\w\)])\s*\n*Possible Root"
regexCause = r"Possible Root\s*\n*cause\s*\n*(.+?[\w\.])\s*\n*Troubleshooting"

pageNumber = []
description = []
rootCause = []

def main(file) :
    with fitz.open(file) as doc:
        for page in doc :
            text = page.get_text()
            matchPage = re.findall(regexPage, text, re.DOTALL)
            matchDescription = re.findall(regexDescription, text, re.DOTALL)
            matchCause = re.findall(regexCause,text,re.DOTALL)

            # If multiple data found on same page
            if len(matchDescription)==len(matchCause) and len(matchDescription)>1 :
                for i in range(len(matchDescription)) :
                    pageNumber.append(matchPage[0])
                    description.append(matchDescription[i].replace("\n"," "))
                    rootCause.append(matchCause[i].replace("\n"," "))
            else :
                pageNumber.append(matchPage[0])
                description.append(matchDescription[0].replace("\n"," "))
                rootCause.append(matchCause[0].replace("\n"," "))

    # Export to csv
    df = pd.DataFrame(list(zip(description,rootCause, pageNumber)),columns=['Description', 'Possible Root Cause','Page Number'])
    df.to_csv(export)
            
if __name__ == '__main__':
    main(file)
