import docx
from docx2pdf import convert
import os
from fpdf import FPDF

def doc_to_pdf():
    
    file_paths = [
        "emergency_rules",
        "recently_adopted_rules",
        "benefits_manual",
    ]
    
    # #convert text files to pdf
    # for file_path in file_paths:
    #     for file in os.listdir(file_path):
    #         if file.endswith(".docx"):
    #             convert(f"{file_path}/{file}")
    #         elif file.endswith(".doc"):
    #             doc = docx.Document(f"{file_path}/{file}")
    #             doc.save(f"{file_path}/{file}.docx")
    #             convert(f"{file_path}/{file}.docx")

    #convert .txt files to pdf
    for file_path in file_paths:
        for file in os.listdir(file_path):
            if file.endswith(".txt"):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                
                # Set margins (left, top, right)
                pdf.set_left_margin(10)
                pdf.set_top_margin(10)
                pdf.set_right_margin(10)
                
                # Calculate the width of the cell
                effective_page_width = pdf.w - pdf.l_margin - pdf.r_margin

                with open(f"{file_path}/{file}", "r", encoding='utf-8') as f:
                    for line in f:
                        pdf.multi_cell(effective_page_width, 10, txt=line.strip(), align='C')
                
                pdf.output(f"{file_path}/{file}.pdf")
                print(f"Converted {file} to PDF")
    
    return "Files converted to PDF"


if __name__ == "__main__":
    doc_to_pdf()
    


