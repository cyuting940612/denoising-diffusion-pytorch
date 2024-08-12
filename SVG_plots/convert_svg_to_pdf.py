import cairosvg
import os

def convert_svg_to_pdf(input_svg, output_pdf, dpi=300):
    try:
        # Convert SVG to PDF with specified DPI
        cairosvg.svg2pdf(url=input_svg, write_to=output_pdf, dpi=dpi)
        print(f"Conversion successful: {input_svg} -> {output_pdf}")
    except Exception as e:
        print(f"Error converting SVG to PDF: {e}")


# List of SVG files
svg_files = ['Aggregate(10weekday_90weekend).svg', 'Aggregate(90weekday_10weekend).svg', 'Conditional_Synthetic_User(Weekday).svg', 'Real_User_Load.svg', 'Real_User_Load_Temperature(3D).svg', 'Real_Whole_LMP.svg', 'Real_Whole_Load.svg', 'Real_Whole_Load_Temperature(3D).svg', 'Real_Whole_Temperature.svg', 'Synthetic_User_Load.svg', 'Synthetic_User_Load_Temperature(3D).svg', 'Synthetic_Whole_LMP.svg', 'Synthetic_Whole_Load.svg', 'Synthetic_Whole_Load_Temperature(3D).svg', 'Synthetic_Whole_Temperature.svg']

# Convert each SVG file to PDF
for svg_file in svg_files:
    # Construct output PDF file name
    pdf_file = os.path.splitext(svg_file)[0] + '.pdf'

    # Call the conversion function
    convert_svg_to_pdf(svg_file, pdf_file, dpi=300)