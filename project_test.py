from groq import Groq
import os
import base64
from dotenv import load_dotenv
load_dotenv()
import json
from fastapi import File, UploadFile, HTTPException
from pdf2image import convert_from_path
import cv2
import numpy as np

from fastapi import FastAPI,File
app = FastAPI()
# import fitz
from PIL import Image
from pyzbar.pyzbar import decode
import jwt
from qreader import QReader
import Levenshtein


def generate_details(image_path : str):
    
    def convert_pdf_to_images(pdf_path):
        images = convert_from_path(pdf_path,poppler_path=r'c:\Users\BitCoding Solutions\Downloads\Release-25.12.0-0\poppler-25.12.0\Library\bin')
        images[0].save(r'myfile.png'.format(pdf_path=pdf_path[:-4]),'PNG')
        return 'myfile.png'
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')   
        
    png_path = convert_pdf_to_images(image_path)
    # Load imgae, grayscale, Gaussian blur, Otsu's threshold
    
    # Create a QReader instance
    qreader = QReader()
    # Get the image (as RGB)
    image = cv2.cvtColor(cv2.imread(png_path), cv2.COLOR_BGR2RGB)
    
    # Use the detect_and_decode function to get the decoded QR data
    decoded_texts = qreader.detect_and_decode(image=image)

    
    data=jwt.decode(decoded_texts[0], options={"verify_signature": False})
    qr_code_data=eval(data['data'])
    
    # Getting the base64 string 
    base64_image = encode_image(png_path)
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    prompt = """
You are an expert invoice data extraction system.

TASK:
Extract invoice details from the provided image and return ONLY valid JSON.
Do not add explanations. Do not add extra text.

STRICT EXTRACTION RULES:

1. IRN Rules:
   - IRN must be exactly 64 characters.
   - If IRN contains "-", remove all "-" characters.
   - If IRN appears in multiple lines, concatenate them.
   - If IRN length is more than 64 after cleanup, truncate to first 64 characters.
   - If IRN length is less than 64, return None.

2. GSTIN Rules:
   - hiib_gstin = Get This Value from the Hyundai India Insurance Broking private limited(HIIB) Who is the buyer 
   - dealer_gstin = Will Get value from the other service provider details Who is the Seller here 
   - Both must be different.
   - GSTIN format = 15 characters.

3. Date Format Rules:
   - invoice_date and ack_date must be in DD-MM-YYYY format.
   - Convert from any format if necessary.

4. Null Handling:
   - If any value is not found, return None.
   - Do NOT guess values.
   - Do NOT hallucinate.

5. Multi-line Pattern Handling:
   If any alphanumeric string is broken across lines like:

   ABC12345-
   XYZ6789

   It must be combined as:
   ABC12345XYZ6789

6. Numeric Fields:
   Extract only numeric values without symbols like ₹, commas, etc.

RETURN FORMAT (STRICT JSON ONLY):

{
    "irn": "",
    "ack_no": "",
    "ack_date": "",
    "invoice_no": "",
    "invoice_date": "",
    "taxable_value": "",
    "cgst_amount": "",
    "sgst_utgst_amount": "",
    "igst_amount": "",
    "total_invoice_value": "",
    "dealer_code": "",
    "hiib_misp_code": "",
    "account_holders_name": "",
    "bank_name": "",
    "account_no": "",
    "branch": "",
    "bank_ifsc": "",
    "micr_code": "",
    "hiib_gstin": "",
    "dealer_gstin": "",
    "hiib_pincode": "",
    "dealer_pincode": "",
    "hiib_state_code": "",
    "dealer_state_code": "",
    "msme_code": "",
    "dealer_pan": "",
    "sac": "",
    "consigner_details": "",
    "consigner_address": "",
    "consigner_pincode": "",
    "buyer_name": "",
    "buyer_address": "",
    "buyer_pincode": "",
    "consigner_place_of_supply": "",
    "buyer_place_of_supply": "",
    "description_of_service": "",
    "oem": "",
    "quantity": "",
    "period_of_service": ""
}
Return only JSON.
"""  
    completion = client.chat.completions.create(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ],
    temperature=1,
    max_completion_tokens=1024,
    top_p=1,
    stream=False,
    response_format={"type": "json_object"},
    stop=None
    )
    
    result=completion.choices[0].message.content
    
    res=json.loads(result)
    
    list1=['irn','hiib_gstin','dealer_gstin']
    
    for i in list1:
        str1 = res[i]
        if i=='irn':
            str2=qr_code_data['Irn']
            distance=Levenshtein.distance(str1,str2)
            if distance<=4:
                res[i]=str2
        elif i=='hiib_gstin':
            str2=qr_code_data['BuyerGstin']
            distance=Levenshtein.distance(str1,str2)
            if distance<=2:
                res[i]=str2
        elif i=='dealer_gstin':
            str2=qr_code_data['SellerGstin']
            distance=Levenshtein.distance(str1,str2)
            if distance<=2:
                res[i]=str2
    return res

@app.post("/details")
async def get_topic_summary(file: UploadFile = File()):
    try:
        contents = file.file.read()
        with open(file.filename, 'wb') as f:
            f.write(contents)
    except Exception:
        raise HTTPException(status_code=500, detail='Something went wrong')
    finally:
        file.file.close()
    result=generate_details(file.filename)
    return result
