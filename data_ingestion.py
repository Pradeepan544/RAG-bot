import pymupdf  # PyMuPDF for extracting text from PDFs
import os
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

# Initialize ChromaDB Client (Persistent Storage)
client = chromadb.PersistentClient(path="./chromadb_store")

# Load Embedding Model
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Define Collection Name
collection_name = "3d_printing_knowledge"

# Remove existing collection (for a fresh start)
try:
    client.delete_collection(collection_name)
except Exception:
    pass  # Collection may not exist

# Create New Collection
collection = client.create_collection(name=collection_name)

# Function to Extract Text from PDFs (page-wise storage)
def extract_text_from_pdf(pdf_path):
    doc = pymupdf.open(pdf_path)
    extracted_data = []
    
    for page_num in range(len(doc)):
        page_text = doc[page_num].get_text("text").strip()
        if page_text:
            extracted_data.append({
                "id": f"{os.path.basename(pdf_path)}_page_{page_num+1}",
                "text_summary": page_text,
                "embedding": embedder.encode(page_text).tolist(),
                "metadata": {
                    "source": pdf_path,
                    "page_number": page_num + 1,
                    "image_paths": ""
                }
            })
    
    return extracted_data

# PDF Files
pdf_files = ["basics-of-3d-printing.pdf", "3D-Printing-Guide.pdf", "troubleshoot.pdf"]

data = []
for pdf_file in pdf_files:
    if os.path.exists(pdf_file):
        extracted_data = extract_text_from_pdf(pdf_file)
        data.extend(extracted_data)
        print(f"Extracted & embedded: {pdf_file}")

# Text Summaries with Multiple Images
text_with_images = [
    {
        "id": "tools_3d",
        "text_summary": "Collection of 3D printing tools",
        "embedding": embedder.encode("A collection of tools and accessories commonly used for 3D printing. It includes essential hand tools such as a shovel, chisels, cut pliers, tweezers, and an Allen wrench with screws. There are also utility items like clips, gloves, a power adapter, a USB cable, and an SD card (2GB). Additionally, 3D printing-specific components like a build plate, acupuncture needle, Teflon tube filament guide, filament tube guide, and spool holder are present.").tolist(),
        "metadata": {
            "source": "manual",
            "page_number": -1,
            "image_paths": "/static/images/tools 3d.png"
        }
    },
    {
        "id": "setup_3d_printer",
        "text_summary": "Setting up 3D printer or RS 3D printer",
        "embedding": embedder.encode("A step-by-step guide on setting up and calibrating an RS 3D printer. The instructions cover: Connecting the Printer: Powering on, using USB to connect to a PC, and selecting the correct COM port. Calibrating the Build Plate: Using the software to roughly measure and fine-tune the Z-height for proper leveling. Adjusting the Nozzle Distance: Using a business card to ensure the correct gap (0.3mm) between the nozzle and the build plate. Fine-Tuning with Screws: Adjusting the screws underneath the plate to level it properly. Finalizing the Calibration: Setting the Z-height, saving settings, and ensuring correct values before printing. Temperature Settings: Setting the target temperature for the extruder (220°C) for proper filament melting.").tolist(),
        "metadata": {
            "source": "techniques",
            "page_number": -1,
            "image_paths": "/static/images/setup 1.png, /static/images/setup 2.png, /static/images/setup 3.png, /static/images/setup 4.png"
        }
    },
    {
        "id": "filament_setup",
        "text_summary": "Filament Setup",
        "embedding": embedder.encode("A step-by-step guide for handling filament on an RS 3D printer. The instructions include: Loading the Filament – Mount the filament spool, guide the filament through the tube, and insert it into the extruder. Preparing the Filament – Cut the filament tip straight to avoid blockage and insert it properly into the extruder. Feeding and Extruding – Heat the extruder to 220°C, push the filament in, and use the motor controls to feed or reverse the filament. Reversing and Changing Filament – When filament runs low, reverse it using the control panel and insert a new filament to continue printing.").tolist(),
        "metadata": {
            "source": "troubleshooting",
            "page_number": -1,
            "image_paths": "/static/images/filament 1.png, /static/images/filament 2.png, /static/images/filament 3.png"
        }
    }
]

data.extend(text_with_images)

# Insert Data into ChromaDB
collection.add(
    ids=[item["id"] for item in data],
    embeddings=[item["embedding"] for item in data],
    metadatas=[item["metadata"] for item in data],
    documents=[item["text_summary"] for item in data]
)

print("Data Ingestion Complete! 🚀")