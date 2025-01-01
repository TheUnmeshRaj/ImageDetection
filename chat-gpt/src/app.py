import random

import requests
import torch
from bs4 import BeautifulSoup
from datasets import load_dataset
from flask import Flask, jsonify, request  # type: ignore
from flask_cors import CORS
from PIL import Image  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import models, transforms
from transformers import AutoModel, AutoTokenizer

app = Flask(__name__)
CORS(app)

# Load datasets
dataset = load_dataset("Mostafijur/Skin_disease_classify_data")
dataset1 = load_dataset("brucewayne0459/Skin_diseases_and_care")

device = torch.device('cpu')
classes = {
    0: 'Acne and Rosacea',
    1: 'Actinic Keratosis Basal Cell Carcinoma',
    2: 'Nail Fungus',
    3: 'Psoriasis Lichen Planus',
    4: 'Seborrheic Keratoses',
    5: 'Tinea Ringworm Candidiasis',
    6: 'Warts Molluscum'
}

# Load models and tokenizers
tokenizer1 = AutoTokenizer.from_pretrained("Unmeshraj/skin-disease-detection")
model1 = AutoModel.from_pretrained("Unmeshraj/skin-disease-detection")
tokenizer2 = AutoTokenizer.from_pretrained("Unmeshraj/skin-disease-treatment-plan")
model2 = AutoModel.from_pretrained("Unmeshraj/skin-disease-treatment-plan")

image_model = models.resnet18(pretrained=False)
image_model.fc = torch.nn.Linear(image_model.fc.in_features, len(classes))
image_model.load_state_dict(torch.load("C:/Users/unmes/OneDrive/Desktop/EL/SkinDiseaseAI2/chat-app/api/model.pth", map_location=device))
image_model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Helper functions
def embed_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

queries, diseases, embeddings = [], [], []
for example in dataset['train']:
    query = example['Skin_disease_classification']['query']
    disease = example['Skin_disease_classification']['disease']
    queries.append(query)
    diseases.append(disease)
    query_embedding = embed_text(query, tokenizer1, model1)
    embeddings.append(query_embedding)

topics, information, topic_embeddings = [], [], []
for example in dataset1['train']:
    topic = example['Topic']
    info = example['Information']
    topics.append(topic)
    information.append(info)
    topic_embedding = embed_text(topic, tokenizer2, model2)
    topic_embeddings.append(topic_embedding)

def find_similar_disease(input_query):
    input_embedding = embed_text(input_query, tokenizer1, model1)
    similarities = [
        cosine_similarity(input_embedding.detach().numpy(), emb.detach().numpy())[0][0]
        for emb in embeddings
    ]
    return diseases[similarities.index(max(similarities))]

def format_treatment(treatment):
    formatted_treatment = treatment.replace('**', '<strong>').replace('**', '</strong>')  # Bold text formatting
    lines = treatment.split('\n')
    formatted_lines = []
    
    for line in lines:
        line = line.strip()
        if line.startswith('*'):  # If it's a list item
            formatted_lines.append(f'<li>{line[1:].strip()}</li>')  # Remove * and wrap in <li> tags
        elif line:  # If it's a non-empty line
            formatted_lines.append(f'<p>{line}</p>')  # Wrap text in <p> tags
    
    return f'<div class="formatted-treatment">{"".join(formatted_lines)}</div>'

treatment_plans = {
    'Acne and Rosacea': """
        **Acne and Rosacea: Diagnosis and Treatment**

        **Diagnosis**: Acne is diagnosed based on the appearance of pimples, cysts, and blackheads on the skin. Rosacea is diagnosed by evaluating the presence of redness, visible blood vessels, and pimples, primarily on the face.

        **Treatment**:
        * Topical Retinoids: Help prevent pores from becoming clogged.
        * Benzoyl Peroxide: Kills acne-causing bacteria.
        * Antibiotics: Used for severe acne to reduce bacteria and inflammation.
        * Chemical Peels: Helps with acne scars.
        * Oral Antibiotics: Prescribed for severe cases.

        Skin Treatments for Rosacea:
        * Topical Metronidazole: Reduces redness and inflammation.
        * Oral Antibiotics: For severe rosacea.
        * Laser Therapy: To reduce redness and visible blood vessels.
        * Avoid Triggers: Sun exposure, hot beverages, spicy foods.

        **Outcome**: Acne treatment can take weeks to months for results, but it can improve with the right regimen. Rosacea can be controlled with ongoing treatment, but there is no cure.
    """,
    'Actinic Keratosis Basal Cell Carcinoma': """
        **Actinic Keratosis and Basal Cell Carcinoma: Diagnosis and Treatment**

        **Diagnosis**: Actinic keratosis is diagnosed by the appearance of scaly, dry patches. Basal cell carcinoma is diagnosed by biopsy after examining a growing, pearly, or ulcerated bump on the skin.

        **Treatment**:
        * Cryotherapy: Freezing the abnormal cells.
        * Topical Treatments: 5-fluorouracil (5-FU) to treat actinic keratosis.
        * Curettage and Electrodesiccation: Removal of the lesion by scraping.
        * Mohs Surgery: A precise surgical method for basal cell carcinoma.
        * Radiation Therapy: For non-surgical candidates.
        * Excision: Surgical removal of lesions.

        **Outcome**: Actinic keratosis can be treated effectively, but recurrences are common. Basal cell carcinoma has a high cure rate if detected early, but it may require multiple treatments.
    """,
    'Nail Fungus': """
        **Nail Fungus: Diagnosis and Treatment**

        **Diagnosis**: Nail fungus is diagnosed by examining the nail's appearance, including thickening, discoloration, and crumbling. A sample of nail debris may be taken for lab analysis.

        **Treatment**:
        * Antifungal Medications: Oral antifungals like terbinafine or itraconazole are prescribed.
        * Topical Antifungal Treatments: Applied directly to the nail.
        * Laser Treatment: Uses lasers to target the fungus.
        * Surgery: In severe cases, the infected nail may be removed.

        **Outcome**: Treatment may take several months, and the recurrence rate is high if not fully treated.
    """,
    'Psoriasis Lichen Planus': """
        **Psoriasis and Lichen Planus: Diagnosis and Treatment**

        **Diagnosis**: Psoriasis is diagnosed through skin examination and biopsy if necessary. Lichen planus is diagnosed by examining the skin, nails, and mouth, often confirmed with a biopsy.

        **Treatment**:
        Psoriasis Treatments:
        * Topical Corticosteroids: To reduce swelling and redness.
        * Vitamin D Analogues: Helps to slow down skin cell production.
        * Phototherapy: Uses ultraviolet light to treat affected areas.
        * Biologic Drugs: Target immune system components responsible for psoriasis.

        Lichen Planus Treatments:
        * Antihistamines: To relieve itching.
        * Topical Corticosteroids: Reduce inflammation and lesions.
        * Retinoids: To help with skin lesions.
        * Oral Medications: For widespread or severe cases.

        **Outcome**: Psoriasis can be managed with treatment, but it tends to flare up periodically. Lichen planus often resolves but may take months to years.
    """,
    'Seborrheic Keratoses': """
        **Seborrheic Keratoses: Diagnosis and Treatment**

        **Diagnosis**: Seborrheic keratoses are diagnosed by their characteristic appearance of raised, waxy, or scaly lesions that appear brown or black. No biopsy is usually necessary unless a diagnosis is uncertain.

        **Treatment**:
        * Cryotherapy: Freezing the lesions off.
        * Curettage: Scraping off the growths.
        * Laser Treatment: Using a laser to remove the lesion.
        * Excision: Surgical removal of larger lesions.

        **Outcome**: Seborrheic keratoses are benign and not harmful. The treatment is typically cosmetic and may not be needed unless the lesions cause irritation.
    """,
    'Tinea Ringworm Candidiasis': """
        **Tinea, Ringworm, and Candidiasis: Diagnosis and Treatment**

        **Diagnosis**: Tinea and ringworm are diagnosed by examining the appearance of circular, red, scaly rashes. Candidiasis is identified by its characteristic rash, typically in warm, moist areas of the body.

        **Treatment**:
        * Antifungal Creams: Over-the-counter creams like clotrimazole and terbinafine for ringworm.
        * Oral Antifungals: For more severe cases of tinea and candidiasis.
        * Topical Antifungal Powder: Used to treat fungal infections in areas like the feet.
        * Steroid Creams: For inflammation and itching.

        **Outcome**: Tinea and ringworm can be treated effectively with antifungals, but recurrence is possible. Candidiasis is also treatable, but it may require longer courses of medication.
    """,
    'Warts Molluscum': """
        **Warts and Molluscum: Diagnosis and Treatment**

        **Diagnosis**: Warts are diagnosed by the appearance of raised, rough growths, typically on hands or feet. Molluscum contagiosum is diagnosed by the presence of small, dome-shaped lesions.

        **Treatment**:
        * Cryotherapy: Freezing the wart or lesion.
        * Topical Treatments: Use of salicylic acid or cantharidin.
        * Laser Removal: For stubborn warts.
        * Excision: Surgical removal of warts.

        **Outcome**: Warts generally resolve with treatment, but new ones can appear. Molluscum typically resolves on its own, but it may persist for months.
    """
}

def find_treatment_plan(disease_name):
    return format_treatment(treatment_plans.get(disease_name, "Treatment plan not available"))

def predict_image(img):
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = image_model(img_tensor)
        _, predicted = torch.max(outputs, 1)
    return classes[predicted.item()]

def fetchDoctors(location, query, mode, backupQuery, backupMode, locality):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"
    }

    def fetch_from_url(query, mode):
        url = f"https://www.practo.com/search/doctors?results_type=doctor&q=%5B%7B%22word%22%3A%22{query}%22%2C%22autocompleted%22%3Atrue%2C%22category%22%3A%22{mode}%22%7D%2C%7B%22word%22%3A%22{locality}%22%2C%22autocompleted%22%3Atrue%2C%22category%22%3A%22locality%22%7D%5D&city={location}"
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return [], response.status_code

        soup = BeautifulSoup(response.text, "html.parser")
        doctor_data = []

        for anchor in soup.find_all("a", href=True, class_=False):
            if "/doctor/" in anchor["href"] and anchor.find("h2", class_="doctor-name"):
                name = anchor.find("h2", class_="doctor-name").get_text(strip=True)
                link = "https://www.practo.com" + anchor["href"]
                doctor_data.append({"name": name, "link": link})
        return doctor_data, response.status_code

    doctor_data, status_code = fetch_from_url(query, mode)

    if not doctor_data:
        return {"error": f"No doctors found for {query} in {location}"}

    return doctor_data

@app.route('/api/ImageAi', methods=['POST'])
def image_ai():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    image = Image.open(file.stream).convert('RGB')
    predicted_disease = predict_image(image)
    treatment_plan = find_treatment_plan(predicted_disease)

    locality = "Indiranagar"
    location = "bangalore"
    query = predicted_disease.replace(" ", "%20")
    mode = "symptom"

    doctor_info = fetchDoctors(location, query, mode, "", "", locality)

    return jsonify({
        'result': predicted_disease,
        'treatment': treatment_plan,
        'doctors': doctor_info
    })

if __name__ == '__main__':
    app.run(debug=True, port=5002)
