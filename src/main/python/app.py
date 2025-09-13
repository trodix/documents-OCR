from mistralai import Mistral
import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import base64
import json
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
import logging
import PyPDF2
import io
import fitz  # pymupdf pour une meilleure extraction
from PIL import Image
import tempfile

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Generic Document Analysis Service", version="2.0.0")

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialisation du client Mistral
client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

class DocumentFieldsRequest(BaseModel):
    fieldsToExtract: List[str]
    language: Optional[str] = "fr"
    additionalOptions: Optional[Dict[str, Any]] = {}

class DocumentAnalysisResponse(BaseModel):
    documentType: Optional[str]  # null si non déterminé
    extractedFields: Dict[str, Any]
    rawResponse: str
    confidence: float = 0.0
    status: str = "success"
    errors: List[str] = []

# Types de documents reconnus avec leurs caractéristiques
DOCUMENT_TYPES = {
    "facture": {
        "keywords": ["facture", "invoice", "devis", "montant", "tva", "ht", "ttc", "fournisseur", "client", "échéance"],
        "typical_fields": ["numero_facture", "date_facture", "montant_ttc", "montant_ht", "tva", "fournisseur", "client"]
    },
    "bon_commande": {
        "keywords": ["commande", "order", "bon de commande", "quantité", "articles", "prix unitaire", "livraison"],
        "typical_fields": ["numero_commande", "date_commande", "articles", "quantites", "prix_unitaires"]
    },
    "bon_livraison": {
        "keywords": ["livraison", "delivery", "transporteur", "expédition", "réception", "bon de livraison"],
        "typical_fields": ["numero_livraison", "date_livraison", "transporteur", "articles_livres"]
    },
    "carte_identite_francaise": {
        "keywords": ["carte", "identité", "république française", "nationalité française", "né(e)", "sexe"],
        "typical_fields": ["nom", "prenoms", "date_naissance", "lieu_naissance", "sexe", "nationalite"]
    },
    "marche": {
        "keywords": ["marché", "contrat", "accord", "convention", "parties contractantes", "objet", "clause"],
        "typical_fields": ["numero_marche", "parties_contractantes", "objet_marche", "montant_marche", "duree"]
    },
    "passeport": {
        "keywords": ["passeport", "passport", "république française", "type", "code pays"],
        "typical_fields": ["nom", "prenoms", "date_naissance", "lieu_naissance", "numero_passeport", "date_expiration"]
    },
    "permis_conduire": {
        "keywords": ["permis", "conduire", "driving", "license", "catégorie", "véhicule"],
        "typical_fields": ["nom", "prenoms", "date_naissance", "numero_permis", "categories", "date_delivrance"]
    }
}

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extrait le texte d'un PDF en utilisant PyMuPDF (fitz)
    """
    try:
        # Méthode 1: PyMuPDF (plus robuste)
        pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
        text_content = ""

        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text_content += page.get_text()
            text_content += "\n\n"  # Séparateur entre pages

        pdf_document.close()

        if text_content.strip():
            return text_content.strip()

    except Exception as e:
        logger.warning(f"Échec extraction PyMuPDF: {e}")

    try:
        # Méthode 2: PyPDF2 (fallback)
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text_content = ""

        for page in pdf_reader.pages:
            text_content += page.extract_text()
            text_content += "\n\n"

        if text_content.strip():
            return text_content.strip()

    except Exception as e:
        logger.warning(f"Échec extraction PyPDF2: {e}")

    return ""

def pdf_to_images(file_bytes: bytes, max_pages: int = 5) -> List[str]:
    """
    Convertit les pages d'un PDF en images base64
    Limite aux premières pages pour éviter les timeouts
    """
    images_b64 = []

    try:
        pdf_document = fitz.open(stream=file_bytes, filetype="pdf")

        # Limiter le nombre de pages pour éviter les timeouts
        num_pages = min(len(pdf_document), max_pages)

        for page_num in range(num_pages):
            page = pdf_document.load_page(page_num)

            # Convertir en image
            mat = fitz.Matrix(2.0, 2.0)  # Facteur d'échelle pour meilleure qualité
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")

            # Encoder en base64
            img_b64 = base64.b64encode(img_data).decode("utf-8")
            images_b64.append(img_b64)

        pdf_document.close()

    except Exception as e:
        logger.error(f"Erreur conversion PDF en images: {e}")

    return images_b64

def detect_document_type_from_text(text_content: str) -> Optional[str]:
    """
    Détecte le type de document basé sur le contenu textuel extrait
    """
    if not text_content or len(text_content.strip()) < 10:
        return None

    text_lower = text_content.lower()

    # Score pour chaque type de document
    type_scores = {}

    for doc_type, config in DOCUMENT_TYPES.items():
        score = 0
        keywords = config["keywords"]

        for keyword in keywords:
            # Recherche plus flexible pour le texte extrait
            if keyword.lower() in text_lower:
                score += 2  # Score plus élevé pour correspondance exacte
            # Recherche partielle
            elif any(word in text_lower for word in keyword.lower().split()):
                score += 1

        # Score relatif au nombre de mots-clés
        if keywords:
            type_scores[doc_type] = score / len(keywords)

    # Retourner le type avec le meilleur score si > 0.3 (seuil un peu plus élevé pour le texte)
    if type_scores:
        best_type = max(type_scores.items(), key=lambda x: x[1])
        if best_type[1] > 0.3:
            logger.info(f"Type détecté depuis texte: {best_type[0]} (score: {best_type[1]:.2f})")
            return best_type[0]

    logger.warning("Impossible de déterminer le type depuis le texte PDF")
    return None

async def detect_document_type_from_image(image_b64: str, content_type: str = "image/png") -> Optional[str]:
    """Détecte le type de document depuis une image"""
    try:
        detection_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """Analyse cette image et détermine de quel type de document il s'agit. 
                        
                        Types possibles: facture, bon_commande, bon_livraison, carte_identite_francaise, marche, passeport, permis_conduire, ou autre.
                        
                        Réponds UNIQUEMENT avec le type de document en un seul mot, ou "autre" si tu ne peux pas déterminer.
                        
                        Exemples de réponses valides: facture, bon_commande, carte_identite_francaise, autre"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{content_type};base64,{image_b64}"
                        }
                    }
                ]
            }
        ]

        detection_response = client.chat.complete(
            model="pixtral-12b-2409",
            messages=detection_messages,
            max_tokens=50,
            temperature=0.1
        )

        detected_type_raw = detection_response.choices[0].message.content.strip().lower()
        logger.info(f"Type détecté par l'IA: {detected_type_raw}")

        # Validation du type détecté
        if detected_type_raw in DOCUMENT_TYPES.keys():
            return detected_type_raw
        elif detected_type_raw != "autre":
            # Essayer de mapper avec les types connus
            for known_type in DOCUMENT_TYPES.keys():
                if known_type in detected_type_raw or detected_type_raw in known_type:
                    return known_type

        return None

    except Exception as e:
        logger.error(f"Erreur détection type depuis image: {e}")
        return None

def create_document_analysis_prompt(detected_type: Optional[str], fields_to_extract: List[str]) -> str:
    """
    Crée un prompt dynamique basé sur le type détecté et les champs demandés
    """
    base_prompt = """Tu es un expert en analyse de documents. Analyse cette image et extrait UNIQUEMENT les informations demandées au format JSON.

IMPORTANT:
- Renvoie UNIQUEMENT un JSON valide, rien d'autre
- Si une information n'est pas trouvée dans le document, utilise null
- Ne devine pas les informations manquantes
- Pour les dates, utilise le format YYYY-MM-DD si possible
- Pour les montants, utilise des nombres sans symbole monétaire
- Sois précis et factuel

"""

    if detected_type:
        type_info = {
            "facture": "Ce document semble être une FACTURE. Concentre-toi sur les informations commerciales et financières.",
            "bon_commande": "Ce document semble être un BON DE COMMANDE. Concentre-toi sur les articles commandés et quantités.",
            "bon_livraison": "Ce document semble être un BON DE LIVRAISON. Concentre-toi sur les informations de transport et livraison.",
            "carte_identite_francaise": "Ce document semble être une CARTE D'IDENTITÉ FRANÇAISE. Concentre-toi sur les informations d'état civil.",
            "marche": "Ce document semble être un MARCHÉ/CONTRAT. Concentre-toi sur les parties contractantes et conditions.",
            "passeport": "Ce document semble être un PASSEPORT. Concentre-toi sur les informations d'identité et de voyage.",
            "permis_conduire": "Ce document semble être un PERMIS DE CONDUIRE. Concentre-toi sur les informations de conduite."
        }
        base_prompt += type_info.get(detected_type, f"Ce document semble être de type: {detected_type}.")
        base_prompt += "\n\n"

    base_prompt += f"""Champs à extraire obligatoirement:
{json.dumps(fields_to_extract, ensure_ascii=False, indent=2)}

Exemple de format de réponse attendu:
{{
"""

    # Ajouter un exemple pour chaque champ
    for i, field in enumerate(fields_to_extract):
        base_prompt += f'  "{field}": null'
        if i < len(fields_to_extract) - 1:
            base_prompt += ","
        base_prompt += "\n"

    base_prompt += "}"

    return base_prompt

def create_document_analysis_prompt_for_text(detected_type: Optional[str], fields_to_extract: List[str], text_content: str) -> str:
    """
    Crée un prompt pour analyser le texte extrait d'un PDF
    """
    base_prompt = f"""Tu es un expert en analyse de documents. Analyse le texte suivant extrait d'un document et extrait UNIQUEMENT les informations demandées au format JSON.

TEXTE DU DOCUMENT:
{text_content[:3000]}  # Limiter la taille pour éviter les timeouts

IMPORTANT:
- Renvoie UNIQUEMENT un JSON valide, rien d'autre
- Si une information n'est pas trouvée dans le texte, utilise null
- Ne devine pas les informations manquantes
- Pour les dates, utilise le format YYYY-MM-DD si possible
- Pour les montants, utilise des nombres sans symbole monétaire
- Sois précis et factuel

"""

    if detected_type:
        type_info = {
            "facture": "Ce document semble être une FACTURE. Concentre-toi sur les informations commerciales et financières.",
            "bon_commande": "Ce document semble être un BON DE COMMANDE. Concentre-toi sur les articles commandés et quantités.",
            "bon_livraison": "Ce document semble être un BON DE LIVRAISON. Concentre-toi sur les informations de transport et livraison.",
            "carte_identite_francaise": "Ce document semble être une CARTE D'IDENTITÉ FRANÇAISE. Concentre-toi sur les informations d'état civil.",
            "marche": "Ce document semble être un MARCHÉ/CONTRAT. Concentre-toi sur les parties contractantes et conditions.",
            "passeport": "Ce document semble être un PASSEPORT. Concentre-toi sur les informations d'identité et de voyage.",
            "permis_conduire": "Ce document semble être un PERMIS DE CONDUIRE. Concentre-toi sur les informations de conduite."
        }
        base_prompt += type_info.get(detected_type, f"Ce document semble être de type: {detected_type}.")
        base_prompt += "\n\n"

    base_prompt += f"""Champs à extraire obligatoirement:
{json.dumps(fields_to_extract, ensure_ascii=False, indent=2)}

Exemple de format de réponse attendu:
{{
"""

    for i, field in enumerate(fields_to_extract):
        base_prompt += f'  "{field}": null'
        if i < len(fields_to_extract) - 1:
            base_prompt += ","
        base_prompt += "\n"

    base_prompt += "}"

    return base_prompt

def extract_json_from_response(response: str) -> Dict[str, Any]:
    """
    Extrait le JSON de la réponse du modèle avec plusieurs stratégies
    """
    try:
        response = response.strip()

        # Stratégie 1: JSON direct
        if response.startswith('{') and response.endswith('}'):
            return json.loads(response)

        # Stratégie 2: Chercher le JSON dans la réponse
        start = response.find('{')
        end = response.rfind('}') + 1

        if start != -1 and end > start:
            json_str = response[start:end]
            return json.loads(json_str)

        # Stratégie 3: Chercher des lignes qui ressemblent à du JSON
        lines = response.split('\n')
        json_lines = []
        in_json = False

        for line in lines:
            if '{' in line and not in_json:
                in_json = True
                json_lines.append(line[line.find('{'):])
            elif in_json:
                json_lines.append(line)
                if '}' in line:
                    break

        if json_lines:
            json_str = '\n'.join(json_lines)
            return json.loads(json_str)

    except json.JSONDecodeError as e:
        logger.error(f"Erreur de parsing JSON: {e}")
        logger.error(f"Réponse problématique: {response}")

    return {}

def create_default_structure(fields: List[str]) -> Dict[str, Any]:
    """Crée une structure par défaut avec tous les champs à null"""
    return {field: None for field in fields}

def calculate_confidence(extracted_fields: Dict[str, Any], expected_fields: List[str]) -> float:
    """Calcule un score de confiance basé sur le nombre de champs extraits"""
    if not expected_fields:
        return 0.0

    non_null_fields = sum(1 for field in expected_fields
                          if field in extracted_fields and extracted_fields[field] is not None)
    return round((non_null_fields / len(expected_fields)) * 100, 2)

@app.post("/analyze-document", response_model=DocumentAnalysisResponse)
async def analyze_document_generic(
        file: UploadFile = File(...),
        fields: str = Form(...)
):
    """
    Analyse générique d'un document avec détection automatique du type
    Support des images (PNG, JPG, etc.) et des PDF
    """
    try:
        # Parse des champs demandés
        try:
            fields_dict = json.loads(fields)
            fields_request = DocumentFieldsRequest(**fields_dict)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Format JSON invalide pour les champs")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Erreur dans les champs: {str(e)}")

        # Lecture du fichier
        file_bytes = await file.read()

        # Validation des formats supportés
        is_pdf = file.content_type == "application/pdf" or (file.filename and file.filename.lower().endswith('.pdf'))
        is_image = file.content_type and file.content_type.startswith("image/")

        if not is_pdf and not is_image:
            raise HTTPException(
                status_code=400,
                detail="Formats supportés: images (PNG, JPG, JPEG, etc.) et PDF uniquement"
            )

        # Variables pour stocker les résultats
        detected_type = None
        raw_response = ""
        extracted_data = {}

        if is_pdf:
            logger.info("Traitement d'un fichier PDF")

            # MÉTHODE 1: Extraction de texte pour analyse rapide
            pdf_text = extract_text_from_pdf(file_bytes)

            if pdf_text:
                logger.info(f"Texte extrait du PDF ({len(pdf_text)} caractères)")

                # Détection du type basée sur le texte
                detected_type = detect_document_type_from_text(pdf_text)

                # Si on a du texte, on peut essayer l'analyse textuelle avec l'IA
                text_analysis_prompt = create_document_analysis_prompt_for_text(
                    detected_type, fields_request.fieldsToExtract, pdf_text
                )

                # Appel à l'IA pour analyser le texte extrait
                text_messages = [
                    {
                        "role": "user",
                        "content": text_analysis_prompt
                    }
                ]

                try:
                    text_response = client.chat.complete(
                        model="mistral-large-latest",  # Meilleur pour l'analyse textuelle
                        messages=text_messages,
                        max_tokens=1000,
                        temperature=0.1
                    )

                    raw_response = text_response.choices[0].message.content
                    extracted_data = extract_json_from_response(raw_response)

                    logger.info("Analyse textuelle du PDF réussie")

                except Exception as e:
                    logger.warning(f"Échec analyse textuelle: {e}")
                    extracted_data = {}

            # MÉTHODE 2: Si l'analyse textuelle a échoué, convertir en images
            if not extracted_data or not detected_type:
                logger.info("Conversion du PDF en images pour analyse visuelle")

                pdf_images = pdf_to_images(file_bytes, max_pages=3)

                if pdf_images:
                    # Utiliser la première page pour la détection du type
                    first_page_b64 = pdf_images[0]

                    # Détection du type sur la première page
                    if not detected_type:
                        detected_type = await detect_document_type_from_image(first_page_b64)

                    # Analyse de la première page (ou de toutes si nécessaire)
                    extraction_prompt = create_document_analysis_prompt(detected_type, fields_request.fieldsToExtract)

                    extraction_messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": extraction_prompt
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{first_page_b64}"
                                    }
                                }
                            ]
                        }
                    ]

                    try:
                        extraction_response = client.chat.complete(
                            model="pixtral-12b-2409",
                            messages=extraction_messages,
                            max_tokens=1000,
                            temperature=0.1
                        )

                        raw_response = extraction_response.choices[0].message.content
                        extracted_data = extract_json_from_response(raw_response)

                        logger.info("Analyse visuelle du PDF réussie")

                    except Exception as e:
                        logger.error(f"Échec analyse visuelle PDF: {e}")
                        extracted_data = {}

        else:
            # Traitement des images (méthode originale)
            logger.info("Traitement d'un fichier image")
            b64 = base64.b64encode(file_bytes).decode("utf-8")

            # Détection du type
            detected_type = await detect_document_type_from_image(b64, file.content_type)

            # Extraction des champs
            extraction_prompt = create_document_analysis_prompt(detected_type, fields_request.fieldsToExtract)

            extraction_messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": extraction_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{file.content_type};base64,{b64}"
                            }
                        }
                    ]
                }
            ]

            extraction_response = client.chat.complete(
                model="pixtral-12b-2409",
                messages=extraction_messages,
                max_tokens=1000,
                temperature=0.1
            )

            raw_response = extraction_response.choices[0].message.content
            extracted_data = extract_json_from_response(raw_response)

        # Si échec d'extraction, créer structure par défaut
        if not extracted_data:
            extracted_data = create_default_structure(fields_request.fieldsToExtract)
            extracted_data["extraction_error"] = "Impossible d'extraire les données du document"

        # S'assurer que tous les champs demandés sont présents
        for field in fields_request.fieldsToExtract:
            if field not in extracted_data:
                extracted_data[field] = None

        # Calcul de la confiance
        confidence = calculate_confidence(extracted_data, fields_request.fieldsToExtract)

        return DocumentAnalysisResponse(
            documentType=detected_type,
            extractedFields=extracted_data,
            rawResponse=raw_response,
            confidence=confidence,
            status="success"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse: {str(e)}")
        return DocumentAnalysisResponse(
            documentType=None,
            extractedFields=create_default_structure(fields_request.fieldsToExtract if 'fields_request' in locals() else []),
            rawResponse="",
            confidence=0.0,
            status="error",
            errors=[str(e)]
        )

@app.get("/health")
async def health_check():
    """Point de santé de l'API"""
    return {"status": "healthy", "service": "Generic Document Analysis Service", "supported_formats": ["images", "pdf"]}

@app.get("/supported-formats")
async def get_supported_formats():
    """Retourne les formats de fichiers supportés"""
    return {
        "image_formats": ["image/png", "image/jpeg", "image/jpg", "image/gif", "image/bmp", "image/webp"],
        "document_formats": ["application/pdf"],
        "processing_methods": {
            "images": "Vision AI directe",
            "pdf": "Extraction de texte + conversion en images si nécessaire"
        },
        "limitations": {
            "pdf_max_pages": 5,
            "pdf_processing": "Les PDF sont traités par extraction de texte en priorité, puis conversion en images si nécessaire"
        }
    }

@app.get("/supported-types")
async def get_supported_document_types():
    """Retourne les types de documents supportés"""
    return {
        "supported_types": list(DOCUMENT_TYPES.keys()),
        "description": {
            "facture": "Factures commerciales",
            "bon_commande": "Bons de commande",
            "bon_livraison": "Bons de livraison",
            "carte_identite_francaise": "Cartes d'identité françaises",
            "marche": "Marchés publics et contrats",
            "passeport": "Passeports français",
            "permis_conduire": "Permis de conduire français"
        },
        "note": "Le service peut traiter d'autres types de documents mais retournera 'null' pour documentType si non reconnu"
    }

@app.post("/test-detection")
async def test_document_detection(file: UploadFile = File(...)):
    """Endpoint de test pour la détection de type seulement"""
    try:
        file_bytes = await file.read()

        is_pdf = file.content_type == "application/pdf" or (file.filename and file.filename.lower().endswith('.pdf'))

        if is_pdf:
            # Test avec PDF
            pdf_text = extract_text_from_pdf(file_bytes)
            detected_from_text = detect_document_type_from_text(pdf_text) if pdf_text else None

            # Test avec conversion en image
            pdf_images = pdf_to_images(file_bytes, max_pages=1)
            detected_from_image = None
            if pdf_images:
                detected_from_image = await detect_document_type_from_image(pdf_images[0])

            return {
                "file_type": "pdf",
                "text_extraction": {"success": bool(pdf_text), "length": len(pdf_text) if pdf_text else 0},
                "detected_from_text": detected_from_text,
                "detected_from_image": detected_from_image,
                "final_detected": detected_from_text or detected_from_image
            }
        else:
            # Test avec image
            b64 = base64.b64encode(file_bytes).decode("utf-8")
            detected = await detect_document_type_from_image(b64, file.content_type)

            return {
                "file_type": "image",
                "detected_type": detected,
                "confidence": "high" if detected in DOCUMENT_TYPES else "low"
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test-pdf")
async def test_pdf_processing(file: UploadFile = File(...)):
    """Endpoint de test pour vérifier le traitement PDF"""
    try:
        if not (file.content_type == "application/pdf" or (file.filename and file.filename.lower().endswith('.pdf'))):
            raise HTTPException(status_code=400, detail="Fichier PDF requis")

        file_bytes = await file.read()

        # Test extraction de texte
        pdf_text = extract_text_from_pdf(file_bytes)
        text_length = len(pdf_text) if pdf_text else 0

        # Test conversion en images
        pdf_images = pdf_to_images(file_bytes, max_pages=2)

        # Test détection de type depuis le texte
        detected_type_text = detect_document_type_from_text(pdf_text) if pdf_text else None

        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "file_size": len(file_bytes),
            "text_extraction": {
                "success": bool(pdf_text),
                "text_length": text_length,
                "preview": pdf_text[:200] + "..." if pdf_text and len(pdf_text) > 200 else pdf_text
            },
            "image_conversion": {
                "success": bool(pdf_images),
                "pages_converted": len(pdf_images)
            },
            "type_detection_from_text": detected_type_text,
            "status": "success"
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "status": "error"}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn