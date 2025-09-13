from mistralai import Mistral
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import base64
import json

app = FastAPI()

# Initialisation du client Mistral avec la nouvelle API
client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

@app.post("/analyze-document")
async def extract_invoice_fields(file: UploadFile = File(...)):
    try:
        # Lecture du fichier
        file_bytes = await file.read()
        b64 = base64.b64encode(file_bytes).decode("utf-8")

        # Vérification du type de fichier
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Seules les images sont supportées")

        # Construction du message avec image pour la vision
        messages = [
            {
                "role": "system",
                "content": "Tu es un assistant qui extrait des informations depuis des factures. Analyse l'image et extrait les informations demandées."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """Analyse cette facture et renvoie uniquement un JSON valide avec les champs suivants :
                        - reference_facture
                        - date_facture  
                        - montant_ttc
                        Si un champ est manquant, renvoie null. Assure-toi que le JSON soit valide."""
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

        # Appel à l'API de chat avec vision
        chat_response = client.chat.complete(
            model="pixtral-12b-2409",  # Modèle avec capacités de vision
            messages=messages,
            max_tokens=500,
            temperature=0.1
        )

        # Récupération de la réponse
        llm_text = chat_response.choices[0].message.content

        # Tentative de parsing JSON pour validation
        try:
            parsed_json = json.loads(llm_text)
        except json.JSONDecodeError:
            # Si ce n'est pas du JSON valide, on nettoie la réponse
            # Chercher le JSON dans la réponse
            start = llm_text.find('{')
            end = llm_text.rfind('}') + 1
            if start != -1 and end != 0:
                json_part = llm_text[start:end]
                try:
                    parsed_json = json.loads(json_part)
                    llm_text = json_part
                except json.JSONDecodeError:
                    parsed_json = {
                        "reference_facture": None,
                        "date_facture": None,
                        "montant_ttc": None,
                        "error": "Impossible de parser la réponse en JSON"
                    }

        return JSONResponse(content={
            "extracted_fields": parsed_json if 'parsed_json' in locals() else llm_text,
            "raw_response": llm_text
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement: {str(e)}")

# Alternative avec le modèle mistral-large si vous préférez
@app.post("/analyze-document-alt")
async def extract_invoice_fields_alt(file: UploadFile = File(...)):
    try:
        # Lecture du fichier
        file_bytes = await file.read()
        b64 = base64.b64encode(file_bytes).decode("utf-8")

        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Seules les images sont supportées")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """Tu es un expert en extraction d'informations de factures. 
                        Analyse cette image de facture et extrait les informations suivantes au format JSON :
                        {
                            "reference_facture": "numéro ou référence de la facture",
                            "date_facture": "date de la facture au format YYYY-MM-DD si possible",
                            "montant_ttc": "montant TTC en nombre (sans symbole monétaire)"
                        }
                        Si une information n'est pas trouvée, utilise null."""
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

        # Utilisation de mistral-large avec vision
        chat_response = client.chat.complete(
            model="mistral-large-latest",
            messages=messages,
            max_tokens=300,
            temperature=0
        )

        llm_response = chat_response.choices[0].message.content

        # Extraction du JSON de la réponse
        try:
            # Chercher le JSON dans la réponse
            start = llm_response.find('{')
            end = llm_response.rfind('}') + 1
            if start != -1 and end != 0:
                json_str = llm_response[start:end]
                extracted_data = json.loads(json_str)
            else:
                extracted_data = {
                    "reference_facture": None,
                    "date_facture": None,
                    "montant_ttc": None,
                    "error": "Aucun JSON trouvé dans la réponse"
                }
        except json.JSONDecodeError as e:
            extracted_data = {
                "reference_facture": None,
                "date_facture": None,
                "montant_ttc": None,
                "error": f"Erreur de parsing JSON: {str(e)}"
            }

        return JSONResponse(content={
            "extracted_fields": extracted_data,
            "raw_llm_response": llm_response
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8015)