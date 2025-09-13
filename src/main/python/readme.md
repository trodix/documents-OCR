# 1. Créer le venv
python3.11 -m venv venv

# 2. Activer l'environnement virtuel (python 3.11)
source venv/bin/activate    # Utiliser bash au lieu de zsh

# 3. Mettre pip à jour
pip install --upgrade pip

# 4. Installer les dépendances
pip install -r requirements.txt

# 4.1. Freeze des versions de dépendances
pip freeze > requirements.txt

# 5. Lancer le service
uvicorn app:app --reload --host 0.0.0.0 --port 8015