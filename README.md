# RAG sur des romans français avec LangChain et BLOOMZ-7B1-MT

Ce projet implémente un système RAG (**Retrieval-Augmented Generation**) pour répondre à des questions sur des romans en français.  
Le système est basé sur :

- **LangChain** pour orchestrer le flux de données et les chaînes
- **FAISS** pour l'indexation vectorielle et la recherche rapide de documents
- **HuggingFace Embeddings** pour convertir le texte en vecteurs sémantiques
- **bigscience/bloomz-7b1-mt** comme modèle de génération multilingue

## Objectif

Permettre à un modèle de répondre à des questions précises à partir du contenu extrait de plusieurs livres, en combinant récupération d'information et génération de texte.

## Pipeline

1. **Extraction du texte**  
   Extraction du contenu page par page à partir de fichiers PDF, en conservant les métadonnées (titre du livre, numéro de page).

2. **Découpage en chunks**  
   Les textes sont découpés en segments de taille fixe (700 à 1000 tokens) avec un chevauchement pour préserver le contexte entre les morceaux.

3. **Indexation FAISS**  
   Les chunks sont transformés en embeddings à l'aide de `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, puis indexés dans FAISS pour une recherche rapide.

4. **Récupération de documents**  
   Lorsqu'une question est posée, FAISS récupère les k documents les plus pertinents.

5. **Génération de réponse**  
   Le modèle `bigscience/bloomz-7b1-mt` reçoit les documents pertinents et génère une réponse, en étant incité à répondre uniquement à partir du contexte fourni.


## Technologies utilisées

- Python 3.10
- LangChain
- FAISS
- Hugging Face Transformers
- Hugging Face Datasets
- PyMuPDF (pour extraire les textes PDF)

## Modèle utilisé

**bigscience/bloomz-7b1-mt**

- Taille : 7.1 milliards de paramètres
- Spécialisé pour le multilingue
- Pré-entraîné sur plus de 46 langues, dont le français

## Limitations

- BLOOMZ-7B1-MT est plus limité en compréhension fine que des modèles plus récents comme GPT-4o ou Mistral-7B.

## Possibilités d'amélioration

- Intégrer des embeddings plus puissants.
- Passer à un modèle plus récent comme `Mistral-7B-Instruct` pour améliorer la qualité de réponse en français.



