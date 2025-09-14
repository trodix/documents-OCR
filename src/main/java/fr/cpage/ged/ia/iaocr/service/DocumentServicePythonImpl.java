package fr.cpage.ged.ia.iaocr.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

@Service
@RequiredArgsConstructor
public class DocumentServicePythonImpl implements DocumentService {

    private final RestTemplate restTemplate;
    private final ObjectMapper objectMapper;

    @Value("${python.service.url:http://localhost:8015}")
    private String pythonServiceUrl;

    /**
     * Analyse générique d'un document avec détection automatique du type
     * et extraction des champs demandés
     */
    public String analyzeDocument(MultipartFile file) throws IOException {
        // Définition des champs standards que nous voulons extraire
        List<String> standardFields = Arrays.asList(
                // Champs communs factures
                "numero_facture", "date_facture", "montant_ht", "montant_ttc", "tva",
                "fournisseur", "client", "date_echeance",

                // Champs communs bons de commande
                "numero_commande", "date_commande", "articles", "quantites", "prix_unitaires",

                // Champs communs bons de livraison
                "numero_livraison", "date_livraison", "transporteur", "destinataire",
                "articles_livres", "etat_livraison",

                // Champs communs marchés/contrats
                "numero_marche", "date_signature", "parties_contractantes", "objet_marche",
                "montant_marche", "duree", "date_debut", "date_fin",

                // Champs communs cartes d'identité
                "nom", "prenoms", "date_naissance", "lieu_naissance", "sexe",
                "nationalite", "adresse", "date_delivrance", "numero_document",

                // Autres champs génériques
                "titre", "reference", "date_document", "montant_total", "description"
        );

        return analyzeDocumentWithFields(file, standardFields);
    }

    /**
     * Analyse d'un document avec des champs spécifiques
     */
    public String analyzeDocumentWithFields(MultipartFile file, List<String> fieldsToExtract) throws IOException {
        String url = pythonServiceUrl + "/analyze-document";

        DocumentAnalysisRequest request = new DocumentAnalysisRequest(fieldsToExtract);

        // Création du body multipart
        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", file.getResource());

        // Ajout des métadonnées sous forme JSON
        String metadataJson = objectMapper.writeValueAsString(request);
        body.add("fields", metadataJson);

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

        try {
            ResponseEntity<String> response = restTemplate.postForEntity(url, requestEntity, String.class);
            return response.getBody();
        } catch (Exception e) {
            throw new RuntimeException("Erreur lors de l'appel au service Python: " + e.getMessage(), e);
        }
    }

    public class DocumentAnalysisRequest {
        private List<String> fieldsToExtract;
        private String language;
        private Map<String, Object> additionalOptions;

        // Constructeurs
        public DocumentAnalysisRequest() {}

        public DocumentAnalysisRequest(List<String> fieldsToExtract) {
            this.fieldsToExtract = fieldsToExtract;
            this.language = "fr";
        }

        // Getters et Setters
        public List<String> getFieldsToExtract() { return fieldsToExtract; }
        public void setFieldsToExtract(List<String> fieldsToExtract) { this.fieldsToExtract = fieldsToExtract; }

        public String getLanguage() { return language; }
        public void setLanguage(String language) { this.language = language; }

        public Map<String, Object> getAdditionalOptions() { return additionalOptions; }
        public void setAdditionalOptions(Map<String, Object> additionalOptions) { this.additionalOptions = additionalOptions; }
    }

    // DTO pour la réponse
    public class DocumentAnalysisResponse {
        private String documentType;  // null si non déterminé
        private Map<String, Object> extractedFields;
        private String rawResponse;
        private double confidence;
        private String status;
        private List<String> errors;

        // Constructeurs, getters et setters
        public DocumentAnalysisResponse() {}

        public String getDocumentType() { return documentType; }
        public void setDocumentType(String documentType) { this.documentType = documentType; }

        public Map<String, Object> getExtractedFields() { return extractedFields; }
        public void setExtractedFields(Map<String, Object> extractedFields) { this.extractedFields = extractedFields; }

        public String getRawResponse() { return rawResponse; }
        public void setRawResponse(String rawResponse) { this.rawResponse = rawResponse; }

        public double getConfidence() { return confidence; }
        public void setConfidence(double confidence) { this.confidence = confidence; }

        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }

        public List<String> getErrors() { return errors; }
        public void setErrors(List<String> errors) { this.errors = errors; }
    }

}
