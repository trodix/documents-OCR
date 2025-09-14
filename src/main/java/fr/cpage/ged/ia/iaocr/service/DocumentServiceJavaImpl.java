package fr.cpage.ged.ia.iaocr.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.apache.pdfbox.Loader;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.rendering.ImageType;
import org.apache.pdfbox.rendering.PDFRenderer;
import org.apache.pdfbox.text.PDFTextStripper;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.content.Media;
import org.springframework.ai.mistralai.MistralAiChatModel;
import org.springframework.ai.mistralai.MistralAiChatOptions;
import org.springframework.ai.mistralai.api.MistralAiApi;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.core.io.Resource;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.util.MimeTypeUtils;
import org.springframework.web.multipart.MultipartFile;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.Serializable;
import java.util.*;

@Slf4j
@Service
@RequiredArgsConstructor
public class DocumentServiceJavaImpl implements DocumentService {

    private final ObjectMapper objectMapper;

    private final MistralAiChatModel mistralAiChatModel;

    @Value("${spring.ai.mistralai.api-key}")
    private String mistralApiKey;

    public static Map<String, List<String>> getDocumentTypesKeywords() {
        Map<String, List<String>> documentTypes = new HashMap<>();

        documentTypes.put("facture", Arrays.asList(
                "facture", "invoice", "devis", "montant", "tva", "ht", "ttc", "fournisseur", "client", "échéance"
        ));

        documentTypes.put("bon_commande", Arrays.asList(
                "commande", "order", "bon de commande", "quantité", "articles", "prix unitaire", "livraison"
        ));

        documentTypes.put("bon_livraison", Arrays.asList(
                "livraison", "delivery", "transporteur", "expédition", "réception", "bon de livraison"
        ));

        documentTypes.put("carte_identite_francaise", Arrays.asList(
                "carte", "identité", "république française", "nationalité française", "né(e)", "sexe"
        ));

        documentTypes.put("marche", Arrays.asList(
                "marché", "contrat", "accord", "convention", "parties contractantes", "objet", "clause"
        ));

        documentTypes.put("passeport", Arrays.asList(
                "passeport", "passport", "république française", "type", "code pays"
        ));

        documentTypes.put("permis_conduire", Arrays.asList(
                "permis", "conduire", "driving", "license", "catégorie", "véhicule"
        ));

        return documentTypes;
    }

    private static final Map<String, String> TYPE_INFO = Map.of(
            "facture", "Ce document semble être une FACTURE. Concentre-toi sur les informations commerciales et financières.",
            "bon_commande", "Ce document semble être un BON DE COMMANDE. Concentre-toi sur les articles commandés et quantités.",
            "bon_livraison", "Ce document semble être un BON DE LIVRAISON. Concentre-toi sur les informations de transport et livraison.",
            "carte_identite_francaise", "Ce document semble être une CARTE D'IDENTITÉ FRANÇAISE. Concentre-toi sur les informations d'état civil.",
            "marche", "Ce document semble être un MARCHÉ/CONTRAT. Concentre-toi sur les parties contractantes et conditions.",
            "passeport", "Ce document semble être un PASSEPORT. Concentre-toi sur les informations d'identité et de voyage.",
            "permis_conduire", "Ce document semble être un PERMIS DE CONDUIRE. Concentre-toi sur les informations de conduite."
    );

    @Override
    public String analyzeDocumentWithFields(MultipartFile file, List<String> fieldsToExtract) throws IOException {
        Map<String, Object> result = analyzeDocumentGeneric(file, fieldsToExtract);
        return objectMapper.writeValueAsString(result);
    }

    @Override
    public String analyzeDocument(MultipartFile file) throws IOException {
        Map<String, Object> result = analyzeDocumentGeneric(file, Collections.emptyList());
        return objectMapper.writeValueAsString(result);
    }

    public Map<String, Object> analyzeDocumentGeneric(MultipartFile file, List<String> fieldsToExtract) throws IOException {

        boolean isPdf = MediaType.APPLICATION_PDF.toString().equals(file.getContentType());
        boolean isImage = file.getContentType() != null && file.getContentType().startsWith("image/");

        if (!isPdf && !isImage) {
            throw new IllegalArgumentException("Type de fichier non supporté. Seuls les PDF et les images sont acceptés.");
        }

        Optional<String> detectedType = Optional.empty();
        Optional<String> rawResponse = Optional.empty();
        Map<String, Serializable> extractedData = new HashMap<>();

        if (isPdf) {
            log.info("Traitement d'un fichier PDF: {}", file.getOriginalFilename());

            // MÉTHODE 1: Extraction de texte pour analyse rapide
            String pdfText = extractTextFromPdf(file.getBytes());

            if (StringUtils.isNotBlank(pdfText)) {
                log.info("Texte extrait du PDF ({} caractères)", pdfText.length());
                // Détection du type basée sur le texte
                detectedType = detectDocumentTypeFromText(pdfText);
                extractedData = analyseTextuelleAvecIA(pdfText, detectedType.orElse("autre"), fieldsToExtract);
            }

            // MÉTHODE 2: Si l'analyse textuelle a échoué, convertir en images
            if (extractedData.isEmpty() || detectedType.isEmpty()) {
                log.info("Conversion du PDF en images pour analyse visuelle");
                extractedData = analyseVisuelleAvecIA(file.getBytes(), detectedType.orElse("autre"), fieldsToExtract);
            }
        } else {
            log.info("Traitement d'un fichier image: {}", file.getOriginalFilename());
            extractedData = analyzeImage(file.getBytes(), file.getContentType(), fieldsToExtract);
        }

        if (extractedData.isEmpty()) {
            throw new IOException("Impossible d'extraire les données du document");
        }

        for (String field : fieldsToExtract) {
            extractedData.putIfAbsent(field, null);
        }

        double confidence = calculateConfidence(extractedData, fieldsToExtract);

        Map<String, Object> response = new HashMap<>();
        response.put("documentType", detectedType.orElse("autre"));
        response.put("extractedFields", extractedData);
        response.put("rawResponse", rawResponse.orElse(""));
        response.put("confidence", confidence);
        response.put("status", "success");
        return response;
    }

    private Map<String, Serializable> analyseTextuelleAvecIA(String pdfText, String detectedType, List<String> fieldsToExtract) {

        String textAnalysisPrompt = createDocumentAnalysisPromptForText(detectedType, fieldsToExtract, pdfText);
        log.info("Prompt pour analyse textuelle:\n{}", textAnalysisPrompt);

        try {

            ChatResponse response = mistralAiChatModel.call(
                    new Prompt(
                            textAnalysisPrompt,
                            MistralAiChatOptions.builder()
                                    .model(MistralAiApi.ChatModel.LARGE.getValue())
                                    .maxTokens(1000)
                                    .temperature(0.1)
                                    .build()
                    ));

            String rawResponse = response.getResult().getOutput().getText();
            log.info("Réponse brute de l'IA:\n{}", rawResponse);
            Map<String, Serializable> extracted_data = extractJsonFromResponse(rawResponse);
            log.info("Analyse textuelle du PDF réussie");
            return extracted_data;
        } catch (Exception e) {
            log.warn("Échec analyse textuelle: {}", e.getMessage());
            return new HashMap<>();
        }

    }

    private Map<String, Serializable> analyseVisuelleAvecIA(byte[] fileBytes, String detectedType, List<String> fieldsToExtract) {
        log.info("Conversion du PDF en images pour analyse visuelle");

        List<Resource> pdfImages = pdfToImages(fileBytes, 3); // max 3 pages

        if (pdfImages.isEmpty()) {
            return Collections.emptyMap();
        }

        Resource firstPage = pdfImages.getFirst();

        // Détection du type sur la première page si non encore détecté
        if (detectedType == null || detectedType.isBlank()) {
            detectedType = detectDocumentTypeFromImage(firstPage);
        }

        String extractionPrompt = createDocumentAnalysisPromptForText(
                detectedType,
                fieldsToExtract,
                ""
        );

        UserMessage extractionMessage = UserMessage.builder()
                .text(extractionPrompt)
                .media(new Media(MimeTypeUtils.IMAGE_PNG, firstPage))
                .build();

        try {

            ChatResponse response = mistralAiChatModel.call(
                    new Prompt(
                            extractionMessage,
                            MistralAiChatOptions.builder()
                                    .model(MistralAiApi.ChatModel.PIXTRAL)
                                    .maxTokens(1000)
                                    .temperature(0.1)
                                    .build()
                    ));

            String rawResponse = response.getResult().getOutput().getText();


            Map<String, Serializable> extractedData = extractJsonFromResponse(rawResponse);

            log.info("Analyse visuelle du PDF réussie");
            return extractedData;

        } catch (Exception e) {
            log.error("Échec analyse visuelle PDF: {}", e.getMessage(), e);
            return Collections.emptyMap();
        }
    }

    private Map<String, Serializable> analyzeImage(byte[] fileBytes, String contentType, List<String> fieldsToExtract) {
        log.info("Traitement d'un fichier image");

        // Détection du type
        Resource image = new ByteArrayResource(fileBytes);
        String detectedType = detectDocumentTypeFromImage(image, contentType);

        // Création du prompt d’extraction
        String extractionPrompt = createDocumentAnalysisPrompt(
                detectedType,
                fieldsToExtract
        );

        // Message pour le modèle (texte + image)
        UserMessage extractionMessage = UserMessage.builder()
                .text(extractionPrompt)
                .media(new Media(MimeTypeUtils.IMAGE_PNG, image))
                .build();

        try {
            ChatResponse response = mistralAiChatModel.call(
                    new Prompt(
                            extractionMessage,
                            MistralAiChatOptions.builder()
                                    .model(MistralAiApi.ChatModel.PIXTRAL)
                                    .maxTokens(1000)
                                    .temperature(0.1)
                                    .build()
                    ));

            String rawResponse = response.getResult().getOutput().getText();
            Map<String, Serializable> extractedData = extractJsonFromResponse(rawResponse);

            log.info("Analyse visuelle de l'image réussie");
            return extractedData;

        } catch (Exception e) {
            log.error("Échec analyse visuelle image: {}", e.getMessage(), e);
            return Map.of();
        }
    }

    private String detectDocumentTypeFromImage(Resource imageBase64) {
        return detectDocumentTypeFromImage(imageBase64, "image/png");
    }

    private String detectDocumentTypeFromImage(Resource imageResource, String contentType) {
        try {
            // Création du prompt pour la détection du type
            String promptText = """
            Analyse cette image et détermine de quel type de document il s'agit.

            Types possibles: facture, bon_commande, bon_livraison, carte_identite_francaise,
            marche, passeport, permis_conduire, ou autre.

            Réponds UNIQUEMENT avec le type de document en un seul mot,
            ou "autre" si tu ne peux pas déterminer.

            Exemples de réponses valides: facture, bon_commande,
            carte_identite_francaise, autre
            """;

            UserMessage detectionMessage = UserMessage.builder()
                    .text(promptText)
                    .media(new Media(MimeTypeUtils.IMAGE_PNG, imageResource))
                    .build();

            ChatResponse response = mistralAiChatModel.call(
                    new Prompt(
                            detectionMessage,
                            MistralAiChatOptions.builder()
                                    .model(MistralAiApi.ChatModel.PIXTRAL)
                                    .maxTokens(50)
                                    .temperature(0.1)
                                    .build()
                    ));

            String detectedRaw = response.getResult().getOutput().getText()
                    .strip()
                    .toLowerCase(Locale.ROOT);

            log.info("Type détecté par l'IA: {}", detectedRaw);

            // Validation stricte
            if (getDocumentTypesKeywords().containsKey(detectedRaw)) {
                return detectedRaw;
            }

            // Vérifier si ce n’est pas "autre"
            if (!"autre".equals(detectedRaw)) {
                // Mapping approximatif (ex: "factures" → "facture")
                for (String knownType : getDocumentTypesKeywords().keySet()) {
                    if (knownType.contains(detectedRaw) || detectedRaw.contains(knownType)) {
                        return knownType;
                    }
                }
            }

            return null;

        } catch (Exception e) {
            log.error("Erreur détection type depuis image: {}", e.getMessage(), e);
            return null;
        }
    }

    private String createDocumentAnalysisPromptForText(String detectedType, List<String> fieldsToExtract, String textContent) {
        // Limiter le texte pour éviter les timeouts (3000 caractères)
        String limitedText = textContent.length() > 3000 ? textContent.substring(0, 3000) : textContent;

        StringBuilder prompt = new StringBuilder();
        prompt.append("Tu es un expert en analyse de documents. Analyse le texte suivant extrait d'un document et extrait UNIQUEMENT les informations demandées au format JSON.\n\n");
        prompt.append("TEXTE DU DOCUMENT:\n").append(limitedText).append("\n\n");
        prompt.append("IMPORTANT:\n")
                .append("- Renvoie UNIQUEMENT un JSON valide, rien d'autre\n")
                .append("- Si une information n'est pas trouvée dans le texte, utilise null\n")
                .append("- Ne devine pas les informations manquantes\n")
                .append("- Pour les dates, utilise le format YYYY-MM-DD si possible\n")
                .append("- Pour les montants, utilise des nombres sans symbole monétaire\n")
                .append("- Sois précis et factuel\n\n");

        if (detectedType != null) {
            prompt.append(TYPE_INFO.getOrDefault(detectedType, "Ce document semble être de type: " + detectedType + "."))
                    .append("\n\n");
        }

        prompt.append("Champs à extraire obligatoirement:\n");
        try {
            prompt.append(objectMapper.writerWithDefaultPrettyPrinter()
                            .writeValueAsString(fieldsToExtract))
                    .append("\n\n");
        } catch (JsonProcessingException e) {
            // fallback simple
            prompt.append(fieldsToExtract.toString()).append("\n\n");
        }

        prompt.append("Exemple de format de réponse attendu:\n{\n");
        for (int i = 0; i < fieldsToExtract.size(); i++) {
            prompt.append("  \"").append(fieldsToExtract.get(i)).append("\": null");
            if (i < fieldsToExtract.size() - 1) {
                prompt.append(",");
            }
            prompt.append("\n");
        }
        prompt.append("}");

        return prompt.toString();
    }

    private String createDocumentAnalysisPrompt(String detectedType, List<String> fieldsToExtract) {
        StringBuilder basePrompt = new StringBuilder();

        basePrompt.append("""
            Tu es un expert en analyse de documents. Analyse cette image et extrait UNIQUEMENT les informations demandées au format JSON.

            IMPORTANT:
            - Renvoie UNIQUEMENT un JSON valide, rien d'autre
            - Si une information n'est pas trouvée dans le document, utilise null
            - Ne devine pas les informations manquantes
            - Pour les dates, utilise le format YYYY-MM-DD si possible
            - Pour les montants, utilise des nombres sans symbole monétaire
            - Sois précis et factuel

            """);

        if (detectedType != null && !detectedType.isBlank()) {
            basePrompt.append(TYPE_INFO.getOrDefault(
                    detectedType,
                    "Ce document semble être de type: " + detectedType + "."
            ));
            basePrompt.append("\n\n");
        }

        basePrompt.append("Champs à extraire obligatoirement:\n");
        basePrompt.append(fieldsToExtract.toString()).append("\n\n");

        basePrompt.append("Exemple de format de réponse attendu:\n{\n");

        for (int i = 0; i < fieldsToExtract.size(); i++) {
            String field = fieldsToExtract.get(i);
            basePrompt.append("  \"").append(field).append("\": null");
            if (i < fieldsToExtract.size() - 1) {
                basePrompt.append(",");
            }
            basePrompt.append("\n");
        }

        basePrompt.append("}");

        return basePrompt.toString();
    }

    public String extractTextFromPdf(byte[] fileBytes) {
        try (PDDocument document = Loader.loadPDF(fileBytes)) {
            PDFTextStripper stripper = new PDFTextStripper();
            String text = stripper.getText(document).trim();
            if (!text.isEmpty()) {
                return text;
            }
        } catch (IOException e) {
            log.warn("Échec extraction PDFBox: {}", e.getMessage());
        }

        return "";
    }

    public List<Resource> pdfToImages(byte[] fileBytes, int maxPages) {
        List<Resource> images = new ArrayList<>();

        try (PDDocument document = Loader.loadPDF(fileBytes)) {
            PDFRenderer pdfRenderer = new PDFRenderer(document);
            int numPages = Math.min(document.getNumberOfPages(), maxPages);

            for (int i = 0; i < numPages; i++) {
                BufferedImage image = pdfRenderer.renderImageWithDPI(i, 144, ImageType.RGB); // 144 DPI pour meilleure qualité

                // Convertir en PNG
                try (ByteArrayOutputStream baos = new ByteArrayOutputStream()) {
                    ImageIO.write(image, "png", baos);
                    Resource img = new ByteArrayResource(baos.toByteArray());
                    images.add(img);
                }
            }

        } catch (IOException e) {
            log.error("Erreur conversion PDF en images: {}", e.getMessage());
        }

        return images;
    }

    public Optional<String> detectDocumentTypeFromText(String textContent) {
        if (textContent == null || textContent.trim().length() < 10) {
            return Optional.empty();
        }

        String textLower = textContent.toLowerCase();
        Map<String, Double> typeScores = new HashMap<>();

        for (Map.Entry<String, List<String>> entry : getDocumentTypesKeywords().entrySet()) {
            String docType = entry.getKey();
            List<String> keywords = entry.getValue();

            int score = 0;
            for (String keyword : keywords) {
                String keywordLower = keyword.toLowerCase();
                if (textLower.contains(keywordLower)) {
                    score += 2; // correspondance exacte
                } else {
                    // correspondance partielle sur chaque mot du mot-clé
                    for (String word : keywordLower.split("\\s+")) {
                        if (textLower.contains(word)) {
                            score += 1;
                        }
                    }
                }
            }

            if (!keywords.isEmpty()) {
                typeScores.put(docType, score / (double) keywords.size());
            }
        }

        if (!typeScores.isEmpty()) {
            Map.Entry<String, Double> bestType = typeScores.entrySet().stream()
                    .max(Map.Entry.comparingByValue())
                    .orElse(null);

            if (bestType != null && bestType.getValue() > 0.3) {
                log.info("Type détecté depuis texte: {} (score: {})", bestType.getKey(), String.format("%.2f", bestType.getValue()));
                return Optional.of(bestType.getKey());
            }
        }

        log.warn("Impossible de déterminer le type depuis le texte PDF");
        return Optional.empty();
    }



    /**
     * Calcule un score de confiance basé sur le nombre de champs extraits
     * @param extractedFields Map des champs extraits
     * @param expectedFields Liste des champs attendus
     * @return score de confiance en pourcentage (0.0 à 100.0)
     */
    public double calculateConfidence(Map<String, Serializable> extractedFields, List<String> expectedFields) {
        if (expectedFields == null || expectedFields.isEmpty()) {
            return 0.0;
        }

        long nonNullFields = expectedFields.stream()
                .filter(field -> extractedFields.containsKey(field) && extractedFields.get(field) != null)
                .count();

        double confidence = ((double) nonNullFields / expectedFields.size()) * 100;
        return Math.round(confidence * 100.0) / 100.0; // arrondi à 2 décimales
    }


    public Map<String, Serializable> extractJsonFromResponse(String response) {
        if (response == null || response.isBlank()) {
            return new HashMap<>();
        }

        response = response.trim();

        try {
            // Stratégie 1: JSON direct
            if (response.startsWith("{") && response.endsWith("}")) {
                return objectMapper.readValue(response, Map.class);
            }

            // Stratégie 2: Chercher le JSON dans la réponse
            int start = response.indexOf("{");
            int end = response.lastIndexOf("}") + 1;

            if (start != -1 && end > start) {
                String jsonStr = response.substring(start, end);
                return objectMapper.readValue(jsonStr, Map.class);
            }

            // Stratégie 3: Chercher des lignes ressemblant à du JSON
            String[] lines = response.split("\\r?\\n");
            StringBuilder jsonBuilder = new StringBuilder();
            boolean inJson = false;

            for (String line : lines) {
                if (line.contains("{") && !inJson) {
                    inJson = true;
                    jsonBuilder.append(line.substring(line.indexOf("{"))).append("\n");
                } else if (inJson) {
                    jsonBuilder.append(line).append("\n");
                    if (line.contains("}")) {
                        break;
                    }
                }
            }

            if (jsonBuilder.length() > 0) {
                String jsonStr = jsonBuilder.toString().trim();
                return objectMapper.readValue(jsonStr, Map.class);
            }

        } catch (JsonProcessingException e) {
            log.error("Erreur de parsing JSON: {}", e.getMessage());
            log.error("Réponse problématique: {}", response);
        }

        return new HashMap<>();
    }

}
