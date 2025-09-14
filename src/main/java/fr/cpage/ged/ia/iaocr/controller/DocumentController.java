package fr.cpage.ged.ia.iaocr.controller;

import com.fasterxml.jackson.databind.ObjectMapper;
import fr.cpage.ged.ia.iaocr.service.DocumentService;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;

@RestController
@RequestMapping("/api/documents")
@RequiredArgsConstructor
public class DocumentController {

    @Qualifier("documentServiceJavaImpl")
    private final DocumentService documentService;

    /**
     * Endpoint pour analyser un document (PDF ou image).
     * Le type de document est déduit automatiquement par le microservice Python.
     *
     * @param file le fichier envoyé depuis le frontend
     * @return JSON contenant "document_type" et "metadata"
     */
    @PostMapping("/analyze")
    public ResponseEntity<String> analyzeDocument(@RequestParam("file") MultipartFile file) {
        try {
            String result = documentService.analyzeDocument(file);
            return ResponseEntity.ok(result);
        } catch (Exception e) {
            // Log de l'erreur si nécessaire
            e.printStackTrace();
            return ResponseEntity.status(500).body(e.getMessage());
        }
    }

    /**
     * Endpoint pour analyser un document avec des champs spécifiques
     */
    @PostMapping("/analyze-with-fields")
    public ResponseEntity<String> analyzeDocumentWithFields(
            @RequestParam("file") MultipartFile file,
            @RequestParam("fields") String fieldsJson) {
        try {
            ObjectMapper mapper = new ObjectMapper();
            List<String> fields = mapper.readValue(fieldsJson,
                    mapper.getTypeFactory().constructCollectionType(List.class, String.class));

            String result = documentService.analyzeDocumentWithFields(file, fields);
            return ResponseEntity.ok(result);
        } catch (Exception e) {
            e.printStackTrace();
            return ResponseEntity.status(500).body(e.getMessage());
        }
    }
}
