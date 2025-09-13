package fr.cpage.ged.ia.iaocr.controller;

import fr.cpage.ged.ia.iaocr.service.DocumentService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/api/documents")
@RequiredArgsConstructor
public class DocumentController {

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
}
