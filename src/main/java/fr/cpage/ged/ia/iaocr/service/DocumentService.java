package fr.cpage.ged.ia.iaocr.service;

import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.List;


public interface DocumentService {

    /**
     * Analyse générique d'un document avec détection automatique du type
     * et extraction des champs demandés
     */
    String analyzeDocument(MultipartFile file) throws IOException;

    /**
     * Analyse d'un document avec des champs spécifiques
     */
    String analyzeDocumentWithFields(MultipartFile file, List<String> fieldsToExtract) throws IOException;

}
