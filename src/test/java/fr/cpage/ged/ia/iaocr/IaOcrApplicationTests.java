package fr.cpage.ged.ia.iaocr;

import fr.cpage.ged.ia.iaocr.service.DocumentService;
import org.apache.tika.exception.TikaException;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.util.ResourceUtils;

import java.io.IOException;

@SpringBootTest
class IaOcrApplicationTests {

    @Autowired
    private DocumentService documentService;

    @Test
    void contextLoads() {
    }

    @Test
    void testTextExtractor() throws IOException, TikaException {
        String filePath = ResourceUtils.getFile("classpath:ocr/facture.png").getAbsolutePath(); // ou PDF image
        String extractedText = documentService.extractText(filePath);
        System.out.println("=== TEXTE EXTRAIT ===");
        System.out.println(extractedText);
    }

}
