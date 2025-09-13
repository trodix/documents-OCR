package fr.cpage.ged.ia.iaocr;

import fr.cpage.ged.ia.iaocr.service.DocumentService;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest
class IaOcrApplicationTests {

    @Autowired
    private DocumentService documentService;

    @Test
    void contextLoads() {
    }

}
