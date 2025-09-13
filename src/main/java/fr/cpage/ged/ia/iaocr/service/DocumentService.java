package fr.cpage.ged.ia.iaocr.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import org.apache.tika.Tika;
import org.apache.tika.exception.TikaException;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;

@Service
@RequiredArgsConstructor
public class DocumentService {

    private final RestTemplate restTemplate;
    private final ObjectMapper objectMapper;

    /**
     * Envoie un fichier au microservice Python et récupère le document_type et metadata.
     *
     * @param file MultipartFile envoyé depuis le frontend
     * @return JsonNode contenant document_type et metadata
     * @throws IOException en cas de problème de lecture du fichier ou parsing JSON
     */
    public String analyzeDocument(MultipartFile file) throws IOException {
        String url = "http://localhost:8015/analyze-document";

        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", file.getResource()); // nom exact "file"

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

        ResponseEntity<String> response = restTemplate.postForEntity(url, requestEntity, String.class);

        return response.getBody();
    }



    /**
     * Extrait le texte d'une image ou d'un PDF image.
     *
     * @param filePath chemin vers le fichier image ou PDF
     * @return texte extrait
     * @throws IOException
     * @throws TikaException
     */
    public String extractText(String filePath) throws IOException, TikaException {
        File file = new File(filePath);
        Tika tika = new Tika();

        // Tika va détecter le type de fichier et extraire le texte via OCR si nécessaire
        String text = tika.parseToString(file);

        return text;
    }

}
