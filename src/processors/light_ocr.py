from typing import Optional
from PIL import Image
import pytesseract

class LightOCRProcessor:
    def __init__(
        self,
        tesseract_cmd: Optional[str] = None,
        lang: str = 'eng',
        config: str = ''
    ):
        """
        Parameters:
        -----------
        tesseract_cmd : Optional[str]
            Full path to the tesseract executable (if it's not on your PATH).
        lang : str
            Language code(s) to use (e.g. 'eng', 'eng+fra', etc.).
        config : str
            Additional tesseract command-line flags (e.g. '--psm 6').
        """
        self.tesseract_cmd = tesseract_cmd
        self.lang = lang
        self.config = config
        self._loaded = False
        self.load_model()

    def load_model(self):
        """
        Configures pytesseract (e.g. sets tesseract_cmd) and marks the model ready.
        Call this once before you run `process()`.
        """
        if self.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
        # you could add sanity checks here (e.g., call pytesseract.get_tesseract_version())
        self._loaded = True

    def process(self, img_path: str) -> str:
        """
        Runs OCR on the image at `img_path` and returns all detected text.

        Parameters:
        -----------
        img_path : str
            Path to the image file.

        Returns:
        --------
        str
            The raw text output from Tesseract.
        """
        if not self._loaded:
            raise RuntimeError("TesseractOCRProcessor not loaded. Call load_model() first.")

        # open image
        img = Image.open(img_path)

        # run tesseract
        text = pytesseract.image_to_string(img, lang=self.lang, config=self.config)
        return text.strip()


# if __name__ == '__main__':
#     # Example usage
#     processor = TesseractOCRProcessor(
#         tesseract_cmd='/usr/bin/tesseract',  # or None if tesseract is on your PATH
#         lang='eng',
#         config='--psm 6'
#     )
#     img_path = '/home/akanksha/Downloads/test_img.jpg'
#     extracted_text = processor.process(img_path)
#     print(extracted_text)
