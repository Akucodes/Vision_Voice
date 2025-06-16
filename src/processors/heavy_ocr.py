from paddleocr import PaddleOCR
from typing import Optional

class HeavyOCRProcessor:
    def __init__(self, use_angle: bool = True, lang: str = 'en'):
        """
        Parameters:
        -----------
        use_angle : bool
            Whether to enable text direction detection.
        lang : str
            Language parameter for PaddleOCR.
        """
        self.use_angle = use_angle
        self.lang = lang
        self.ocr: Optional[PaddleOCR] = None
        self.load_model()

    def load_model(self):
        """Loads the PaddleOCR model into memory. Call this before process()."""
        self.ocr = PaddleOCR(use_angle=self.use_angle, lang=self.lang)

    def process(self, img_path: str) -> str:
        """
        Runs OCR on the given image and returns the concatenated text.

        Parameters:
        -----------
        img_path : str
            Path to the image file.

        Returns:
        --------
        str
            All detected text joined into one string.
        """
        if self.ocr is None:
            raise RuntimeError("Model not loaded. Please call load_model() first.")

        # run OCR
        result = self.ocr.ocr(img_path, cls=True)

        # `result` is a list of pages; each page is a list of lines
        # flatten all lines across all pages
        lines = [line for page in result for line in page]

        # each `line` is [box, (text, confidence)]
        texts = [line[1][0] for line in lines]

        # join with spaces (you can customize delimiter)
        return ' '.join(texts)


# if __name__ == '__main__':
#     # example usage
#     processor = OCRProcessor(use_angle=True, lang='en')
#     img_path = '/home/akanksha/Downloads/test_img.jpg'
#     text = processor.process(img_path)
#     print(text)
