
from paddleocr import PaddleOCR
import cv2

class OCRProcessor:
    def __init__(self, lang='en', use_angle_cls=True, use_space_char=True, show_log=False, enable_mkldnn=True):
        self.ocr = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang, use_space_char=use_space_char, show_log=show_log, enable_mkldnn=enable_mkldnn)

    def process_image(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        result = self.ocr.ocr(img_path, cls=True)

        ocr_string = ""
        for i in range(len(result[0])):
            ocr_string = ocr_string + result[0][i][1][0] + " "

        return ocr_string

