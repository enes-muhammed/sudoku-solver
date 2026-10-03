import cv2
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

NORMALIZED_SIZE = 64

DIGIT_PADDING = 6

MIN_CONFIDENCE = 0.55

FONT_CONFIGS = [
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
    cv2.FONT_HERSHEY_COMPLEX_SMALL,
    cv2.FONT_HERSHEY_PLAIN,
]

TEMPLATE_SCALES = (1.2, 1.6, 2.0, 2.4)

TEMPLATE_THICKNESSES = (1, 2, 3)


# ============================================================
# ANALYZE CELL
# ============================================================

def analyze_cell(cell):
    """
    Temizlenmiş bir Sudoku hücresini analiz eder.

    Returns:
        {
            "empty": bool,
            "bbox": (x, y, w, h) | None,
            "area": float,
            "digit": np.ndarray | None,
        }
    """

    if cell is None or cell.size == 0:
        return {
            "empty": True,
            "bbox": None,
            "area": 0,
            "digit": None,
        }

    dark = cv2.inRange(
        cell,
        0,
        127,
    )

    contours, _ = cv2.findContours(
        dark,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return {
            "empty": True,
            "bbox": None,
            "area": 0,
            "digit": None,
        }

    contour = max(
        contours,
        key=cv2.contourArea,
    )

    area = cv2.contourArea(
        contour
    )

    if area <= 0:
        return {
            "empty": True,
            "bbox": None,
            "area": 0,
            "digit": None,
        }

    x, y, w, h = cv2.boundingRect(
        contour
    )

    digit = cell[
        y:y + h,
        x:x + w,
    ]

    return {
        "empty": False,
        "bbox": (x, y, w, h),
        "area": area,
        "digit": digit,
    }


# ============================================================
# NORMALIZE DIGIT (bbox based)
# ============================================================

def _bbox_normalize(gray):
    """
    Rakamın bounding kutusunu kırpar, en-boy oranını
    koruyarak 64x64'lük tuvalin ortasına yerleştirir.
    """

    if gray is None or gray.size == 0:
        return np.full(
            (NORMALIZED_SIZE, NORMALIZED_SIZE),
            255,
            dtype=np.uint8,
        )

    binary = np.where(
        gray < 160,
        0,
        255,
    ).astype(np.uint8)

    ys, xs = np.where(binary == 0)

    if len(xs) == 0:
        return np.full(
            (NORMALIZED_SIZE, NORMALIZED_SIZE),
            255,
            dtype=np.uint8,
        )

    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1

    crop = binary[y1:y2, x1:x2]

    h, w = crop.shape

    available = (
        NORMALIZED_SIZE
        - DIGIT_PADDING * 2
    )

    scale = min(
        available / max(w, 1),
        available / max(h, 1),
    )

    new_width = max(1, round(w * scale))
    new_height = max(1, round(h * scale))

    resized = cv2.resize(
        crop,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )

    result = np.full(
        (NORMALIZED_SIZE, NORMALIZED_SIZE),
        255,
        dtype=np.uint8,
    )

    y = (NORMALIZED_SIZE - new_height) // 2
    x = (NORMALIZED_SIZE - new_width) // 2

    result[y:y + new_height, x:x + new_width] = resized

    return result


def normalize_digit(digit):
    """Rakam görüntüsünü 64x64 standart görüntüye dönüştürür."""

    return _bbox_normalize(digit)


# ============================================================
# TEMPLATE BANK
# ============================================================

def build_digit_template_bank():
    """
    Birden fazla Hershey fontu, farklı ölçek ve
    kalınlıklar içeren zengin bir template bankası kurar.

    Her şablon, hücre normalize edilirken kullanılan
    aynı bbox-temelli normalize ile hizalanır.
    Eşleşmezse "emin değilim" denir ve 0 (boş) döneriz.
    """

    bank = {}

    for value in range(10):
        variants = []

        for font in FONT_CONFIGS:
            for scale in TEMPLATE_SCALES:
                for thickness in TEMPLATE_THICKNESSES:
                    canvas = np.full(
                        (96, 96),
                        255,
                        dtype=np.uint8,
                    )

                    cv2.putText(
                        canvas,
                        str(value),
                        (8, 80),
                        font,
                        scale,
                        0,
                        thickness,
                        cv2.LINE_AA,
                    )

                    variants.append(
                        _bbox_normalize(canvas)
                    )

        bank[value] = variants

    return bank


TEMPLATE_BANK = build_digit_template_bank()


# ============================================================
# CLASSIFY
# ============================================================

def classify_normalized_digit(digit):
    """
    Normalize edilmiş 64x64 rakam görüntüsünü sınıflandırır.

    Returns:
        (digit_value, confidence)
    """

    if digit is None or digit.size == 0:
        return 0, 0.0

    image = np.asarray(
        digit,
        dtype=np.uint8,
    )

    if image.shape != (
        NORMALIZED_SIZE,
        NORMALIZED_SIZE,
    ):
        image = cv2.resize(
            image,
            (NORMALIZED_SIZE, NORMALIZED_SIZE),
            interpolation=cv2.INTER_AREA,
        )

    binary = np.where(
        image < 200,
        0,
        255,
    ).astype(np.uint8)

    dark_ratio = float(
        np.mean(binary == 0)
    )

    if dark_ratio < 0.02:
        return 0, 0.0

    best_digit = 0
    best_score = -1.0
    second_score = -1.0

    for value in range(10):
        for template in TEMPLATE_BANK[value]:
            score_map = cv2.matchTemplate(
                binary,
                template,
                cv2.TM_CCOEFF_NORMED,
            )

            score = float(
                np.max(score_map)
            )

            if score > best_score:
                second_score = best_score
                best_score = score
                best_digit = value
            elif score > second_score:
                second_score = score

    if best_score < MIN_CONFIDENCE:
        return 0, best_score

    return int(best_digit), best_score


# ============================================================
# RECOGNIZE GRID
# ============================================================

def recognize_grid(cell_images):
    """
    Her hücreyi boş / rakam olarak değerlendirir.

    Returns:
        recognition        9x9 int (0 = boş/emin değil)
        normalized_digits  9x9 np.ndarray | None
        occupied           9x9 int (1 = hücrede bir şey var)
        confidences        9x9 float
    """

    recognition = []
    normalized_digits = []
    occupied = []
    confidences = []

    for row in range(9):
        recognition_row = []
        normalized_row = []
        occupied_row = []
        confidence_row = []

        for col in range(9):
            cell = cell_images[row][col]
            result = analyze_cell(cell)

            if result["empty"]:
                recognition_row.append(0)
                normalized_row.append(None)
                occupied_row.append(0)
                confidence_row.append(0.0)
                continue

            occupied_row.append(1)

            normalized = normalize_digit(
                result["digit"]
            )

            predicted, confidence = (
                classify_normalized_digit(
                    normalized
                )
            )

            recognition_row.append(
                int(predicted)
            )
            normalized_row.append(
                normalized
            )
            confidence_row.append(
                float(confidence)
            )

        recognition.append(recognition_row)
        normalized_digits.append(normalized_row)
        occupied.append(occupied_row)
        confidences.append(confidence_row)

    return (
        recognition,
        normalized_digits,
        occupied,
        confidences,
    )


# ============================================================
# OVERLAYS
# ============================================================

def create_recognition_overlay(
    warped,
    recognition,
    occupied,
    confidences,
    manual_values=None,
):
    """
    Warped görüntü üzerine yarı saydam güven renkleri basar.

      - yüksek güven   -> yeşil
      - düşük güven    -> kırmızı
      - dolu ama okunamayan -> turuncu (sayı yok)
      - manuel giriş   -> mavi
    """

    result = warped.copy()

    h, w = warped.shape[:2]
    cell_width = w / 9.0
    cell_height = h / 9.0

    overlay = result.copy()

    for row in range(9):
        for col in range(9):
            x1 = round(col * cell_width)
            x2 = round((col + 1) * cell_width)
            y1 = round(row * cell_height)
            y2 = round((row + 1) * cell_height)

            manual = (
                manual_values[row][col]
                if manual_values is not None
                else 0
            )

            value = recognition[row][col]
            is_occupied = occupied[row][col] == 1
            confidence = confidences[row][col]

            if manual != 0:
                color = (230, 130, 0)
            elif value != 0 and confidence >= MIN_CONFIDENCE:
                color = (60, 200, 90)
            elif value != 0:
                color = (60, 60, 220)
            elif is_occupied:
                color = (40, 165, 245)
            else:
                continue

            cv2.rectangle(
                overlay,
                (x1, y1),
                (x2, y2),
                color,
                -1,
            )

    result = cv2.addWeighted(
        overlay,
        0.28,
        result,
        0.72,
        0,
    )

    for row in range(9):
        for col in range(9):
            manual = (
                manual_values[row][col]
                if manual_values is not None
                else 0
            )

            value = (
                manual
                if manual != 0
                else recognition[row][col]
            )

            if value == 0:
                continue

            x1 = round(col * cell_width)
            x2 = round((col + 1) * cell_width)
            y1 = round(row * cell_height)
            y2 = round((row + 1) * cell_height)

            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)

            cv2.putText(
                result,
                str(value),
                (cx - 16, cy + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.4,
                (20, 20, 20),
                3,
                cv2.LINE_AA,
            )

    return result
