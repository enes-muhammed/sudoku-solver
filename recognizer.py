import sys
import cv2
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

NORMALIZED_SIZE = 64

DIGIT_PADDING = 8


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
# NORMALIZE DIGIT
# ============================================================

def normalize_digit(digit):
    """
    Rakam görüntüsünü 64x64 standart görüntüye dönüştürür.

    Rakamın oranı korunur.
    """

    if digit is None or digit.size == 0:
        return np.full(
            (
                NORMALIZED_SIZE,
                NORMALIZED_SIZE,
            ),
            255,
            dtype=np.uint8,
        )

    h, w = digit.shape

    available_size = (
        NORMALIZED_SIZE
        - DIGIT_PADDING * 2
    )

    scale = min(
        available_size / max(w, 1),
        available_size / max(h, 1),
    )

    new_width = max(
        1,
        round(w * scale),
    )

    new_height = max(
        1,
        round(h * scale),
    )

    resized = cv2.resize(
        digit,
        (
            new_width,
            new_height,
        ),
        interpolation=cv2.INTER_AREA,
    )

    result = np.full(
        (
            NORMALIZED_SIZE,
            NORMALIZED_SIZE,
        ),
        255,
        dtype=np.uint8,
    )

    x = (
        NORMALIZED_SIZE
        - new_width
    ) // 2

    y = (
        NORMALIZED_SIZE
        - new_height
    ) // 2

    result[
        y:y + new_height,
        x:x + new_width,
    ] = resized

    return result


# ============================================================
# RECOGNIZE GRID
# ============================================================

def build_digit_template_bank():
    """
    Çok küçük bir template bankası oluşturur.
    Amaç, çok kolay ve güvenli bir ilk tanıma katmanı kurmaktır.
    Emin değilsek 0 döndürürüz.
    """

    bank = {}

    for value in range(10):
        variants = []

        for scale, thickness in (
            (2.2, 2),
            (2.4, 1),
            (2.0, 3),
        ):
            canvas = np.full(
                (64, 64),
                255,
                dtype=np.uint8,
            )

            cv2.putText(
                canvas,
                str(value),
                (8, 48),
                cv2.FONT_HERSHEY_COMPLEX,
                scale,
                0,
                thickness,
                cv2.LINE_AA,
            )

            variants.append(
                canvas
            )

        bank[value] = variants

    return bank


TEMPLATE_BANK = build_digit_template_bank()


def classify_normalized_digit(digit):
    """
    Normalize edilmiş 64x64 rakam görüntüsünü güvenli şekilde 0..9'a çevirir.

    Ana koşul: yüksek güven varsa sınıflandır.
    Kesin değilse 0 döndürür (boş / yoksay).
    """

    if digit is None or digit.size == 0:
        return 0

    image = np.asarray(
        digit,
        dtype=np.uint8,
    )

    if image.shape != (64, 64):
        image = cv2.resize(
            image,
            (64, 64),
            interpolation=cv2.INTER_AREA,
        )

    binary = np.where(
        image < 200,
        0,
        255,
    ).astype(np.uint8)

    dark_ratio = float(
        np.mean(
            binary == 0
        )
    )

    if dark_ratio < 0.02:
        return 0

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

    # Yalnızca net ve ayrışmış eşleşmeler kabul edilir.
    # Emin değilsek boş bırakılır.
    if best_score < 0.40:
        return 0

    if second_score >= 0 and (best_score - second_score) < 0.02:
        return 0

    return int(best_digit)


def recognize_grid(cell_images):
    """
    Her hücreyi boş / rakam olarak değerlendirir.
    Güvenli rakamlar 1..9, emin değilsek 0 döner.

    Returns:
        recognition
        normalized_digits
        occupied
    """

    recognition = []
    normalized_digits = []
    occupied = []

    for row in range(9):
        recognition_row = []
        normalized_row = []
        occupied_row = []

        for col in range(9):
            cell = cell_images[row][col]
            result = analyze_cell(cell)

            if result["empty"]:
                recognition_row.append(0)
                normalized_row.append(None)
                occupied_row.append(0)
                continue

            occupied_row.append(1)

            normalized = normalize_digit(
                result["digit"]
            )

            predicted = classify_normalized_digit(
                normalized
            )

            recognition_row.append(
                int(predicted)
            )
            normalized_row.append(
                normalized
            )

        recognition.append(
            recognition_row
        )

        normalized_digits.append(
            normalized_row
        )

        occupied.append(
            occupied_row
        )

    return (
        recognition,
        normalized_digits,
        occupied,
    )


def prompt_for_known_filled_cells(occupied, recognition):
    """
    Dolu olduğu bilinen ama tanıyamadığı hücreler için kullanıcıdan değer ister.
    Bu, yanlış OCR yerine kullanıcı katılımını tercih etmek için tasarlandı.
    """

    pending = []

    for row in range(9):
        for col in range(9):
            if occupied[row][col] == 1 and recognition[row][col] == 0:
                pending.append((row, col))

    if not pending:
        return recognition

    if not sys.stdin.isatty():
        print("\nNon-interactive input detected; skipping manual value prompts.")
        return recognition

    print("\nKnown non-empty cells without a confident digit:")

    for row, col in pending:
        while True:
            try:
                answer = input(
                    f"  Cell [{row},{col}] value (1-9, or 0 to skip): "
                ).strip()
            except EOFError:
                print("\n  Input closed; skipping remaining prompts.")
                return recognition

            if answer == "":
                continue

            try:
                value = int(answer)
            except ValueError:
                print("  Please enter a number from 0 to 9.")
                continue

            if value == 0:
                break

            if 1 <= value <= 9:
                recognition[row][col] = value
                break

            print("  Value must be between 1 and 9.")

    return recognition


# ============================================================
# CREATE RECOGNITION RESULT
# ============================================================

def create_rebuilt_with_recognition(rebuilt, recognition, occupied):
    """
    Rebuilt board'un sağ sütununda tanınan hücrelere yarı saydam mavi perde gelir.
    Rakam metni siyah kalır; renk sadece arka plan dokuya uygulanır.
    """

    result = cv2.cvtColor(
        rebuilt,
        cv2.COLOR_GRAY2BGR,
    )

    h, w = rebuilt.shape
    cell_width = w / 9.0
    cell_height = h / 9.0

    overlay = result.copy()

    for row in range(9):
        for col in range(9):
            if recognition[row][col] == 0:
                continue

            x1 = round(col * cell_width)
            x2 = round((col + 1) * cell_width)
            y1 = round(row * cell_height)
            y2 = round((row + 1) * cell_height)

            cv2.rectangle(
                overlay,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                -1,
            )

    result = cv2.addWeighted(
        overlay,
        0.30,
        result,
        0.70,
        0,
    )

    return result


def create_recognition_result(rebuilt, recognition, occupied):
    """
    Renk, sayı metninin kendisinde değil hücre arka planında kullanılır.
    Bu yüzden tüm rakamlar siyah yazılır; arka plan sadece perde gibi renklendirir.
    """

    result = cv2.cvtColor(
        rebuilt,
        cv2.COLOR_GRAY2BGR,
    )

    h, w = rebuilt.shape
    cell_width = w / 9.0
    cell_height = h / 9.0

    overlay = result.copy()

    for row in range(9):
        for col in range(9):
            x1 = round(col * cell_width)
            x2 = round((col + 1) * cell_width)
            y1 = round(row * cell_height)
            y2 = round((row + 1) * cell_height)

            value = recognition[row][col]
            is_occupied = occupied[row][col] == 1

            if is_occupied and value == 0:
                cv2.rectangle(
                    overlay,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 255),
                    -1,
                )
                continue

            if not is_occupied and value != 0:
                cv2.rectangle(
                    overlay,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    -1,
                )
                continue

            if is_occupied and value != 0:
                cv2.rectangle(
                    overlay,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    -1,
                )

    result = cv2.addWeighted(
        overlay,
        0.30,
        result,
        0.70,
        0,
    )

    for row in range(9):
        for col in range(9):
            value = recognition[row][col]
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
                (cx - 11, cy + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.78,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

    return result