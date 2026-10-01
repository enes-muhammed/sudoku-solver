import cv2
import numpy as np


# ============================================================
# CONFIG
# ============================================================

GRID_SIZE = 9
OUTPUT_SIZE = 900

MIN_SQUARENESS = 0.55

DEBUG_FONT_SCALE = 0.45

LINE_SEARCH_RATIO = 0.25

MIN_LINE_DISTANCE_RATIO = 0.45


# ============================================================
# BASIC GEOMETRY
# ============================================================

def order_points(points):
    points = np.array(
        points,
        dtype=np.float32
    )

    result = np.zeros(
        (4, 2),
        dtype=np.float32
    )

    s = points.sum(axis=1)

    d = np.diff(
        points,
        axis=1
    ).reshape(-1)

    result[0] = points[np.argmin(s)]
    result[2] = points[np.argmax(s)]

    result[1] = points[np.argmin(d)]
    result[3] = points[np.argmax(d)]

    return result


def four_point_transform(
    image,
    points,
    output_size
):
    rect = order_points(points)

    dst = np.array([
        [0, 0],
        [output_size - 1, 0],
        [output_size - 1, output_size - 1],
        [0, output_size - 1]
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(
        rect,
        dst
    )

    warped = cv2.warpPerspective(
        image,
        matrix,
        (
            output_size,
            output_size
        )
    )

    return warped


# ============================================================
# BOARD DETECTION
# ============================================================

def preprocess_for_detection(image):
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    threshold = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3)
    )

    threshold = cv2.morphologyEx(
        threshold,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    return threshold


def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b

    denominator = (
        np.linalg.norm(ba)
        *
        np.linalg.norm(bc)
    )

    if denominator == 0:
        return 0

    cosine = np.dot(
        ba,
        bc
    ) / denominator

    cosine = np.clip(
        cosine,
        -1.0,
        1.0
    )

    return np.degrees(
        np.arccos(cosine)
    )


def calculate_squareness(quad):
    quad = order_points(quad)

    tl, tr, br, bl = quad

    angles = [
        calculate_angle(
            bl,
            tl,
            tr
        ),

        calculate_angle(
            tl,
            tr,
            br
        ),

        calculate_angle(
            tr,
            br,
            bl
        ),

        calculate_angle(
            br,
            bl,
            tl
        ),
    ]

    score = 1.0 - (
        np.mean(
            np.abs(
                np.array(angles) - 90
            )
        )
        /
        90
    )

    return max(
        0.0,
        score
    )


def contour_to_quad(contour):
    perimeter = cv2.arcLength(
        contour,
        True
    )

    for epsilon_ratio in [
        0.01,
        0.015,
        0.02,
        0.025,
        0.03,
        0.04
    ]:
        epsilon = (
            epsilon_ratio
            *
            perimeter
        )

        approx = cv2.approxPolyDP(
            contour,
            epsilon,
            True
        )

        if len(approx) == 4:
            return (
                approx
                .reshape(4, 2)
                .astype(np.float32)
            )

    return None


def find_sudoku_contour(image):
    binary = preprocess_for_detection(
        image
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    image_area = (
        image.shape[0]
        *
        image.shape[1]
    )

    candidates = []

    for contour in contours:
        area = cv2.contourArea(
            contour
        )

        if area < image_area * 0.08:
            continue

        if area > image_area * 0.98:
            continue

        quad = contour_to_quad(
            contour
        )

        if quad is None:
            continue

        squareness = calculate_squareness(
            quad
        )

        if squareness < MIN_SQUARENESS:
            continue

        perimeter = cv2.arcLength(
            contour,
            True
        )

        candidates.append(
            (
                area,
                squareness,
                perimeter,
                quad
            )
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item:
            item[0]
            *
            (
                0.5
                +
                0.5 * item[1]
            ),
        reverse=True
    )

    return candidates[0][3]


# ============================================================
# IMAGE CLEANING
# ============================================================

def clean_warped_image(warped):
    gray = cv2.cvtColor(
        warped,
        cv2.COLOR_BGR2GRAY
    )

    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    background = cv2.GaussianBlur(
        blur,
        (0, 0),
        25
    )

    normalized = cv2.divide(
        blur,
        background,
        scale=255
    )

    binary = cv2.adaptiveThreshold(
        normalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        8
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (2, 2)
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel
    )

    return binary


# ============================================================
# GRID DETECTION
# ============================================================

def prepare_grid_binary(binary):
    dark = cv2.bitwise_not(
        binary
    )

    height, width = binary.shape

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            max(
                15,
                width // 18
            ),
            1
        )
    )

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            1,
            max(
                15,
                height // 18
            )
        )
    )

    horizontal_lines = cv2.morphologyEx(
        dark,
        cv2.MORPH_OPEN,
        horizontal_kernel
    )

    vertical_lines = cv2.morphologyEx(
        dark,
        cv2.MORPH_OPEN,
        vertical_kernel
    )

    return (
        horizontal_lines,
        vertical_lines
    )


def calculate_projection(binary, axis):
    if axis == 0:
        projection = np.sum(
            binary > 0,
            axis=0
        )
    else:
        projection = np.sum(
            binary > 0,
            axis=1
        )

    return projection.astype(
        np.float32
    )


def smooth_projection(
    projection,
    kernel_size=9
):
    kernel_size = max(
        3,
        int(kernel_size)
    )

    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = (
        np.ones(
            kernel_size,
            dtype=np.float32
        )
        /
        kernel_size
    )

    return np.convolve(
        projection,
        kernel,
        mode="same"
    )


# ============================================================
# INTERNAL LINE SEARCH
# ============================================================

def local_line_score(
    morphology_projection,
    raw_projection,
    position,
    search_radius
):
    left = max(
        0,
        position - search_radius
    )

    right = min(
        len(morphology_projection),
        position + search_radius + 1
    )

    morph = morphology_projection[
        left:right
    ]

    raw = raw_projection[
        left:right
    ]

    if len(morph) == 0:
        return position

    morph_max = np.max(morph)

    best_score = -1
    best_position = position

    for p in range(
        left,
        right
    ):
        morph_value = (
            morphology_projection[p]
        )

        raw_value = (
            raw_projection[p]
        )

        distance = abs(
            p - position
        )

        distance_penalty = (
            distance
            /
            max(
                search_radius,
                1
            )
        )

        score = (
            morph_value * 2.5
            +
            raw_value * 0.8
            -
            morph_max
            * 0.15
            * distance_penalty
        )

        if score > best_score:
            best_score = score
            best_position = p

    return best_position


def find_internal_grid_lines(
    morphology_projection,
    raw_projection,
    length,
    grid_size
):
    internal_lines = []

    cell_size = (
        length
        /
        grid_size
    )

    search_radius = int(
        cell_size
        *
        LINE_SEARCH_RATIO
    )

    search_radius = max(
        5,
        min(
            search_radius,
            int(
                cell_size
                *
                0.40
            )
        )
    )

    for i in range(
        1,
        grid_size
    ):
        expected = int(
            round(
                i
                *
                cell_size
            )
        )

        position = local_line_score(
            morphology_projection,
            raw_projection,
            expected,
            search_radius
        )

        internal_lines.append(
            position
        )

    return internal_lines


def enforce_line_spacing(
    lines,
    length,
    grid_size
):
    if grid_size <= 1:
        return [
            0,
            length - 1
        ]

    cell_size = (
        length
        /
        grid_size
    )

    min_distance = int(
        cell_size
        *
        MIN_LINE_DISTANCE_RATIO
    )

    result = [0]

    for position in lines[1:-1]:
        position = int(
            position
        )

        previous = result[-1]

        position = max(
            position,
            previous
            +
            min_distance
        )

        remaining_internal = (
            grid_size
            -
            len(result)
            -
            1
        )

        max_allowed = (
            length - 1
            -
            remaining_internal
            *
            min_distance
        )

        position = min(
            position,
            max_allowed
        )

        result.append(
            position
        )

    result.append(
        length - 1
    )

    return result


def detect_grid_lines(binary):
    height, width = binary.shape

    (
        horizontal_lines,
        vertical_lines
    ) = prepare_grid_binary(
        binary
    )

    # --------------------------------------------------------
    # VERTICAL
    # --------------------------------------------------------

    vertical_morph = calculate_projection(
        vertical_lines,
        axis=0
    )

    vertical_raw = calculate_projection(
        cv2.bitwise_not(binary),
        axis=0
    )

    vertical_morph = smooth_projection(
        vertical_morph,
        5
    )

    vertical_raw = smooth_projection(
        vertical_raw,
        7
    )

    vertical_internal = (
        find_internal_grid_lines(
            vertical_morph,
            vertical_raw,
            width,
            GRID_SIZE
        )
    )

    vertical = (
        [0]
        +
        vertical_internal
        +
        [width - 1]
    )

    vertical = enforce_line_spacing(
        vertical,
        width,
        GRID_SIZE
    )

    # --------------------------------------------------------
    # HORIZONTAL
    # --------------------------------------------------------

    horizontal_morph = calculate_projection(
        horizontal_lines,
        axis=1
    )

    horizontal_raw = calculate_projection(
        cv2.bitwise_not(binary),
        axis=1
    )

    horizontal_morph = smooth_projection(
        horizontal_morph,
        5
    )

    horizontal_raw = smooth_projection(
        horizontal_raw,
        7
    )

    horizontal_internal = (
        find_internal_grid_lines(
            horizontal_morph,
            horizontal_raw,
            height,
            GRID_SIZE
        )
    )

    horizontal = (
        [0]
        +
        horizontal_internal
        +
        [height - 1]
    )

    horizontal = enforce_line_spacing(
        horizontal,
        height,
        GRID_SIZE
    )

    return (
        vertical,
        horizontal
    )


# ============================================================
# LINE REFINEMENT
# ============================================================

def refine_line_positions(
    binary,
    positions,
    axis
):
    height, width = binary.shape

    if axis == "vertical":
        length = width
    else:
        length = height

    refined = list(
        positions
    )

    refined[0] = 0
    refined[-1] = length - 1

    dark = cv2.bitwise_not(
        binary
    )

    if axis == "vertical":
        projection = np.sum(
            dark > 0,
            axis=0
        ).astype(
            np.float32
        )
    else:
        projection = np.sum(
            dark > 0,
            axis=1
        ).astype(
            np.float32
        )

    projection = smooth_projection(
        projection,
        5
    )

    cell_size = (
        length
        /
        GRID_SIZE
    )

    radius = max(
        4,
        int(
            cell_size
            *
            0.12
        )
    )

    for i in range(
        1,
        len(refined) - 1
    ):
        current = refined[i]

        start = max(
            1,
            current - radius
        )

        end = min(
            length - 2,
            current + radius
        )

        local = projection[
            start:end + 1
        ]

        if len(local) == 0:
            continue

        refined[i] = (
            start
            +
            int(
                np.argmax(local)
            )
        )

    refined[0] = 0
    refined[-1] = length - 1

    return refined


# ============================================================
# DEBUG GRID
# ============================================================

def draw_debug_grid(
    binary,
    vertical_lines,
    horizontal_lines
):
    debug = cv2.cvtColor(
        binary,
        cv2.COLOR_GRAY2BGR
    )

    height, width = binary.shape

    # --------------------------------------------------------
    # Vertical lines
    # --------------------------------------------------------

    for i, x in enumerate(
        vertical_lines
    ):
        x = int(x)

        cv2.line(
            debug,
            (x, 0),
            (x, height - 1),
            (0, 0, 255),
            2
        )

        label = (
            f"V{i}: {x}"
        )

        cv2.putText(
            debug,
            label,
            (
                min(
                    x + 4,
                    width - 100
                ),
                20
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            DEBUG_FONT_SCALE,
            (0, 0, 255),
            1,
            cv2.LINE_AA
        )

    # --------------------------------------------------------
    # Horizontal lines
    # --------------------------------------------------------

    for i, y in enumerate(
        horizontal_lines
    ):
        y = int(y)

        cv2.line(
            debug,
            (0, y),
            (width - 1, y),
            (255, 0, 0),
            2
        )

        label = (
            f"H{i}: {y}"
        )

        cv2.putText(
            debug,
            label,
            (
                5,
                max(
                    15,
                    y - 5
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            DEBUG_FONT_SCALE,
            (255, 0, 0),
            1,
            cv2.LINE_AA
        )

    # --------------------------------------------------------
    # Cell labels
    # --------------------------------------------------------

    for row in range(
        GRID_SIZE
    ):
        for col in range(
            GRID_SIZE
        ):
            x1 = vertical_lines[col]
            x2 = vertical_lines[col + 1]

            y1 = horizontal_lines[row]
            y2 = horizontal_lines[row + 1]

            cx = int(
                (x1 + x2) / 2
            )

            cy = int(
                (y1 + y2) / 2
            )

            label = (
                f"{row + 1},{col + 1}"
            )

            text_size = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                1
            )[0]

            tx = (
                cx
                -
                text_size[0] // 2
            )

            ty = (
                cy
                +
                text_size[1] // 2
            )

            cv2.putText(
                debug,
                label,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 128, 0),
                1,
                cv2.LINE_AA
            )

    return debug