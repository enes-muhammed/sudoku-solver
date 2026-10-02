from pathlib import Path
import cv2
import numpy as np
import tkinter as tk
from tkinter import simpledialog
from PIL import Image, ImageTk
from board import (
    OUTPUT_SIZE,
    find_sudoku_contour,
    four_point_transform,
    clean_warped_image,
    detect_grid_lines,
    refine_line_positions,
    draw_debug_grid,
)
from cells import (
    extract_cell_images,
    add_rebuilt_grid,
)
from recognizer import (
    recognize_grid,
    prompt_for_known_filled_cells,
    create_rebuilt_with_recognition,
    create_recognition_result,
)

SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp",
}

# ============================================================
# RUN MODE
# ============================================================

SINGLE_MODE = True

SINGLE_FILE = "current.jpeg"

COLLECTION_DIR = "sudoku_collection"


# ============================================================
# CONSTANTS
# ============================================================

# Candidate detection
MIN_CONTOUR_AREA_RATIO = 0.06
MAX_CANDIDATES = 80

# Candidate validation
MIN_GRID_SCORE = 250.0

# Candidate geometry
MIN_QUAD_SQUARENESS = 0.30
MIN_QUAD_AREA_RATIO = 0.06

# Duplicate candidates
DUPLICATE_DISTANCE = 8.0

# Edge / contour detection
CANNY_LOW = 40
CANNY_HIGH = 150


# ============================================================
# BASIC GEOMETRY
# ============================================================

def order_quad(points):
    points = np.asarray(
        points,
        dtype=np.float32,
    )

    if points.shape != (4, 2):
        return None

    result = np.zeros(
        (4, 2),
        dtype=np.float32,
    )

    sums = points.sum(axis=1)
    diffs = np.diff(
        points,
        axis=1,
    ).reshape(-1)

    result[0] = points[np.argmin(sums)]
    result[2] = points[np.argmax(sums)]

    result[1] = points[np.argmin(diffs)]
    result[3] = points[np.argmax(diffs)]

    return result


def quad_area(quad):
    quad = order_quad(quad)

    if quad is None:
        return 0.0

    return abs(
        cv2.contourArea(
            quad.astype(np.float32)
        )
    )


def quad_squareness(quad):
    """
    Measures how balanced the four sides are.

    This is NOT a requirement for a square.
    A perspective-distorted Sudoku can be a parallelogram
    or a heavily skewed quadrilateral.

    We only reject completely absurd shapes.
    """

    quad = order_quad(quad)

    if quad is None:
        return 0.0

    top_left, top_right, bottom_right, bottom_left = quad

    width_top = np.linalg.norm(
        top_right - top_left
    )

    width_bottom = np.linalg.norm(
        bottom_right - bottom_left
    )

    height_left = np.linalg.norm(
        bottom_left - top_left
    )

    height_right = np.linalg.norm(
        bottom_right - top_right
    )

    values = [
        width_top,
        width_bottom,
        height_left,
        height_right,
    ]

    if min(values) <= 1:
        return 0.0

    width_ratio = (
        min(width_top, width_bottom)
        / max(width_top, width_bottom)
    )

    height_ratio = (
        min(height_left, height_right)
        / max(height_left, height_right)
    )

    return float(
        min(
            width_ratio,
            height_ratio,
        )
    )


def quad_center(quad):
    quad = order_quad(quad)

    if quad is None:
        return np.zeros(
            2,
            dtype=np.float32,
        )

    return quad.mean(axis=0)


# ============================================================
# QUAD VALIDATION
# ============================================================

def is_valid_quad(
    quad,
    image_shape,
):
    """
    Basic sanity check.

    IMPORTANT:
    We intentionally allow:
        - parallelograms
        - strong perspective
        - non-90-degree corners

    We only reject geometrically impossible candidates.
    """

    if quad is None:
        return False

    quad = order_quad(quad)

    if quad is None:
        return False

    height, width = image_shape[:2]

    image_area = height * width

    area = quad_area(quad)

    if area < image_area * MIN_QUAD_AREA_RATIO:
        return False

    squareness = quad_squareness(
        quad
    )

    if squareness < MIN_QUAD_SQUARENESS:
        return False

    # All points must be inside the image with a small tolerance.
    tolerance = max(
        10,
        int(min(height, width) * 0.03),
    )

    if np.any(
        quad[:, 0] < -tolerance
    ):
        return False

    if np.any(
        quad[:, 1] < -tolerance
    ):
        return False

    if np.any(
        quad[:, 0] > width - 1 + tolerance
    ):
        return False

    if np.any(
        quad[:, 1] > height - 1 + tolerance
    ):
        return False

    return True


# ============================================================
# CONTOUR CANDIDATE GENERATION
# ============================================================

def collect_contour_quads(
    image,
):
    """
    Generate MANY possible Sudoku quadrilaterals.

    We do not immediately trust the largest contour.

    This is specifically important when:
    - two Sudokus are visible,
    - one Sudoku is partially above another,
    - page borders merge with Sudoku borders,
    - a contour connects pieces from different boards.

    OpenCV's contour + polygon approximation is useful for
    generating candidates, but candidates are validated later.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    image_area = (
        image.shape[0]
        * image.shape[1]
    )

    all_candidates = []

    preprocessing_variants = []

    # --------------------------------------------------------
    # Variant 1: Canny
    # --------------------------------------------------------

    edges = cv2.Canny(
        gray,
        CANNY_LOW,
        CANNY_HIGH,
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (7, 7),
    )

    closed = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel,
    )

    preprocessing_variants.append(
        closed
    )

    # --------------------------------------------------------
    # Variant 2: adaptive threshold
    # --------------------------------------------------------

    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        9,
    )

    adaptive = cv2.morphologyEx(
        adaptive,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (5, 5),
        ),
    )

    preprocessing_variants.append(
        adaptive
    )

    # --------------------------------------------------------
    # Variant 3: blackhat
    # --------------------------------------------------------

    blackhat_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (21, 21),
    )

    blackhat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        blackhat_kernel,
    )

    _, blackhat_binary = cv2.threshold(
        blackhat,
        25,
        255,
        cv2.THRESH_BINARY,
    )

    blackhat_binary = cv2.morphologyEx(
        blackhat_binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (5, 5),
        ),
    )

    preprocessing_variants.append(
        blackhat_binary
    )

    # --------------------------------------------------------
    # Extract contours from every variant
    # --------------------------------------------------------

    for binary in preprocessing_variants:

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        contours = sorted(
            contours,
            key=cv2.contourArea,
            reverse=True,
        )

        for contour in contours[:MAX_CANDIDATES]:

            area = cv2.contourArea(
                contour
            )

            if (
                area
                < image_area
                * MIN_CONTOUR_AREA_RATIO
            ):
                continue

            perimeter = cv2.arcLength(
                contour,
                True,
            )

            if perimeter <= 0:
                continue

            # Several approximation strengths.
            #
            # This matters for curved / imperfect / skewed
            # Sudoku borders.
            for epsilon_ratio in (
                0.005,
                0.008,
                0.012,
                0.016,
                0.020,
                0.025,
                0.030,
                0.040,
                0.050,
                0.065,
                0.080,
                0.100,
            ):

                approx = cv2.approxPolyDP(
                    contour,
                    epsilon_ratio * perimeter,
                    True,
                )

                if len(approx) != 4:
                    continue

                if not cv2.isContourConvex(
                    approx
                ):
                    continue

                quad = approx.reshape(
                    4,
                    2,
                ).astype(
                    np.float32
                )

                quad = order_quad(
                    quad
                )

                if not is_valid_quad(
                    quad,
                    image.shape,
                ):
                    continue

                all_candidates.append(
                    quad
                )

                # Do not generate dozens of almost
                # identical approximations from one contour.
                break

    return deduplicate_quads(
        all_candidates
    )


def deduplicate_quads(
    candidates,
):
    result = []

    for candidate in candidates:

        duplicate = False

        for old in result:

            mean_distance = np.mean(
                np.linalg.norm(
                    candidate - old,
                    axis=1,
                )
            )

            if (
                mean_distance
                < DUPLICATE_DISTANCE
            ):
                duplicate = True
                break

        if not duplicate:
            result.append(
                candidate
            )

    return result


# ============================================================
# GRID QUALITY
# ============================================================

def line_spacing_score(
    lines,
    expected_spacing,
):
    if len(lines) != 10:
        return -1e9

    lines = np.asarray(
        lines,
        dtype=np.float32,
    )

    spacing = np.diff(
        lines
    )

    if len(spacing) != 9:
        return -1e9

    mean_error = np.mean(
        np.abs(
            spacing
            - expected_spacing
        )
    ) / max(
        expected_spacing,
        1.0,
    )

    std_error = np.std(
        spacing
    ) / max(
        expected_spacing,
        1.0,
    )

    max_error = np.max(
        np.abs(
            spacing
            - expected_spacing
        )
    ) / max(
        expected_spacing,
        1.0,
    )

    # Soft scoring.
    #
    # We do NOT require a perfect square.
    # Perspective can make the spacing imperfect.
    score = 1000.0

    score -= mean_error * 1000.0
    score -= std_error * 500.0
    score -= max_error * 300.0

    return score


def grid_quality_score(
    warped,
):
    """
    Determine whether a candidate quadrilateral actually
    produces a Sudoku-like 9x9 grid.

    This is the most important protection against the bug:

        bottom corners of another Sudoku
        becoming the top corners of our Sudoku.
    """

    try:
        clean = clean_warped_image(
            warped
        )

        vertical, horizontal = (
            detect_grid_lines(
                clean
            )
        )

        if len(vertical) != 10:
            return -1e9

        if len(horizontal) != 10:
            return -1e9

        vertical = refine_line_positions(
            clean,
            vertical,
            "vertical",
        )

        horizontal = refine_line_positions(
            clean,
            horizontal,
            "horizontal",
        )

        vertical = np.asarray(
            vertical,
            dtype=np.float32,
        )

        horizontal = np.asarray(
            horizontal,
            dtype=np.float32,
        )

        expected_v = (
            warped.shape[1] - 1
        ) / 9.0

        expected_h = (
            warped.shape[0] - 1
        ) / 9.0

        v_score = line_spacing_score(
            vertical,
            expected_v,
        )

        h_score = line_spacing_score(
            horizontal,
            expected_h,
        )

        if (
            v_score < -1000
            or h_score < -1000
        ):
            return -1e9

        # ----------------------------------------------------
        # Extra check:
        # The grid should actually contain dark line energy.
        # ----------------------------------------------------

        dark = 255 - clean

        horizontal_energy = []

        for y in horizontal:
            yi = int(
                round(y)
            )

            y1 = max(
                0,
                yi - 2,
            )

            y2 = min(
                clean.shape[0],
                yi + 3,
            )

            horizontal_energy.append(
                float(
                    dark[
                        y1:y2,
                        :
                    ].mean()
                )
            )

        vertical_energy = []

        for x in vertical:
            xi = int(
                round(x)
            )

            x1 = max(
                0,
                xi - 2,
            )

            x2 = min(
                clean.shape[1],
                xi + 3,
            )

            vertical_energy.append(
                float(
                    dark[
                        :,
                        x1:x2
                    ].mean()
                )
            )

        horizontal_energy = np.asarray(
            horizontal_energy
        )

        vertical_energy = np.asarray(
            vertical_energy
        )

        energy_score = (
            horizontal_energy.mean()
            + vertical_energy.mean()
        )

        # Weak grid candidates should not win simply because
        # their spacing happens to look regular.
        if energy_score < 15:
            return -1e9

        score = (
            v_score
            + h_score
            + energy_score * 10.0
        )

        return float(score)

    except Exception:
        return -1e9


# ============================================================
# BORDER SUPPORT
# ============================================================

def border_support_score(
    image,
    quad,
):
    """
    Check whether the four sides of the candidate actually
    coincide with dark Sudoku-like borders.

    This helps reject a quad created by mixing:
        bottom corners of Sudoku A
        bottom corners of Sudoku B
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    edges = cv2.Canny(
        gray,
        50,
        150,
    )

    quad = order_quad(
        quad
    )

    if quad is None:
        return -1e9

    height, width = gray.shape

    total = 0.0

    for index in range(4):

        p1 = quad[index]
        p2 = quad[(index + 1) % 4]

        length = np.linalg.norm(
            p2 - p1
        )

        if length < 20:
            return -1e9

        sample_count = max(
            50,
            int(length / 5),
        )

        values = []

        for t in np.linspace(
            0.02,
            0.98,
            sample_count,
        ):

            x = (
                p1[0]
                * (1.0 - t)
                + p2[0]
                * t
            )

            y = (
                p1[1]
                * (1.0 - t)
                + p2[1]
                * t
            )

            x = int(
                round(x)
            )

            y = int(
                round(y)
            )

            if (
                x < 0
                or x >= width
                or y < 0
                or y >= height
            ):
                continue

            # Search a small perpendicular neighborhood.
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]

            norm = np.hypot(
                dx,
                dy,
            )

            if norm <= 0:
                continue

            nx = -dy / norm
            ny = dx / norm

            local_values = []

            for offset in (
                -4,
                -2,
                0,
                2,
                4,
            ):

                xx = int(
                    round(
                        x + nx * offset
                    )
                )

                yy = int(
                    round(
                        y + ny * offset
                    )
                )

                if (
                    0 <= xx < width
                    and 0 <= yy < height
                ):
                    local_values.append(
                        edges[yy, xx]
                    )

            if local_values:
                values.append(
                    max(local_values)
                )

        if not values:
            continue

        total += (
            np.mean(values) / 255.0
        )

    return total / 4.0


# ============================================================
# CANDIDATE SCORING
# ============================================================

def candidate_score(
    image,
    quad,
):
    """
    Complete candidate score.

    A candidate wins because it behaves like a Sudoku,
    not merely because it is a large rectangle.
    """

    if not is_valid_quad(
        quad,
        image.shape,
    ):
        return -1e9

    try:
        warped = four_point_transform(
            image,
            quad,
            OUTPUT_SIZE,
        )

    except Exception:
        return -1e9

    grid_score = grid_quality_score(
        warped
    )

    if grid_score < -1000:
        return -1e9

    border_score = (
        border_support_score(
            image,
            quad,
        )
    )

    area = quad_area(
        quad
    )

    image_area = (
        image.shape[0]
        * image.shape[1]
    )

    area_ratio = (
        area
        / image_area
    )

    # Area is only a small bonus.
    #
    # This is deliberate.
    #
    # We do NOT want:
    #
    # "largest rectangle = Sudoku"
    #
    # because that is exactly how a nearby Sudoku can hijack
    # the detection.
    area_bonus = (
        np.clip(
            area_ratio,
            0.0,
            0.9,
        )
        * 50.0
    )

    total = (
        grid_score
        + border_score * 150.0
        + area_bonus
    )

    return float(
        total
    )


# ============================================================
# BEST QUAD
# ============================================================

def find_best_sudoku_quad(
    image,
):
    """
    Main Sudoku detector.

    IMPORTANT DIFFERENCE FROM THE PREVIOUS VERSION:

    We NEVER immediately trust the existing detector.

    Every plausible candidate is compared.

    This specifically protects against:
        Sudoku A bottom-left
        Sudoku A bottom-right
        Sudoku B bottom-right
        Sudoku B bottom-left

    becoming one fake four-point board.
    """

    candidates = []

    # --------------------------------------------------------
    # 1. Existing detector
    # --------------------------------------------------------

    try:
        primary = find_sudoku_contour(
            image
        )

        if primary is not None:
            primary = order_quad(
                np.asarray(
                    primary,
                    dtype=np.float32,
                )
            )

            if is_valid_quad(
                primary,
                image.shape,
            ):
                candidates.append(
                    primary
                )

    except Exception:
        pass

    # --------------------------------------------------------
    # 2. Additional contour candidates
    # --------------------------------------------------------

    candidates.extend(
        collect_contour_quads(
            image
        )
    )

    candidates = deduplicate_quads(
        candidates
    )

    if not candidates:
        return None

    print(
        f"    Testing "
        f"{len(candidates)} Sudoku candidates..."
    )

    best_quad = None
    best_score = -1e12

    scores = []

    for index, quad in enumerate(
        candidates,
        start=1,
    ):

        score = candidate_score(
            image,
            quad,
        )

        scores.append(
            (
                score,
                quad,
            )
        )

        if score > best_score:
            best_score = score
            best_quad = quad

    scores.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    if best_quad is None:
        return None

    # --------------------------------------------------------
    # Diagnostic information
    # --------------------------------------------------------

    print(
        f"    Best candidate score: "
        f"{best_score:.1f}"
    )

    if len(scores) > 1:
        print(
            f"    Second candidate score: "
            f"{scores[1][0]:.1f}"
        )

    # --------------------------------------------------------
    # Only reject if absolutely no Sudoku-like candidate
    # exists.
    #
    # We intentionally do NOT make this too strict.
    # --------------------------------------------------------

    if best_score < MIN_GRID_SCORE:
        print(
            "    Warning: Sudoku candidate "
            "is weak, using best available candidate."
        )

    return best_quad


# ============================================================
# UI
# ============================================================

class SudokuViewer:

    def __init__(
        self,
        results,
    ):
        self.results = results
        self.index = 0

        self.root = tk.Tk()

        self.root.title(
            "Sudoku Image Processor"
        )

        self.root.geometry(
            "1500x900"
        )

        self.root.configure(
            bg="#111111"
        )

        self.title_label = tk.Label(
            self.root,
            text="",
            font=(
                "Arial",
                18,
                "bold",
            ),
            fg="white",
            bg="#111111",
        )

        self.title_label.pack(
            pady=(15, 5)
        )

        self.counter_label = tk.Label(
            self.root,
            text="",
            font=(
                "Arial",
                11,
            ),
            fg="#bbbbbb",
            bg="#111111",
        )

        self.counter_label.pack(
            pady=(0, 10)
        )

        self.image_frame = tk.Frame(
            self.root,
            bg="#111111",
        )

        self.image_frame.pack(
            fill="both",
            expand=True,
            padx=20,
            pady=10,
        )

        self.image_labels = []

        for _ in range(3):

            label = tk.Label(
                self.image_frame,
                bg="#222222",
            )

            label.pack(
                side="left",
                fill="both",
                expand=True,
                padx=5,
            )

            self.image_labels.append(
                label
            )

        self.right_column = tk.Frame(
            self.image_frame,
            bg="#111111",
        )

        self.right_column.pack(
            side="left",
            fill="both",
            expand=True,
            padx=5,
        )

        self.rebuilt_panel = tk.Frame(
            self.right_column,
            bg="#222222",
        )

        self.rebuilt_panel.pack(
            fill="both",
            expand=True,
        )

        self.rebuilt_label = tk.Label(
            self.rebuilt_panel,
            bg="#222222",
            fg="white",
            font=(
                "Arial",
                10,
                "bold",
            ),
            compound="bottom",
        )

        self.rebuilt_label.pack(
            fill="both",
            expand=True,
        )

        self.recognition_panel = tk.Frame(
            self.right_column,
            bg="#222222",
        )

        self.recognition_panel.pack(
            fill="both",
            expand=True,
            pady=(8, 0),
        )

        self.recognition_label = tk.Label(
            self.recognition_panel,
            bg="#222222",
            fg="white",
            font=(
                "Arial",
                10,
                "bold",
            ),
            compound="bottom",
        )

        self.recognition_label.pack(
            fill="both",
            expand=True,
        )

        self.button_frame = tk.Frame(
            self.root,
            bg="#111111",
        )

        self.button_frame.pack(
            pady=(5, 20)
        )

        self.previous_button = tk.Button(
            self.button_frame,
            text="← Önceki",
            command=self.previous,
            width=15,
            font=(
                "Arial",
                11,
            ),
        )

        self.previous_button.pack(
            side="left",
            padx=10,
        )

        self.next_button = tk.Button(
            self.button_frame,
            text="Sonraki →",
            command=self.next,
            width=15,
            font=(
                "Arial",
                11,
            ),
        )

        self.next_button.pack(
            side="left",
            padx=10,
        )

        self.root.bind(
            "<Left>",
            lambda event: self.previous(),
        )

        self.root.bind(
            "<Right>",
            lambda event: self.next(),
        )

        self.root.bind(
            "<Escape>",
            lambda event: self.root.destroy(),
        )

        self.show_current()

    def resize_image(
        self,
        image,
        max_width=280,
        max_height=700,
    ):
        h, w = image.shape[:2]

        scale = min(
            max_width / w,
            max_height / h,
        )

        scale = min(
            scale,
            1.0,
        )

        new_width = max(
            1,
            int(w * scale),
        )

        new_height = max(
            1,
            int(h * scale),
        )

        return cv2.resize(
            image,
            (
                new_width,
                new_height,
            ),
            interpolation=cv2.INTER_AREA,
        )

    def cv_to_tk(
        self,
        image,
        max_width=280,
        max_height=700,
    ):
        if len(image.shape) == 2:

            image = cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2RGB,
            )

        else:

            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB,
            )

        image = self.resize_image(
            image,
            max_width=max_width,
            max_height=max_height,
        )

        pil_image = Image.fromarray(
            image
        )

        return ImageTk.PhotoImage(
            pil_image
        )

    def show_current(self):

        result = self.results[
            self.index
        ]

        images = [
            ("Original", result["original"]),
            ("Warp 1", result["warped"]),
            ("Grid Debug", result["grid_debug"]),
        ]

        for i, (name, image) in enumerate(images):
            tk_image = self.cv_to_tk(
                image
            )

            self.image_labels[i].config(
                image=tk_image,
                text=name,
                compound="bottom",
                fg="white",
                font=(
                    "Arial",
                    10,
                    "bold",
                ),
            )

            self.image_labels[i].image = (
                tk_image
            )

        rebuilt_image = self.cv_to_tk(
            result["rebuilt"]
        )

        self.rebuilt_label.config(
            image=rebuilt_image,
            text="Rebuilt",
            compound="bottom",
            fg="white",
            font=(
                "Arial",
                10,
                "bold",
            ),
        )

        self.rebuilt_label.image = rebuilt_image

        recognition_image = self.cv_to_tk(
            result["recognition_result"]
        )

        self.recognition_label.config(
            image=recognition_image,
            text="Recognition Result",
            compound="bottom",
            fg="white",
            font=(
                "Arial",
                10,
                "bold",
            ),
        )

        self.recognition_label.image = recognition_image

        self.title_label.config(
            text=result["name"]
        )

        self.counter_label.config(
            text=(
                f"{self.index + 1}"
                f" / "
                f"{len(self.results)}"
            )
        )

        self.previous_button.config(
            state=(
                tk.NORMAL
                if self.index > 0
                else tk.DISABLED
            )
        )

        self.next_button.config(
            state=(
                tk.NORMAL
                if self.index
                < len(self.results) - 1
                else tk.DISABLED
            )
        )

    def previous(self):

        if self.index > 0:
            self.index -= 1
            self.show_current()

    def next(self):

        if (
            self.index
            < len(self.results) - 1
        ):
            self.index += 1
            self.show_current()

    def run(self):
        self.root.mainloop()


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(
    image_path,
):
    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise RuntimeError(
            f"Could not read image:\n"
            f"{image_path}"
        )

    print(
        f"Processing: "
        f"{image_path.name}"
    )

    # --------------------------------------------------------
    # 1. Find Sudoku
    # --------------------------------------------------------

    quad = find_best_sudoku_quad(
        image
    )

    if quad is None:
        raise RuntimeError(
            "Sudoku board could not be detected."
        )

    print(
        "    Sudoku candidate selected."
    )

    # --------------------------------------------------------
    # 2. Warp 1
    # --------------------------------------------------------

    warped = four_point_transform(
        image,
        quad,
        OUTPUT_SIZE,
    )

    # --------------------------------------------------------
    # 3. Clean
    # --------------------------------------------------------

    clean = clean_warped_image(
        warped
    )

    # --------------------------------------------------------
    # 4. Grid detection
    # --------------------------------------------------------

    vertical_lines, horizontal_lines = (
        detect_grid_lines(
            clean
        )
    )

    vertical_lines = refine_line_positions(
        clean,
        vertical_lines,
        "vertical",
    )

    horizontal_lines = refine_line_positions(
        clean,
        horizontal_lines,
        "horizontal",
    )

    # --------------------------------------------------------
    # The outer grid boundaries MUST always be image bounds.
    # --------------------------------------------------------

    vertical_lines[0] = 0

    vertical_lines[-1] = (
        clean.shape[1] - 1
    )

    horizontal_lines[0] = 0

    horizontal_lines[-1] = (
        clean.shape[0] - 1
    )

    # --------------------------------------------------------
    # 5. Debug grid
    # --------------------------------------------------------

    grid_debug = draw_debug_grid(
        clean,
        vertical_lines,
        horizontal_lines,
    )

    # --------------------------------------------------------
    # 6. Rebuilt
    # --------------------------------------------------------

    # ========================================================
    # EXTRACT CELLS
    # ========================================================

    cell_images = extract_cell_images(
        clean,
        vertical_lines,
        horizontal_lines,
    )

    # ========================================================
    # REBUILD
    # ========================================================

    cell_rows = []

    for row_cells in cell_images:
        cell_row = cv2.hconcat(
            row_cells
        )

        cell_rows.append(
            cell_row
        )

    rebuilt = cv2.vconcat(
        cell_rows
    )

    rebuilt = add_rebuilt_grid(
        rebuilt
    )

    # ========================================================
    # RECOGNITION
    # ========================================================

    recognition, normalized_digits, occupied = recognize_grid(
        cell_images
    )

    known_values = [[0 for _ in range(9)] for _ in range(9)]

    root = None
    try:
        root = tk.Tk()
        root.withdraw()
    except tk.TclError:
        root = None

    for row in range(9):
        for col in range(9):
            if occupied[row][col] == 1 and recognition[row][col] == 0:
                if root is None:
                    break

                try:
                    value = simpledialog.askinteger(
                        "Known filled cell",
                        f"Cell [{row},{col}] is filled. Enter value (1-9) or 0 to skip:",
                        parent=root,
                        minvalue=0,
                        maxvalue=9,
                    )
                except tk.TclError:
                    break

                if value is None:
                    value = 0

                if 1 <= value <= 9:
                    recognition[row][col] = value
                    known_values[row][col] = value
                else:
                    known_values[row][col] = 0

    if root is not None:
        root.destroy()

    print(
        "    Recognition grid:"
    )

    for row in recognition:
        print(
            "    "
            + " ".join(
                str(value)
                for value in row
            )
        )

    # ========================================================
    # RECOGNITION RESULT
    # ========================================================

    rebuilt_with_recognition = create_rebuilt_with_recognition(
        rebuilt,
        recognition,
        occupied,
    )

    recognition_result = create_recognition_result(
        rebuilt,
        recognition,
        occupied,
        known_values,
    )

    return {
        "name": image_path.name,
        "original": image,
        "warped": warped,
        "grid_debug": grid_debug,
        "recognition_result": recognition_result,
        "rebuilt": rebuilt_with_recognition,

        "cell_images": cell_images,
        "recognition": recognition,
        "occupied": occupied,
        "known_values": known_values,
        "normalized_digits": normalized_digits,
    }


# ============================================================
# COLLECTION
# ============================================================

def find_images(
    collection_dir,
):
    images = []

    for path in collection_dir.iterdir():

        if not path.is_file():
            continue

        if (
            path.suffix.lower()
            not in SUPPORTED_EXTENSIONS
        ):
            continue

        images.append(
            path
        )

    images.sort(
        key=lambda path:
        path.name.lower()
    )

    return images


# ============================================================
# MAIN
# ============================================================

def main():
    base_dir = Path(
        __file__
    ).resolve().parent

    # ========================================================
    # SINGLE MODE
    # ========================================================

    if SINGLE_MODE:
        image_path = (
            base_dir / SINGLE_FILE
        )

        if not image_path.exists():
            print(
                "ERROR: Selected file not found."
            )
            print(
                f"Expected:\n"
                f"{image_path}"
            )
            return

        image_paths = [image_path]

    # ========================================================
    # COLLECTION MODE
    # ========================================================

    else:
        collection_dir = (
            base_dir
            / COLLECTION_DIR
        )

        if not collection_dir.exists():
            print(
                "ERROR: "
                "Collection folder not found."
            )
            print(
                f"Expected:\n"
                f"{collection_dir}"
            )
            return

        image_paths = find_images(
            collection_dir
        )

        if not image_paths:
            print(
                "ERROR: "
                "No Sudoku images found."
            )
            return

    # ========================================================
    # PROCESS
    # ========================================================

    print("=" * 60)

    if SINGLE_MODE:
        print(
            "SUDOKU PROCESSOR - SINGLE MODE"
        )
    else:
        print(
            "SUDOKU PROCESSOR - COLLECTION MODE"
        )

    print("=" * 60)

    print(
        f"Processing "
        f"{len(image_paths)} image(s)."
    )

    results = []

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):
        print(
            f"[{index}/{len(image_paths)}] "
            f"{image_path.name}"
        )

        try:
            result = process_image(
                image_path
            )

            results.append(
                result
            )

            print(
                "    OK"
            )

        except Exception as error:
            print(
                f"    ERROR: "
                f"{error}"
            )

    print("=" * 60)

    print(
        f"Successfully processed "
        f"{len(results)} / "
        f"{len(image_paths)}"
    )

    print("=" * 60)

    if not results:
        return

    viewer = SudokuViewer(
        results
    )

    viewer.run()

if __name__ == "__main__":
    main()
