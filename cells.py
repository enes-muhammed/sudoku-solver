import cv2
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

GRID_SIZE = 9

CELL_CROP_RATIO = 0.015

MIN_COMPONENT_AREA = 12

DIGIT_CENTER_RATIO = 0.68

MIN_DIGIT_HEIGHT_RATIO = 0.15
MAX_DIGIT_HEIGHT_RATIO = 0.90

MIN_DIGIT_WIDTH_RATIO = 0.03
MAX_DIGIT_WIDTH_RATIO = 0.90


# ============================================================
# FIND DIGIT COMPONENTS
# ============================================================

def find_digit_components(cell):
    h, w = cell.shape

    dark = cv2.inRange(
        cell,
        0,
        127,
    )

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            dark,
            connectivity=8,
        )
    )

    center_x = w / 2.0
    center_y = h / 2.0

    center_width = (
        w * DIGIT_CENTER_RATIO
    )

    center_height = (
        h * DIGIT_CENTER_RATIO
    )

    center_x1 = (
        center_x
        - center_width / 2
    )

    center_x2 = (
        center_x
        + center_width / 2
    )

    center_y1 = (
        center_y
        - center_height / 2
    )

    center_y2 = (
        center_y
        + center_height / 2
    )

    candidates = []

    for label in range(
        1,
        num_labels,
    ):
        x = stats[
            label,
            cv2.CC_STAT_LEFT,
        ]

        y = stats[
            label,
            cv2.CC_STAT_TOP,
        ]

        width = stats[
            label,
            cv2.CC_STAT_WIDTH,
        ]

        height = stats[
            label,
            cv2.CC_STAT_HEIGHT,
        ]

        area = stats[
            label,
            cv2.CC_STAT_AREA,
        ]

        cx, cy = centroids[
            label
        ]

        # ----------------------------------------------------
        # Minimum amount of dark pixels
        # ----------------------------------------------------

        if area < MIN_COMPONENT_AREA:
            continue

        # ----------------------------------------------------
        # Ignore anything touching the cell wall
        # ----------------------------------------------------

        touches_border = (
            x <= 0
            or y <= 0
            or x + width >= w - 1
            or y + height >= h - 1
        )

        if touches_border:
            continue

        # ----------------------------------------------------
        # Component centroid must be near the center
        # ----------------------------------------------------

        center_inside = (
            center_x1 <= cx <= center_x2
            and center_y1 <= cy <= center_y2
        )

        if not center_inside:
            continue

        # ----------------------------------------------------
        # Bounding-box center must also be near the center
        # ----------------------------------------------------

        box_center_x = (
            x + width / 2
        )

        box_center_y = (
            y + height / 2
        )

        box_center_inside = (
            center_x1
            <= box_center_x
            <= center_x2
            and
            center_y1
            <= box_center_y
            <= center_y2
        )

        if not box_center_inside:
            continue

        # ----------------------------------------------------
        # Digit height
        # ----------------------------------------------------

        height_ratio = (
            height
            / max(h, 1)
        )

        if (
            height_ratio
            < MIN_DIGIT_HEIGHT_RATIO
        ):
            continue

        if (
            height_ratio
            > MAX_DIGIT_HEIGHT_RATIO
        ):
            continue

        # ----------------------------------------------------
        # Digit width
        # ----------------------------------------------------

        width_ratio = (
            width
            / max(w, 1)
        )

        if (
            width_ratio
            < MIN_DIGIT_WIDTH_RATIO
        ):
            continue

        if (
            width_ratio
            > MAX_DIGIT_WIDTH_RATIO
        ):
            continue

        candidates.append(
            (
                label,
                area,
                width,
                height,
                cx,
                cy,
            )
        )

    return (
        labels,
        candidates,
    )


# ============================================================
# KEEP DIGIT / REMOVE EVERYTHING ELSE
# ============================================================

def keep_digit_and_remove_everything_else(cell):
    h, w = cell.shape

    labels, candidates = find_digit_components(
        cell
    )

    # --------------------------------------------------------
    # No meaningful component
    # --------------------------------------------------------

    if len(candidates) == 0:
        return np.full_like(
            cell,
            255,
        )

    # --------------------------------------------------------
    # KURAL 1:
    #
    # Bir hücrede birden fazla anlamlı şekil varsa
    # bunların hiçbiri rakam kabul edilmez.
    # --------------------------------------------------------

    if len(candidates) > 1:
        return np.full_like(
            cell,
            255,
        )

    # --------------------------------------------------------
    # Exactly one candidate
    # --------------------------------------------------------

    candidate = candidates[0]

    (
        label,
        area,
        width,
        height,
        cx,
        cy,
    ) = candidate

    # --------------------------------------------------------
    # Extra size filtering
    #
    # Tek kalan küçük şekil bir not olabilir.
    # --------------------------------------------------------

    area_ratio = (
        area
        / max(w * h, 1)
    )

    height_ratio = (
        height
        / max(h, 1)
    )

    width_ratio = (
        width
        / max(w, 1)
    )

    if area_ratio < 0.018:
        return np.full_like(
            cell,
            255,
        )

    if height_ratio < 0.25:
        return np.full_like(
            cell,
            255,
        )

    if width_ratio < 0.04:
        return np.full_like(
            cell,
            255,
        )

    # --------------------------------------------------------
    # Exactly one sufficiently large component
    #
    # This is our digit.
    # --------------------------------------------------------

    result = np.full_like(
        cell,
        255,
    )

    keep_mask = labels == label

    result[keep_mask] = cell[keep_mask]

    return result


# ============================================================
# EXTRACT 81 INDIVIDUAL CELLS
# ============================================================

def extract_cell_images(
    clean,
    vertical_lines,
    horizontal_lines,
):
    """
    Sudoku görüntüsünü 81 ayrı hücreye böler.

    Returns:
        list[list[np.ndarray]]
    """

    cells = []

    for row in range(
        GRID_SIZE
    ):
        row_cells = []

        for col in range(
            GRID_SIZE
        ):
            x1 = int(
                vertical_lines[col]
            )

            x2 = int(
                vertical_lines[col + 1]
            )

            y1 = int(
                horizontal_lines[row]
            )

            y2 = int(
                horizontal_lines[row + 1]
            )

            width = x2 - x1
            height = y2 - y1

            # ------------------------------------------------
            # Remove a small amount around cell walls
            # ------------------------------------------------

            margin_x = max(
                1,
                int(
                    width
                    * CELL_CROP_RATIO
                ),
            )

            margin_y = max(
                1,
                int(
                    height
                    * CELL_CROP_RATIO
                ),
            )

            cx1 = x1 + margin_x
            cx2 = x2 - margin_x

            cy1 = y1 + margin_y
            cy2 = y2 - margin_y

            # ------------------------------------------------
            # Keep coordinates inside image
            # ------------------------------------------------

            cx1 = max(
                0,
                cx1,
            )

            cy1 = max(
                0,
                cy1,
            )

            cx2 = min(
                clean.shape[1],
                cx2,
            )

            cy2 = min(
                clean.shape[0],
                cy2,
            )

            # ------------------------------------------------
            # Extract cell
            # ------------------------------------------------

            cell = clean[
                cy1:cy2,
                cx1:cx2,
            ]

            # ------------------------------------------------
            # Remove notes / walls / noise
            # ------------------------------------------------

            cell = (
                keep_digit_and_remove_everything_else(
                    cell
                )
            )

            row_cells.append(
                cell
            )

        cells.append(
            row_cells
        )

    return cells


# ============================================================
# REBUILD SUDOKU FROM CELLS
# ============================================================

def add_rebuilt_grid(
    image,
):
    result = image.copy()

    h, w = result.shape[:2]

    vertical_positions = [
        round(
            i
            * (w - 1)
            / GRID_SIZE
        )
        for i in range(
            GRID_SIZE + 1
        )
    ]

    horizontal_positions = [
        round(
            i
            * (h - 1)
            / GRID_SIZE
        )
        for i in range(
            GRID_SIZE + 1
        )
    ]

    thin_thickness = max(
        1,
        round(
            min(h, w)
            / 450
        ),
    )

    thick_thickness = max(
        2,
        round(
            min(h, w)
            / 250
        ),
    )

    # --------------------------------------------------------
    # Vertical lines
    # --------------------------------------------------------

    for index, x in enumerate(
        vertical_positions
    ):
        is_block_line = (
            index % 3 == 0
        )

        thickness = (
            thick_thickness
            if is_block_line
            else thin_thickness
        )

        cv2.line(
            result,
            (x, 0),
            (x, h - 1),
            0,
            thickness,
        )

    # --------------------------------------------------------
    # Horizontal lines
    # --------------------------------------------------------

    for index, y in enumerate(
        horizontal_positions
    ):
        is_block_line = (
            index % 3 == 0
        )

        thickness = (
            thick_thickness
            if is_block_line
            else thin_thickness
        )

        cv2.line(
            result,
            (0, y),
            (w - 1, y),
            0,
            thickness,
        )

    return result


# ============================================================
# EXTRACT CELLS + REBUILD IMAGE
# ============================================================

def extract_cells(
    binary,
    vertical_lines,
    horizontal_lines,
):
    """
    Hücreleri çıkarır, temizler ve tekrar
    9x9 Sudoku görüntüsü halinde birleştirir.

    Returns:
        rebuilt Sudoku image
    """

    cell_images = extract_cell_images(
        binary,
        vertical_lines,
        horizontal_lines,
    )

    # --------------------------------------------------------
    # Her satırdaki 9 hücreyi birleştir
    # --------------------------------------------------------

    cell_rows = []

    for row_cells in cell_images:
        cell_row = cv2.hconcat(
            row_cells
        )

        cell_rows.append(
            cell_row
        )

    # --------------------------------------------------------
    # 9 satırı birleştir
    # --------------------------------------------------------

    rebuilt = cv2.vconcat(
        cell_rows
    )

    # --------------------------------------------------------
    # Rebuilt Sudoku grid
    # --------------------------------------------------------

    rebuilt = add_rebuilt_grid(
        rebuilt
    )

    return rebuilt