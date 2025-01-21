import base64
from pathlib import Path
from typing import Literal, NamedTuple, TypeAlias, cast

import svgwrite
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from reportlab.graphics import renderPDF
from svglib.svglib import svg2rlg

FilteredDict: TypeAlias = dict[str, str | list["FilteredDict"]]
ThemeType = Literal["light", "dark"]

__support_pdf__ = True


class ColorPalette(NamedTuple):
    """Defines a color palette for drawing molecules.

    Attributes:
        atom_colors: A dictionary mapping atomic numbers to RGB color tuples.
        annotation: An RGBA color tuple for annotations.
        border: An RGB color tuple for borders.
        text: An RGB color tuple for text.
        background: An RGBA color tuple for background.
    """

    atom_colors: dict[int, tuple[float, float, float]]
    annotation: tuple[float, float, float, float]
    border: tuple[float, float, float]
    text: tuple[float, float, float]
    background: tuple[float, float, float, float]


DARK_PALETTE = ColorPalette(
    atom_colors={
        0: (0.9, 0.9, 0.9),
        1: (0.9, 0.9, 0.9),
        6: (0.9, 0.9, 0.9),
        7: (0.33, 0.41, 0.92),
        8: (1.0, 0.2, 0.2),
        9: (0.2, 0.8, 0.8),
        15: (1.0, 0.5, 0.0),
        16: (0.8, 0.8, 0.0),
        17: (0.0, 0.802, 0.0),
        35: (0.71, 0.4, 0.07),
        53: (0.89, 0.004, 1),
        201: (0.68, 0.85, 0.90),
    },
    annotation=(1, 1, 1, 1),
    border=(0.89, 0.91, 0.94),
    text=(0.81, 0.81, 0.81),
    background=(0, 0, 0, 1),
)

LIGHT_PALETTE = ColorPalette(
    atom_colors={
        0: (0.9, 0.9, 0.9),
        1: (0.9, 0.9, 0.9),
        7: (0.33, 0.41, 0.92),
        8: (1.0, 0.2, 0.2),
        9: (0.2, 0.8, 0.8),
        15: (1.0, 0.5, 0.0),
        16: (0.8, 0.8, 0.0),
        17: (0.0, 0.802, 0.0),
        35: (0.71, 0.4, 0.07),
        53: (0.89, 0.004, 1),
        201: (0.68, 0.85, 0.90),
    },
    annotation=(0, 0, 0, 1),
    border=(0.12, 0.16, 0.23),
    text=(0.2, 0.2, 0.2),
    background=(1, 1, 1, 1),
)


class RetroSynthesisTree:
    """Basic tree structure for retrosynthesis visualization."""

    def __init__(self, idx: int = 0) -> None:
        """
        Initializes a new node in the retrosynthesis tree.

        Args:
            idx: The unique identifier for the node.
        """
        self.node_id = idx
        self.smiles = ""
        self.children: list[RetroSynthesisTree] = []

    def build_tree(self, path_dict: FilteredDict) -> int:
        """Recursively builds the retrosynthesis tree from a dictionary.

        Args:
            path_dict: A dictionary representing the tree structure.

        Returns:
            The next available node ID.
        """
        self.smiles = cast(str, path_dict["smiles"])
        cur_id = self.node_id + 1

        if "children" in path_dict:
            for child in cast(list[FilteredDict], path_dict["children"]):
                node = RetroSynthesisTree(idx=cur_id)
                cur_id = node.build_tree(path_dict=child)
                self.children.append(node)
        return cur_id

    def __str__(self) -> str:
        """Returns a string representation of the tree node and its children."""
        child_ids = [str(child.node_id) for child in self.children]
        return f"Node ID: {self.node_id}, Children: {child_ids}, SMILES: {self.smiles}\n" + "".join(
            str(child) for child in self.children
        )


class TreeDimensions(NamedTuple):
    """Represents the dimensions of a tree or subtree."""

    width: int
    height: int


def compute_subtree_dimensions(
    tree: RetroSynthesisTree, img_width: int, img_height: int, y_offset: int
) -> TreeDimensions:
    """Compute dimensions of a subtree for layout.

    Args:
        tree: The subtree to compute dimensions for.
        img_width: The width of the molecule image.
        img_height: The height of the molecule image.
        y_offset: The vertical offset between nodes.

    Returns:
        The dimensions of the subtree.
    """
    if not tree.children:
        return TreeDimensions(img_width, img_height + y_offset)

    width = img_width
    height = img_height + y_offset

    for child in tree.children:
        child_dims = compute_subtree_dimensions(child, img_width, img_height, y_offset)
        width += child_dims.width
        height = max(height, child_dims.height + img_height + y_offset)

    return TreeDimensions(width, height)


def compute_canvas_dimensions(
    tree: RetroSynthesisTree, img_width: int, img_height: int, y_offset: int
) -> TreeDimensions:
    """Compute overall canvas dimensions.

    Args:
        tree: The retrosynthesis tree.
        img_width: The width of the molecule image.
        img_height: The height of the molecule image.
        y_offset: The vertical offset between nodes.

    Returns:
        The dimensions of the canvas.
    """
    child_dims = [compute_subtree_dimensions(child, img_width, img_height, y_offset) for child in tree.children]
    width = sum(d.width for d in child_dims)
    height = max((d.height for d in child_dims), default=0) + img_height + y_offset
    return TreeDimensions(width, height + 100)


def check_overlap(
    new_x: int,
    new_y: int,
    existing_boxes: list[tuple[int, int]],
    img_width: int,
    img_height: int,
) -> bool:
    """Check if a new node overlaps with existing nodes.

    Args:
        new_x: The x-coordinate of the new node.
        new_y: The y-coordinate of the new node.
        existing_boxes: A list of tuples representing the coordinates of existing nodes.
        img_width: The width of the molecule image.
        img_height: The height of the molecule image.

    Returns:
        True if there is an overlap, False otherwise.
    """
    return any(
        (x - img_width < new_x < x + img_width) and (y - img_height < new_y < y + img_height) for x, y in existing_boxes
    )


def draw_molecule(smiles: str, size: tuple[int, int], theme: ThemeType) -> str:
    """Render a SMILES string as base64-encoded PNG.

    Args:
        smiles: The SMILES string of the molecule.
        size: The desired size (width, height) of the image.
        theme: The color theme ("light" or "dark").

    Returns:
        The base64-encoded PNG image data.

    Raises:
        ValueError: If the SMILES string is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    draw_width, draw_height = size
    drawer = rdMolDraw2D.MolDraw2DCairo(draw_width, draw_height)
    opts = drawer.drawOptions()

    palette = DARK_PALETTE if theme == "dark" else LIGHT_PALETTE
    background_color = palette.background
    if not __support_pdf__:
        background_color = (
            background_color[0],
            background_color[1],
            background_color[2],
            0,
        )
    opts.setBackgroundColour(background_color)
    opts.setAtomPalette(palette.atom_colors)
    opts.setAnnotationColour(palette.annotation)

    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    png_data = drawer.GetDrawingText()
    return base64.b64encode(png_data).decode("utf-8")


def draw_tree_svg(
    tree: RetroSynthesisTree,
    width: int,
    height: int,
    x_margin: int,
    y_margin: int,
    theme: ThemeType,
    force_canvas_width: int | None = None,
) -> str:
    """Create SVG visualization of the retrosynthesis tree.

    Args:
        tree: The retrosynthesis tree to visualize.
        width: The width of each molecule image.
        height: The height of each molecule image.
        x_margin: The horizontal margin between nodes.
        y_margin: The vertical margin between nodes.
        theme: The color theme ("light" or "dark").
        force_canvas_width: An optional width to force for the canvas.

    Returns:
        The SVG content as a string.
    """

    initial_dims = compute_canvas_dimensions(tree, width, height, y_margin)
    canvas_width = force_canvas_width if force_canvas_width else initial_dims.width
    drawing = svgwrite.Drawing(size=(canvas_width, initial_dims.height))

    existing_boxes: list[tuple[int, int]] = []
    memo = {"left_x": float("inf"), "right_x": float("-inf")}

    def draw_node(node: RetroSynthesisTree, nx: int, ny: int) -> None:
        """Draws a single node of the retrosynthesis tree.

        Args:
            node: The tree node to draw.
            nx: The x-coordinate of the node.
            ny: The y-coordinate of the node.
        """
        while check_overlap(nx, ny, existing_boxes, width, height) or check_overlap(
            nx, ny - y_margin, existing_boxes, width, height
        ):
            nx += width // 2

        existing_boxes.append((nx, ny))
        memo["left_x"] = min(memo["left_x"], nx - width // 2)
        memo["right_x"] = max(memo["right_x"], nx + width // 2)

        # Draw molecule
        b64_img = draw_molecule(node.smiles, (width, height), theme)
        drawing.add(
            drawing.image(
                href=f"data:image/png;base64,{b64_img}",
                insert=(nx, ny),
                size=(width, height),
            )
        )

        # Draw border
        palette = DARK_PALETTE if theme == "dark" else LIGHT_PALETTE
        border_color = svgwrite.rgb(*[c * 255 for c in palette.border])

        box = drawing.rect(
            insert=(nx, ny),
            size=(width, height),
            rx=20,
            ry=20,
            fill="none",
            stroke=border_color,
            stroke_width=4,
        )
        drawing.add(box)

        # Draw node ID
        text_color = svgwrite.rgb(*[c * 255 for c in palette.text])
        node_label = drawing.text(
            f"ID: {node.node_id}",
            insert=(nx, ny + height + 35),
            fill=text_color,
            font_size=20,
            font_family="Arial",
        )
        drawing.add(node_label)

        # Draw children
        child_count = len(node.children)
        if child_count > 0:
            next_x = nx if child_count == 1 else nx - (child_count - 1) * width // 2
            next_y = ny + y_margin + height

            for child in node.children:
                # Draw connecting line
                line = drawing.line(
                    start=(nx + width / 2, ny + height),
                    end=(next_x + width / 2, next_y),
                    stroke=border_color,
                    stroke_width=4,
                )
                drawing.add(line)

                draw_node(child, next_x, next_y)
                next_x += x_margin + width

    # Draw the root
    root_x = (canvas_width - width) // 2
    draw_node(tree, root_x, 50)

    # Adjust canvas if needed
    final_width = int(memo["right_x"] - memo["left_x"] + width * 2 + x_margin * 2)
    if final_width > canvas_width and force_canvas_width is None:
        return draw_tree_svg(
            tree,
            width,
            height,
            x_margin,
            y_margin,
            theme,
            force_canvas_width=final_width,
        )

    return cast(str, drawing.tostring())


def create_tree_from_path_string(path_string: str) -> RetroSynthesisTree:
    """Parse a dictionary-like string into a RetroSynthesisTree.

    Args:
        path_string: A string representing the tree structure as a dictionary.

    Returns:
        A RetroSynthesisTree object.
    """
    path_dict: FilteredDict = eval(path_string)  # TODO: Use safer parsing
    retro_tree = RetroSynthesisTree()
    retro_tree.build_tree(path_dict=path_dict)
    return retro_tree


def draw_tree_from_path_string(
    path_string: str,
    save_path: Path,
    width: int = 400,
    height: int = 400,
    x_margin: int = 50,
    y_margin: int = 100,
    theme: str = "light",
) -> str:
    """Generate SVG and PDF visualizations from a path string.

    Args:
        path_string: A string representing the tree structure as a dictionary.
        save_path: The path to save the generated SVG and PDF files.
        width: The width of each molecule image.
        height: The height of each molecule image.
        x_margin: The horizontal margin between nodes.
        y_margin: The vertical margin between nodes.
        theme: The color theme ("light" or "dark").

    Returns:
        The SVG content as a string.
    """
    assert theme in ["light", "dark"]
    theme = cast(ThemeType, theme)

    retro_tree = create_tree_from_path_string(path_string)
    svg_content = draw_tree_svg(
        retro_tree,
        width=width,
        height=height,
        x_margin=x_margin,
        y_margin=y_margin,
        theme=theme,
    )

    svg_path = save_path.with_suffix(".svg")
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg_content)

    # Convert to PDF
    drawing = svg2rlg(str(svg_path))
    renderPDF.drawToFile(drawing, str(save_path.with_suffix(".pdf")))
    # remove SVG file
    svg_path.unlink()
    return svg_content


if __name__ == "__main__":
    path = "{'smiles':'O=C(c1ccc(NS(=O)(=O)c2cccc3cccnc23)cc1)N1CCN(CC2CC2)CC1','children':[{'smiles':'O=C(O)c1ccc(NS(=O)(=O)c2cccc3cccnc23)cc1','children':[{'smiles':'CCOC(=O)c1ccc(NS(=O)(=O)c2cccc3cccnc23)cc1','children':[{'smiles':'CCOC(=O)c1ccc(N)cc1'},{'smiles':'O=S(=O)(Cl)c1cccc2cccnc12'}]}]},{'smiles':'C1CN(CC2CC2)CCN1'}]}"

    svg_str = draw_tree_from_path_string(
        path_string=path,
        save_path=Path("data/figures/modern_svg_light_clean"),
        width=400,
        height=400,
        x_margin=50,
        y_margin=100,
        theme="light",
    )
