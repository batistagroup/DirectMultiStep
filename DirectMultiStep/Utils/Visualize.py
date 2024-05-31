# MIT License

# Copyright (c) 2024 Batista Lab (Yale University)

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import cairo
from rdkit import Chem # type: ignore
from rdkit.Chem import Draw # type: ignore
import cairosvg # type: ignore
from pathlib import Path
from typing import Dict, List, Tuple, Union, cast

FilteredDict = Dict[str, Union[str, List["FilteredDict"]]]


class RetroSynthesisTree:
    def __init__(self, idx: int = 0) -> None:
        self.node_id = idx
        self.smiles: str
        self.children: List[RetroSynthesisTree] = []

    def build_tree(self, path_dict: FilteredDict) -> int:
        self.smiles = cast(str, path_dict["smiles"])
        cur_id = self.node_id
        cur_id += 1
        if "children" in path_dict:
            for child in cast(FilteredDict, path_dict["children"]):
                node = RetroSynthesisTree(idx=cur_id)
                cur_id = node.build_tree(path_dict=cast(FilteredDict, child))
                self.children.append(node)
        return cur_id

    def __str__(self) -> str:
        child_ids = [child.node_id for child in self.children]
        header = (
            f"Node ID: {self.node_id}, Children: {child_ids}, SMILES: {self.smiles}\n"
        )
        body = ""
        for child in self.children:
            body += str(child)
        return header + body


def create_tree_from_path_string(path_string: str) -> RetroSynthesisTree:
    path_dict: FilteredDict = eval(path_string)
    retro_tree = RetroSynthesisTree()
    retro_tree.build_tree(path_dict=path_dict)
    # print(retro_tree)
    return retro_tree


def compute_subtree_dimensions(
    tree: "RetroSynthesisTree", img_width: int, img_height: int, y_offset: int
) -> Tuple[int, int]:
    """Compute the dimensions of the subtree rooted at the given node."""
    if not tree.children:
        return img_width, img_height + y_offset

    width = img_width
    height = img_height + y_offset

    for child in tree.children:
        child_width, child_height = compute_subtree_dimensions(
            child, img_width, img_height, y_offset
        )
        width += child_width + img_width
        height = max(height, child_height + img_height + y_offset)
    return width, height


def compute_canvas_dimensions(
    tree: "RetroSynthesisTree", img_width: int, img_height: int, y_offset: int
) -> Tuple[int, int]:
    """Compute the dimensions of the canvas based on the tree structure."""
    # Compute the dimensions of the subtree rooted at each child
    child_dimensions = [
        compute_subtree_dimensions(child, img_width, img_height, y_offset)
        for child in tree.children
    ]
    # Compute the width and height of the canvas
    width = sum(dim[0] for dim in child_dimensions) + img_width * len(child_dimensions)
    height = (
        max((dim[1] for dim in child_dimensions), default=0) + img_height + y_offset
    )
    return width, height + 100


def check_overlap(
    new_x: int,
    new_y: int,
    existing_boxes: List[Tuple[int, int]],
    img_width: int,
    img_height: int,
) -> bool:
    """Check if a new box would overlap with any existing boxes."""
    for x, y in existing_boxes:
        if (x - img_width < new_x < x + img_width) and (
            y - img_height < new_y < y + img_height
        ):
            return True
    return False


def draw_rounded_rectangle(
    ctx: cairo.Context, x: int, y: int, width: int, height: int, corner_radius: int # type: ignore
) -> None:
    """Draws a rounded rectangle."""
    ctx.new_sub_path()
    ctx.arc(
        x + width - corner_radius, y + corner_radius, corner_radius, -0.5 * 3.14159, 0
    )
    ctx.arc(
        x + width - corner_radius,
        y + height - corner_radius,
        corner_radius,
        0,
        0.5 * 3.14159,
    )
    ctx.arc(
        x + corner_radius,
        y + height - corner_radius,
        corner_radius,
        0.5 * 3.14159,
        3.14159,
    )
    ctx.arc(x + corner_radius, y + corner_radius, corner_radius, 3.14159, 1.5 * 3.14159)
    ctx.close_path()


def draw_molecule_tree(
    tree: "RetroSynthesisTree",
    filename: str,
    width: int = 400,
    height: int = 400,
    x_margin: int = 50,
    y_margin: int = 50,
)->None:
    canvas_width, canvas_height = compute_canvas_dimensions(
        tree, width, height, y_margin
    )
    surface = cairo.SVGSurface(filename, canvas_width, canvas_height)
    ctx = cairo.Context(surface)

    existing_boxes: List[Tuple[int, int]] = []

    def draw_node(node:'RetroSynthesisTree', x:int, y:int)->None:
        # Check for overlap and adjust position
        while check_overlap(x, y, existing_boxes, width, height) or check_overlap(
            x, y - y_margin, existing_boxes, width, height
        ):
            x += width // 2

        existing_boxes.append((x, y))

        if isinstance(node, RetroSynthesisTree):
            # Draw molecule at the current position (x, y)
            mol = Chem.MolFromSmiles(node.smiles)
            mol_img = Draw.MolToImage(mol, size=(width, height))
            mol_img.save("temp.png")
            img_surface = cairo.ImageSurface.create_from_png("temp.png")
            ctx.set_source_surface(img_surface, x, y)
            ctx.paint()

            # Draw rounded rectangle around the molecule
            draw_rounded_rectangle(ctx, x, y, width, height, 20)
            ctx.set_line_width(4)
            ctx.set_source_rgb(0, 0, 1)
            ctx.stroke()

            # Additional information
            # score = round(node.info["Score"], 3)
            ctx.set_source_rgb(0, 0, 0)  # Black
            ctx.select_font_face(
                "Arial", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL
            )
            ctx.set_font_size(30)

            # ID to the left
            id_text = f"ID: {node.node_id}"
            ctx.move_to(x, y + height + 40)
            ctx.show_text(id_text)

            # Score to the right
            # score_text = f"Score: {score}"
            # (x_bearing, y_bearing, width, height, x_advance, y_advance) = ctx.text_extents(score_text)
            # ctx.move_to(x + img_width - width, y + img_height + 40)
            # ctx.show_text(score_text)

        # Draw children
        included_children = [child for child in node.children]
        children_count = len(included_children)

        if children_count == 1:
            new_x = x
        else:
            new_x = x - (children_count - 1) * (x_margin + width) // 2

        new_y = y + y_margin + height  # y-coordinate for the children

        for child in included_children:
            # Draw line from parent to child
            if isinstance(node, RetroSynthesisTree):
                ctx.move_to(x + width / 2, y + height)
                line_to_x = new_x + width / 2
                line_to_y = new_y

            if isinstance(child, RetroSynthesisTree):
                line_to_y = new_y

            ctx.line_to(line_to_x, line_to_y)
            ctx.set_line_width(4)
            ctx.set_source_rgb(0, 0, 0)
            ctx.stroke()
            # Draw child and its descendants
            draw_node(child, new_x, new_y)
            new_x += x_margin + width

    # Draw the root and its descendants
    draw_node(node=tree, x=(canvas_width - width) // 2, y=50)

    # Finish the drawing
    surface.finish()
    os.remove("temp.png")


def draw_tree_from_path_string(path_string: str, save_path: Path, width: int = 400, height: int = 400, x_margin: int = 50, y_margin: int = 100)->None:
    assert save_path.suffix == "", "Please provide a path without extension"
    retro_tree = create_tree_from_path_string(path_string=path_string)

    draw_molecule_tree(
        retro_tree,
        filename=str(save_path.with_suffix(".svg")),
        x_margin=x_margin,
        y_margin=y_margin,
        width=width,
        height=height,
    )
    cairosvg.svg2pdf(
        url=str(save_path.with_suffix(".svg")),
        write_to=str(save_path.with_suffix(".pdf")),
    )
    os.remove(save_path.with_suffix(".svg"))


if __name__ == "__main__":
    path = "{'smiles':'O=C(c1ccc(NS(=O)(=O)c2cccc3cccnc23)cc1)N1CCN(CC2CC2)CC1','children':[{'smiles':'O=C(O)c1ccc(NS(=O)(=O)c2cccc3cccnc23)cc1','children':[{'smiles':'CCOC(=O)c1ccc(NS(=O)(=O)c2cccc3cccnc23)cc1','children':[{'smiles':'CCOC(=O)c1ccc(N)cc1'},{'smiles':'O=S(=O)(Cl)c1cccc2cccnc12'}]}]},{'smiles':'C1CN(CC2CC2)CCN1'}]}"
    create_tree_from_path_string(path_string=path)
