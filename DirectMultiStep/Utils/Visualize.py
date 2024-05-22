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
from rdkit import Chem
from rdkit.Chem import Draw


class MoleculeNode:
    def __init__(self, identity, smiles, template_index=None, parent=None):
        self.identity = identity
        self.smiles = smiles
        self.template_index = template_index
        self.parent = parent
        self.children = []  # List of TemplateNodes
        self.step = 0
        self.info = {}

    def get_mol(self):
        return Chem.MolFromSmiles(self.smiles)

    def add_template_node(self, template_node):
        assert isinstance(template_node, TemplateNode)
        template_node.step = template_node.parent.step + 1
        self.children.append(template_node)
        return template_node

    def add_info(self, key, value):
        self.info[key] = value


class TemplateNode:
    def __init__(self, identity, template_smarts, parent=None):
        self.identity = identity
        self.template_index = identity
        self.template_smarts = template_smarts
        self.parent = parent
        self.children = []  # List of MoleculeNodes
        self.step = 0
        self.info = {}

    def add_molecule_node(self, molecule_node):
        assert isinstance(molecule_node, MoleculeNode)
        molecule_node.step = molecule_node.parent.step
        self.children.append(molecule_node)
        return molecule_node

    def add_info(self, key, value):
        self.info[key] = value


class RetrosynthesisTree:
    def __init__(self, root_smiles):
        self.root = MoleculeNode(identity=0, smiles=root_smiles)
        self.mol_last_identity = 0
        self.temp_last_identity = -1
        self.all_template_mapping = []

    def add_mol_node(self, parent_identity: int, child_smiles: str):
        parent_node = self.find_temp_node(self.root, parent_identity)
        if parent_node is not None:
            self.mol_last_identity += 1
            child_identity = self.mol_last_identity  # Using new id for child
            chile_node = MoleculeNode(
                identity=child_identity,
                smiles=child_smiles,
                template_index=parent_node.identity,
                parent=parent_node,
            )
            return parent_node.add_molecule_node(chile_node)
        else:
            return None

    def add_temp_node(self, parent_identity, template_smarts):
        parent_node = self.find_mol_node(self.root, parent_identity)
        if parent_node is not None:
            self.temp_last_identity += 1
            child_identity = self.temp_last_identity  # Using new id for child
            chile_node = TemplateNode(
                identity=child_identity,
                template_smarts=template_smarts,
                parent=parent_node,
            )
            self.all_template_mapping.append(template_smarts)
            return parent_node.add_template_node(chile_node)
        else:
            return None

    def find_mol_node(self, node, identity: int):
        if isinstance(node, MoleculeNode) and node.identity == identity:
            return node
        for child in node.children:
            found_node = self.find_mol_node(child, identity)
            if found_node:
                return found_node
        return None

    def find_temp_node(self, node, identity: int):
        if isinstance(node, TemplateNode) and node.identity == identity:
            return node
        for child in node.children:
            found_node = self.find_temp_node(child, identity)
            if found_node:
                return found_node
        return None

    def display(self, node=None, level=0, verbose=0, chosen_index=[]):
        if node is None:
            node = self.root
        if isinstance(node, MoleculeNode):
            indent = "-" * level
            print(
                f"{indent}{node.smiles} (Mol Identity: {node.identity}, Template Index: {node.template_index})"
            )
            for child in node.children:
                self.display(
                    child, level + 1, verbose=verbose, chosen_index=chosen_index
                )
        elif isinstance(node, TemplateNode):
            indent = "*" * level
            print(f"{indent} (Temp Identity: {node.identity})")
            for child in node.children:
                self.display(child, level, verbose=verbose, chosen_index=chosen_index)

    def serialize_single(self, current_node):
        """
        Serialize single path.
        """
        path_dict = {}
        path_dict["smiles"] = current_node.smiles
        if current_node.children:
            path_dict["children"] = []
        for temp_node in current_node.children:
            for mol_node in temp_node.children:
                path_dict["children"].append(self.serialize_single(mol_node))
        return path_dict

    def serialize_single_path(self, current_node, path=[]):
        """
        Serialize single path that have template identities in PATH.
        """
        path_dict = {}
        path_dict["smiles"] = current_node.smiles
        if current_node.children:
            path_dict["children"] = []
        for temp_node in current_node.children:
            if temp_node.identity in path:
                for mol_node in temp_node.children:
                    path_dict["children"].append(
                        self.serialize_single_path(mol_node, path)
                    )
        if "children" in path_dict and path_dict["children"] == []:
            del path_dict["children"]
        return path_dict


def store_in_node(retro_tree, parent_identity, path_dict):
    if "children" in path_dict:
        template_node = retro_tree.add_temp_node(
            parent_identity=parent_identity, template_smarts=""
        )
        template_identity = template_node.identity
        for child in path_dict["children"]:
            mol_node = retro_tree.add_mol_node(
                parent_identity=template_identity, child_smiles=child["smiles"]
            )
            store_in_node(retro_tree, mol_node.identity, child)


def draw_rounded_rectangle(ctx, x, y, width, height, corner_radius):
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


def compute_subtree_dimensions(node, img_width, img_height, y_offset, path):
    """Compute the dimensions of the subtree rooted at the given node."""
    if not node.children:
        return img_width, img_height + y_offset

    width = img_width
    height = img_height + y_offset

    for child in node.children:
        if not path or child.template_index in path:
            child_width, child_height = compute_subtree_dimensions(
                child, img_width, img_height, y_offset, path
            )
            width += child_width + img_width
            height = max(height, child_height + img_height + y_offset)
    return width, height + 60


def compute_canvas_dimensions(root, img_width, img_height, y_offset, path):
    """Compute the dimensions of the canvas based on the tree structure."""
    # Compute the dimensions of the subtree rooted at each child
    if not path:
        child_dimensions = [
            compute_subtree_dimensions(child, img_width, img_height, y_offset, path)
            for child in root.children
        ]
    else:
        child_dimensions = [
            compute_subtree_dimensions(child, img_width, img_height, y_offset, path)
            for child in root.children
            if child.template_index in path
        ]
    # Compute the width and height of the canvas
    width = sum(dim[0] for dim in child_dimensions) + img_width * len(child_dimensions)
    height = (
        max((dim[1] for dim in child_dimensions), default=0) + img_height + y_offset
    )
    return width, height


def check_overlap(new_x, new_y, existing_boxes, img_width, img_height):
    """Check if a new box would overlap with any existing boxes."""
    for x, y in existing_boxes:
        if (x - img_width < new_x < x + img_width) and (
            y - img_height < new_y < y + img_height
        ):
            return True
    return False


def draw_molecule_tree(
    root,
    path=None,
    x_offset=600,
    y_offset=600,
    img_size=(400, 400),
    filename="molecule_tree.svg",
    use_rank=False,
):
    # Compute the dimensions of the canvas
    img_width, img_height = img_size
    canvas_width, canvas_height = compute_canvas_dimensions(
        root, img_width, img_height, y_offset, path
    )

    # Create the SVG surface and cairo context
    surface = cairo.SVGSurface(filename, canvas_width, canvas_height)
    ctx = cairo.Context(surface)

    existing_boxes = []

    def draw_node(node, x, y):
        # Check for overlap and adjust position
        while check_overlap(
            x, y, existing_boxes, img_width, img_height
        ) or check_overlap(x, y - y_offset, existing_boxes, img_width, img_height):
            x += img_width // 2

        existing_boxes.append((x, y))

        if isinstance(node, MoleculeNode):
            # Draw molecule at the current position (x, y)
            mol_img = Draw.MolToImage(node.get_mol(), size=img_size)
            mol_img.save("temp.png")
            img_surface = cairo.ImageSurface.create_from_png("temp.png")
            ctx.set_source_surface(img_surface, x, y)
            ctx.paint()

            # Draw rounded rectangle around the molecule
            draw_rounded_rectangle(ctx, x, y, img_width, img_height, 20)
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
            id_text = f"ID: {node.identity}"
            ctx.move_to(x, y + img_height + 40)
            ctx.show_text(id_text)

            # Score to the right
            # score_text = f"Score: {score}"
            # (x_bearing, y_bearing, width, height, x_advance, y_advance) = ctx.text_extents(score_text)
            # ctx.move_to(x + img_width - width, y + img_height + 40)
            # ctx.show_text(score_text)

        elif isinstance(node, TemplateNode):
            # Draw circle
            radius = img_width / 4  # Reduced radius for TemplateNode
            ctx.arc(x + img_width / 2, y + img_height / 2, radius, 0, 2 * 3.14159)
            ctx.set_source_rgb(0.5, 0.5, 0.5)
            ctx.fill()

            # Template identity inside the circle
            ctx.set_source_rgb(1, 1, 1)
            ctx.select_font_face(
                "Arial", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD
            )
            ctx.set_font_size(40)
            if use_rank:
                # count the rank of the template from their parent
                top = node.parent.children.index(node) + 1
                (x_bearing, y_bearing, width, height, x_advance, y_advance) = (
                    ctx.text_extents(str(top))
                )
            else:
                # template identity/index
                (x_bearing, y_bearing, width, height, x_advance, y_advance) = (
                    ctx.text_extents(str(node.identity))
                )
            ctx.move_to(
                x + img_width / 2 - width / 2 - x_bearing,
                y + img_height / 2 - height / 2 - y_bearing,
            )
            if use_rank:
                ctx.show_text(str(top))
            else:
                ctx.show_text(str(node.identity))

        # Draw children
        included_children = [
            child for child in node.children if not path or child.template_index in path
        ]
        children_count = len(included_children)

        if children_count == 1:
            new_x = x
        else:
            new_x = x - (children_count - 1) * (x_offset + img_width) // 2

        new_y = y + y_offset + img_height  # y-coordinate for the children

        for child in included_children:
            # Draw line from parent to child
            if isinstance(node, MoleculeNode):
                ctx.move_to(x + img_width / 2, y + img_height)
                line_to_x = new_x + img_width / 2
                line_to_y = new_y
            elif isinstance(node, TemplateNode):
                ctx.move_to(x + img_width / 2, y + img_height / 2 + img_width / 4)
                line_to_x = new_x + img_width / 2
                line_to_y = new_y + img_height / 2 - img_width / 4

            if isinstance(child, MoleculeNode):
                line_to_y = new_y
            elif isinstance(child, TemplateNode):
                line_to_y = new_y + img_height / 2 - img_width / 4

            ctx.line_to(line_to_x, line_to_y)
            ctx.set_line_width(4)
            ctx.set_source_rgb(0, 0, 0)
            ctx.stroke()
            # Draw child and its descendants
            draw_node(child, new_x, new_y)
            new_x += x_offset + img_width

    # Draw the root and its descendants
    draw_node(root, (canvas_width - img_width) // 2, 50)

    # Finish the drawing
    surface.finish()
    os.remove("temp.png")

import cairosvg
from pathlib import Path
def draw_tree_from_path_string(path_string: str, save_path: Path):
    assert save_path.suffix == '', "Please provide a path without extension"
    retro_tree = RetrosynthesisTree(root_smiles=eval(path_string)["smiles"])
    store_in_node(retro_tree, parent_identity=0, path_dict=eval(path_string))
    draw_molecule_tree(retro_tree.root, filename=str(save_path.with_suffix('.svg')), x_offset=50, y_offset=10)
    cairosvg.svg2pdf(url=str(save_path.with_suffix('.svg')), write_to=str(save_path.with_suffix('.pdf')))
    os.remove(save_path.with_suffix('.svg'))