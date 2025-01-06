import matplotlib.pyplot as plt
import matplotlib.patches as patches
import yaml
from dataclasses import dataclass, field
from typing import List, Tuple, Union, Optional

@dataclass
class Plot:
    width: int
    length: int
    extra_space: int

    @property
    def new_width(self):
        return self.width + 2 * self.extra_space

    @property
    def new_length(self):
        return self.length + 2 * self.extra_space

@dataclass
class Structure:
    name: str
    points: List[Tuple[float, float]]
    edgecolor: str = "red"
    linewidth: float = 1.5

@dataclass
class ShapedStructure:
    name: str
    shape: str  # 'rectangle', 'square', 'circle'
    position: Tuple[float, float]
    dimensions: Union[Tuple[float, float], float]  # (length, breadth) or radius
    alignment: Optional[str] = "center"  # 'center', 'top-left', etc. (for rectangles/squares)
    edgecolor: str = "blue"
    linewidth: float = 1.5

    def calculate_points(self) -> List[Tuple[float, float]]:
        x, y = self.position

        if self.shape.lower() == "rectangle":
            if not isinstance(self.dimensions, tuple) or len(self.dimensions) != 2:
                raise ValueError("Rectangle dimensions must be a tuple of (length, breadth).")
            length, breadth = self.dimensions
            return self._rectangle_points(x, y, length, breadth, self.alignment)

        elif self.shape.lower() == "square":
            if not isinstance(self.dimensions, (int, float)):
                raise ValueError("Square dimensions must be a single value for side length.")
            side = self.dimensions
            return self._rectangle_points(x, y, side, side, self.alignment)

        else:
            raise ValueError(f"Unsupported shape: {self.shape}")

    def _rectangle_points(self, x, y, length, breadth, alignment) -> List[Tuple[float, float]]:
        if alignment == "center":
            return [
                (x - length / 2, y - breadth / 2),
                (x - length / 2, y + breadth / 2),
                (x + length / 2, y + breadth / 2),
                (x + length / 2, y - breadth / 2)
            ]
        elif alignment == "top-left":
            return [
                (x, y),
                (x, y - breadth),
                (x + length, y - breadth),
                (x + length, y)
            ]
        elif alignment == "top-right":
            return [
                (x, y),
                (x, y - breadth),
                (x - length, y - breadth),
                (x - length, y)
            ]
        elif alignment == "bottom-left":
            return [
                (x, y),
                (x, y + breadth),
                (x + length, y + breadth),
                (x + length, y)
            ]
        elif alignment == "bottom-right":
            return [
                (x, y),
                (x, y + breadth),
                (x - length, y + breadth),
                (x - length, y)
            ]
        else:
            raise ValueError(f"Invalid alignment: {alignment}")

@dataclass
class Layout:
    plot: Plot
    structures: List[Union[Structure, ShapedStructure]] = field(default_factory=list)

    def add_structure(self, structure: Union[Structure, ShapedStructure]):
        self.structures.append(structure)

def draw_layout(layout: Layout, filename: str):
    aspect_ratio = layout.plot.new_width / layout.plot.new_length
    fig, ax = plt.subplots(figsize=(11 / aspect_ratio, 11))

    # Gridlines
    major_xticks = range(-5, layout.plot.new_length + 1, 5)  # Major ticks every 10 feet
    major_yticks = range(-5, layout.plot.new_width + 1, 5)
    minor_xticks = range(-5, layout.plot.new_length + 1)
    minor_yticks = range(-5, layout.plot.new_width + 1)

    ax.set_xticks(major_xticks)
    ax.set_yticks(major_yticks)
    ax.set_xticks(minor_xticks, minor=True)
    ax.set_yticks(minor_yticks, minor=True)
    ax.grid(which="minor", linestyle=":", linewidth=0.5, color="gray")
    ax.grid(which="major", linestyle="-", linewidth=0.8, color="black")

    # Limits and labels
    ax.set_xlim(-5, layout.plot.new_length-5)
    ax.set_ylim(-5, layout.plot.new_width-5)
    ax.set_xlabel("Length (feet)", fontsize=10)
    ax.set_ylabel("Width (feet)", fontsize=10)
    ax.tick_params(axis="x", which="major", labelsize=8, rotation=90)
    ax.tick_params(axis="y", which="major", labelsize=8)
    ax.tick_params(axis="both", which="minor", labelsize=6)

    # Plot boundary
    ax.add_patch(
        patches.Rectangle(
            (0, 0), layout.plot.length, layout.plot.width,
            fill=None, edgecolor="blue", linewidth=1.5, label="Plot Boundary"
        )
    )

    # Add structures
    for structure in layout.structures:
        if isinstance(structure, ShapedStructure):
            if structure.shape.lower() == "circle":
                x, y = structure.position
                radius = structure.dimensions
                circle = patches.Circle(
                    (x, y),
                    radius=radius,
                    edgecolor=structure.edgecolor,
                    linewidth=structure.linewidth,
                    fill=False,
                    label=structure.name
                )
                ax.add_patch(circle)
            else:
                points = structure.calculate_points()
                polygon = patches.Polygon(
                    points,
                    closed=True,
                    fill=False,
                    edgecolor=structure.edgecolor,
                    linewidth=structure.linewidth,
                    label=structure.name
                )
                ax.add_patch(polygon)
        elif isinstance(structure, Structure):
            polygon = patches.Polygon(
                structure.points,
                closed=True,
                fill=False,
                edgecolor=structure.edgecolor,
                linewidth=structure.linewidth,
                label=structure.name
            )
            ax.add_patch(polygon)

    # Create a custom legend
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))  # Remove duplicates
    # ax.legend(unique_labels.values(), unique_labels.keys(), loc="upper right", fontsize=8)

    # Save and display
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.show()

def load_layout_from_yaml(yaml_file: str) -> Layout:
    with open(yaml_file, "r") as file:
        data = yaml.safe_load(file)

    # Create plot
    plot_data = data["plot"]
    plot = Plot(
        width=plot_data["width"],
        length=plot_data["length"],
        extra_space=plot_data["extra_space"]
    )

    # Create structures
    structures = []
    for structure_data in data.get("structures", []):
        struct_type = structure_data.get("type", "polygon").lower()
        if struct_type == "polygon":
            structure = Structure(
                name=structure_data["name"],
                points=[tuple(point) for point in structure_data["points"]],
                edgecolor=structure_data.get("edgecolor", "red"),
                linewidth=structure_data.get("linewidth", 1.5)
            )
            structures.append(structure)
        elif struct_type == "shaped":
            shape = structure_data["shape"].lower()
            position = (structure_data["position"]["x"], structure_data["position"]["y"])
            dimensions = structure_data["dimensions"]
            if shape in ["rectangle", "square"]:
                if shape == "rectangle":
                    dims = (dimensions["length"], dimensions["breadth"])
                elif shape == "square":
                    dims = dimensions["side"] if "side" in dimensions else dimensions.get("length", 10)  # Default side length
                alignment = structure_data.get("alignment", "center")
                shaped_structure = ShapedStructure(
                    name=structure_data["name"],
                    shape=shape,
                    position=position,
                    dimensions=dims,
                    alignment=alignment,
                    edgecolor=structure_data.get("edgecolor", "blue"),
                    linewidth=structure_data.get("linewidth", 1.5)
                )
                structures.append(shaped_structure)
            elif shape == "circle":
                radius = dimensions["radius"]
                shaped_structure = ShapedStructure(
                    name=structure_data["name"],
                    shape=shape,
                    position=position,
                    dimensions=radius,
                    edgecolor=structure_data.get("edgecolor", "blue"),
                    linewidth=structure_data.get("linewidth", 1.5)
                )
                structures.append(shaped_structure)
            else:
                raise ValueError(f"Unsupported shaped structure: {shape}")
        else:
            raise ValueError(f"Unsupported structure type: {struct_type}")

    # Return layout
    return Layout(plot=plot, structures=structures)


# Load layout from YAML and draw
layout = load_layout_from_yaml("layout_config.yaml")
draw_layout(layout, "layout_output.pdf")


