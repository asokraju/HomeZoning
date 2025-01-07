import matplotlib.pyplot as plt
import matplotlib.patches as patches
import yaml
from dataclasses import dataclass, field
from typing import List, Tuple, Union, Optional
import math

###############################################################################
#                            Plot and Structures
###############################################################################

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
    """
    Polygon-based structure (e.g. House, Patio, etc.) with optional irrigation.
    """
    name: str
    points: List[Tuple[float, float]]
    edgecolor: str = "red"
    linewidth: float = 1.5
    zorder: int = 1

    # New irrigation-related fields
    main_line_alignment: str = "optimal"           # (Top, Bottom, Left, Right, None, "optimal")
    number_of_drip_lines: int = 2
    main_line_inlet: Optional[Tuple[float, float]] = None
    main_line_end: Optional[Tuple[float, float]] = None
    needs_irrigation: bool = False

    def bounding_box(self) -> Tuple[float, float, float, float]:
        """
        Return (x_min, x_max, y_min, y_max).
        """
        xs, ys = zip(*self.points)
        return (min(xs), max(xs), min(ys), max(ys))


@dataclass
class ShapedStructure:
    """
    Represents a shape (rectangle, square, circle) with optional irrigation.
    """
    name: str
    shape: str  # 'rectangle', 'square', 'circle'
    position: Tuple[float, float]
    dimensions: Union[Tuple[float, float], float]  # (length, breadth) or radius
    alignment: str = "center"
    edgecolor: str = "blue"
    linewidth: float = 1.5
    zorder: int = 1

    # New irrigation fields
    main_line_alignment: str = "optimal"           # "optimal" or a fixed side
    number_of_drip_lines: int = 2
    main_line_inlet: Optional[Tuple[float, float]] = None
    main_line_end: Optional[Tuple[float, float]] = None
    needs_irrigation: bool = False

    def calculate_points(self) -> List[Tuple[float, float]]:
        """
        Return corner points for rectangle/square; circles are handled differently.
        """
        if self.shape.lower() == "rectangle":
            if not isinstance(self.dimensions, tuple) or len(self.dimensions) != 2:
                raise ValueError("Rectangle dimensions must be (length, breadth).")
            x, y = self.position
            length, breadth = self.dimensions
            return self._rectangle_points(x, y, length, breadth, self.alignment)

        elif self.shape.lower() == "square":
            side = self.dimensions
            if not isinstance(side, (int, float)):
                raise ValueError("Square dimension must be a single numeric side.")
            x, y = self.position
            return self._rectangle_points(x, y, side, side, self.alignment)

        else:
            # Circle or other
            raise ValueError(f"Unsupported shape for calculate_points: {self.shape}")

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
                (x,         y),
                (x,         y - breadth),
                (x + length, y - breadth),
                (x + length, y)
            ]
        elif alignment == "top-right":
            return [
                (x,         y),
                (x,         y - breadth),
                (x - length, y - breadth),
                (x - length, y)
            ]
        elif alignment == "bottom-left":
            return [
                (x,          y),
                (x,          y + breadth),
                (x + length, y + breadth),
                (x + length, y)
            ]
        elif alignment == "bottom-right":
            return [
                (x,          y),
                (x,          y + breadth),
                (x - length, y + breadth),
                (x - length, y)
            ]
        else:
            raise ValueError(f"Invalid alignment: {alignment}")

    def bounding_box(self) -> Tuple[float, float, float, float]:
        """
        For rectangle/square: bounding box from corner points.
        For circle, bounding box is center ± radius.
        """
        if self.shape.lower() in ["rectangle", "square"]:
            pts = self.calculate_points()
            xs, ys = zip(*pts)
            return (min(xs), max(xs), min(ys), max(ys))
        elif self.shape.lower() == "circle":
            # bounding box for circle
            r = self.dimensions
            cx, cy = self.position
            return (cx - r, cx + r, cy - r, cy + r)
        else:
            raise ValueError(f"Unsupported shape in bounding_box: {self.shape}")


###############################################################################
#                           Irrigation Classes
###############################################################################

@dataclass
class IrrigationLine:
    name: str
    coordinates: List[Tuple[float, float]]
    color: str = "magenta"
    linewidth: float = 1.5
    zorder: int = 5

    # Length is computed from coordinates
    @property
    def length(self) -> float:
        """
        Summation of distances between consecutive points in 'coordinates'.
        """
        dist = 0.0
        for i in range(len(self.coordinates) - 1):
            x1, y1 = self.coordinates[i]
            x2, y2 = self.coordinates[i+1]
            dist += math.dist((x1, y1), (x2, y2))  # Python 3.8+
        return dist


@dataclass
class IrrigationFitting:
    name: str
    fitting_type: str  # 'tee', 'elbow', 'end_cap_main', 'goof_plug', 'valve', 'quarter_inch_coupling'
    position: Tuple[float, float]
    color: str = "lime"
    marker: str = "o"
    size: float = 30  # smaller default
    zorder: int = 6


###############################################################################
#                                Layout
###############################################################################

@dataclass
class Layout:
    plot: Plot
    structures: List[Union[Structure, ShapedStructure]] = field(default_factory=list)
    irrigation_lines: List[IrrigationLine] = field(default_factory=list)
    irrigation_fittings: List[IrrigationFitting] = field(default_factory=list)

    def add_structure(self, s: Union[Structure, ShapedStructure]):
        self.structures.append(s)

    def add_irrigation_line(self, line: IrrigationLine):
        self.irrigation_lines.append(line)

    def add_irrigation_fitting(self, fitting: IrrigationFitting):
        self.irrigation_fittings.append(fitting)

    def finalize_irrigation_for_beds(self):
        """
        For each bed that needs irrigation:
          1. Determine bounding box
          2. Align main line on the specified side (or compute 'optimal')
          3. Place number_of_drip_lines with 1/4 inch couplings
          4. Place fittings: elbow, tee, end caps, etc.
          5. Save lengths & fittings for total supply computations
        """
        for s in self.structures:
            if not s.needs_irrigation:
                continue

            # Grab bounding box
            x_min, x_max, y_min, y_max = s.bounding_box()
            w = x_max - x_min
            h = y_max - y_min

            # Validate main_line_inlet / main_line_end if provided
            if s.main_line_inlet:
                if s.main_line_inlet not in s.points and hasattr(s, "points"):
                    raise ValueError(f"Inlet {s.main_line_inlet} not a valid point of structure {s.name}")

            # Determine actual alignment
            alignment = s.main_line_alignment.lower()  # e.g. "top", "bottom", "optimal"...
            if alignment == "optimal":
                # pick the shorter side
                # if width < height => main line along left or right
                alignment = "right"  # default guess
                if w < h:
                    alignment = "bottom"

            # We will create a main line as a two-point line (start -> end)
            main_line_coords = []
            main_line_name = f"{s.name}_MainLine"

            # Decide main line based on alignment
            # We'll keep it inset by 0.5 ft from the boundary, for clarity
            if alignment == "top":
                main_line_coords = [(x_min + 0.5, y_max - 0.5), (x_max - 0.5, y_max - 0.5)]
            elif alignment == "bottom":
                main_line_coords = [(x_min + 0.5, y_min + 0.5), (x_max - 0.5, y_min + 0.5)]
            elif alignment == "left":
                main_line_coords = [(x_min + 0.5, y_min + 0.5), (x_min + 0.5, y_max - 0.5)]
            elif alignment == "right":
                main_line_coords = [(x_max - 0.5, y_min + 0.5), (x_max - 0.5, y_max - 0.5)]
            elif alignment == "none":
                # User doesn't want a main line
                main_line_coords = []
            else:
                raise ValueError(f"Unrecognized alignment: {alignment}")

            # If user gave main_line_inlet or main_line_end, override or check
            if s.main_line_inlet and s.main_line_end:
                main_line_coords = [s.main_line_inlet, s.main_line_end]
            elif s.main_line_inlet or s.main_line_end:
                # partial specification => error
                raise ValueError(f"{s.name}: Must specify both inlet & end or neither.")

            # Create the main line object if coords are not empty
            if main_line_coords:
                main_line = IrrigationLine(
                    name=main_line_name,
                    coordinates=main_line_coords,
                    color="darkorange",   # thicker orange line
                    linewidth=2.5,        # thicker
                    zorder=5
                )
                self.add_irrigation_line(main_line)

                # Place end cap on main line end
                end_fitting = IrrigationFitting(
                    name=f"{s.name}_MainLine_EndCap",
                    fitting_type="end_cap_main",
                    position=main_line_coords[-1],
                    color="blue",
                    marker="s",   # square marker
                    size=50,
                    zorder=6
                )
                self.add_irrigation_fitting(end_fitting)

                # Possibly place an inlet fitting at main_line_coords[0]
                inlet_fitting = IrrigationFitting(
                    name=f"{s.name}_MainLine_Inlet",
                    fitting_type="valve",  # we can treat the inlet as a valve or TEE
                    position=main_line_coords[0],
                    color="red",
                    marker="^",  # triangle up
                    size=50,
                    zorder=6
                )
                self.add_irrigation_fitting(inlet_fitting)

                # Optionally place elbows if line is vertical or horizontal
                # If it is exactly vertical or horizontal, we might consider an elbow shape
                # but we can skip if it’s just a straightforward line.

            # 2. Drip lines: "number_of_drip_lines" inside the bounding box, parallel to short side
            # If alignment is top/bottom => drip lines are vertical (since main line is horizontal).
            # If alignment is left/right => drip lines are horizontal.
            drip_lines = []
            spacing = 0.5  # how far from the main line each drip line starts, purely for example
            if alignment in ["top", "bottom"]:
                # main line is horizontal => drip lines vertical
                # let's space them out along x direction
                step = (x_max - x_min) / (s.number_of_drip_lines + 1)
                for i in range(s.number_of_drip_lines):
                    x_coord = x_min + step * (i + 1)
                    drip_lines.append([
                        (x_coord, y_min + 1.0),
                        (x_coord, y_max - 1.0)
                    ])
            elif alignment in ["left", "right"]:
                # main line is vertical => drip lines horizontal
                step = (y_max - y_min) / (s.number_of_drip_lines + 1)
                for i in range(s.number_of_drip_lines):
                    y_coord = y_min + step * (i + 1)
                    drip_lines.append([
                        (x_min + 1.0, y_coord),
                        (x_max - 1.0, y_coord)
                    ])

            # Create drip lines
            for idx, coords in enumerate(drip_lines, start=1):
                dl = IrrigationLine(
                    name=f"{s.name}_DripLine_{idx}",
                    coordinates=coords,
                    color="green",
                    linewidth=1.5,
                    zorder=5
                )
                self.add_irrigation_line(dl)

                # At the start of each drip line, place a 1/4 inch coupling
                coupling = IrrigationFitting(
                    name=f"{s.name}_DripLine_{idx}_Coupling",
                    fitting_type="quarter_inch_coupling",
                    position=coords[0],
                    color="purple",
                    marker="d",  # diamond
                    size=40,
                    zorder=6
                )
                self.add_irrigation_fitting(coupling)

                # At the end, place a goof plug (end cap for drip line)
                goof = IrrigationFitting(
                    name=f"{s.name}_DripLine_{idx}_GoofPlug",
                    fitting_type="goof_plug",
                    position=coords[-1],
                    color="black",
                    marker="x",  # small 'x'
                    size=40,
                    zorder=6
                )
                self.add_irrigation_fitting(goof)

    def compute_irrigation_totals(self):
        """
        Return a dict summarizing total length of main lines, drip lines, 
        and count of each fitting type.
        """
        summary = {
            "main_line_length": 0.0,
            "drip_line_length": 0.0,
            "fittings_count": {}
        }
        # Identify main lines by name (ends with "_MainLine")
        # Drip lines by name (contains "_DripLine_")
        for line in self.irrigation_lines:
            if line.name.endswith("_MainLine"):
                summary["main_line_length"] += line.length
            elif "_DripLine_" in line.name:
                summary["drip_line_length"] += line.length

        # Count fittings by type
        for f in self.irrigation_fittings:
            summary["fittings_count"].setdefault(f.fitting_type, 0)
            summary["fittings_count"][f.fitting_type] += 1

        return summary


###############################################################################
#                                DRAWING
###############################################################################

def draw_layout(layout: Layout, filename: str):
    aspect_ratio = layout.plot.new_width / layout.plot.new_length
    fig, ax = plt.subplots(figsize=(30 / aspect_ratio, 30))

    # Gridlines
    major_xticks = range(-5, layout.plot.new_length + 1, 5)
    major_yticks = range(-5, layout.plot.new_width + 1, 5)
    minor_xticks = range(-5, layout.plot.new_length + 1)
    minor_yticks = range(-5, layout.plot.new_width + 1)

    ax.set_xticks(major_xticks)
    ax.set_yticks(major_yticks)
    ax.set_xticks(minor_xticks, minor=True)
    ax.set_yticks(minor_yticks, minor=True)
    ax.grid(which="minor", linestyle=":", linewidth=0.5, color="gray")
    ax.grid(which="major", linestyle="-", linewidth=0.8, color="black")

    # Limits
    ax.set_xlim(-5, layout.plot.new_length - 5)
    ax.set_ylim(-5, layout.plot.new_width - 5)
    ax.set_xlabel("Length (feet)", fontsize=10)
    ax.set_ylabel("Width (feet)", fontsize=10)
    ax.tick_params(axis="x", which="major", labelsize=8, rotation=90)
    ax.tick_params(axis="y", which="major", labelsize=8)
    ax.tick_params(axis="both", which="minor", labelsize=6)

    # Plot boundary
    ax.add_patch(patches.Rectangle(
        (0, 0),
        layout.plot.length,
        layout.plot.width,
        fill=False, edgecolor="blue", linewidth=1.5,
        label="Plot Boundary", zorder=0
    ))

    # Sort structures by zorder
    sorted_structures = sorted(layout.structures, key=lambda s: s.zorder)

    # Add structures
    for structure in sorted_structures:
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
                    label=structure.name,
                    zorder=structure.zorder
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
                    label=structure.name,
                    zorder=structure.zorder
                )
                ax.add_patch(polygon)
        else:
            # Normal polygon structure
            polygon = patches.Polygon(
                structure.points,
                closed=True,
                fill=False,
                edgecolor=structure.edgecolor,
                linewidth=structure.linewidth,
                label=structure.name,
                zorder=structure.zorder
            )
            ax.add_patch(polygon)

    # Draw irrigation lines
    for line in layout.irrigation_lines:
        xs, ys = zip(*line.coordinates)
        ax.plot(
            xs, ys,
            color=line.color,
            linewidth=line.linewidth,
            label=line.name,
            zorder=line.zorder
        )

    # Draw irrigation fittings
    for fitting in layout.irrigation_fittings:
        ax.scatter(
            fitting.position[0],
            fitting.position[1],
            c=fitting.color,
            marker=fitting.marker,
            s=fitting.size,
            label=fitting.name,
            zorder=fitting.zorder
        )

    # Legend (remove duplicates)
    handles, labels = ax.get_legend_handles_labels()
    unique_dict = dict(zip(labels, handles))
    # ax.legend(unique_dict.values(), unique_dict.keys(), loc="upper right", fontsize=8)

    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.show()


###############################################################################
#                               YAML LOADER
###############################################################################

def load_layout_from_yaml(yaml_file: str) -> Layout:
    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)

    # Create Plot
    plot_data = data["plot"]
    plot = Plot(
        width=plot_data["width"],
        length=plot_data["length"],
        extra_space=plot_data["extra_space"]
    )

    # Parse structures
    structures = []
    polygons = data.get("structures", {}).get("polygons", [])
    for poly in polygons:
        structure = Structure(
            name=poly["name"],
            points=[tuple(pt) for pt in poly["points"]],
            edgecolor=poly.get("edgecolor", "red"),
            linewidth=poly.get("linewidth", 1.5),
            zorder=poly.get("zorder", 1),
            main_line_alignment=poly.get("main_line_alignment", "optimal"),
            number_of_drip_lines=poly.get("number_of_drip_lines", 2),
            main_line_inlet=poly.get("main_line_inlet", None),
            main_line_end=poly.get("main_line_end", None),
            needs_irrigation=poly.get("needs_irrigation", False)
        )
        structures.append(structure)

    shaped_polys = data.get("structures", {}).get("shaped_structures", [])
    for shaped in shaped_polys:
        shape = shaped["shape"].lower()
        if shape in ["rectangle", "square"]:
            if shape == "rectangle":
                dims = (shaped["dimensions"]["length"], shaped["dimensions"]["breadth"])
            else:
                dims = shaped["dimensions"]["side"]
        elif shape == "circle":
            dims = shaped["dimensions"]["radius"]
        else:
            raise ValueError(f"Unsupported shape {shape}")

        for pos in shaped["positions"]:
            s_obj = ShapedStructure(
                name=shaped["name"],
                shape=shape,
                position=(pos["x"], pos["y"]),
                dimensions=dims,
                alignment=shaped.get("alignment", "center"),
                edgecolor=shaped.get("edgecolor", "blue"),
                linewidth=shaped.get("linewidth", 1.5),
                zorder=shaped.get("zorder", 1),
                main_line_alignment=shaped.get("main_line_alignment", "optimal"),
                number_of_drip_lines=shaped.get("number_of_drip_lines", 2),
                main_line_inlet=shaped.get("main_line_inlet", None),
                main_line_end=shaped.get("main_line_end", None),
                needs_irrigation=shaped.get("needs_irrigation", False)
            )
            structures.append(s_obj)

    layout = Layout(plot=plot, structures=structures)
    return layout


###############################################################################
#                            EXAMPLE EXECUTION
###############################################################################

if __name__ == "__main__":
    layout = load_layout_from_yaml("layout_config.yaml")
    # Generate irrigation lines/fittings for any bed with "needs_irrigation=True"
    layout.finalize_irrigation_for_beds()

    # Draw
    draw_layout(layout, "layout_output.pdf")

    # Summaries
    irrigation_summary = layout.compute_irrigation_totals()
    print("Irrigation Summary:", irrigation_summary)
