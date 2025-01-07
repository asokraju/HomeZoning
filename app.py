import matplotlib.pyplot as plt
import matplotlib.patches as patches
import yaml
import math

from dataclasses import dataclass, field
from typing import List, Tuple, Union, Optional

###############################################################################
#                            PLOT AND STRUCTURES
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
    Polygon-based structure (e.g. House, Patio) with optional irrigation fields.
    """
    name: str
    points: List[Tuple[float, float]]
    edgecolor: str = "red"
    linewidth: float = 1.5
    zorder: int = 1

    main_line_alignment: str = "optimal"  # top/bottom/left/right/none/optimal
    number_of_drip_lines: int = 2
    main_line_inlet: Optional[Tuple[float, float]] = None
    main_line_end: Optional[Tuple[float, float]] = None
    needs_irrigation: bool = False

    def bounding_box(self) -> Tuple[float, float, float, float]:
        xs, ys = zip(*self.points)
        return (min(xs), max(xs), min(ys), max(ys))


@dataclass
class ShapedStructure:
    """
    Represents a shape (rectangle, square, circle), each possibly having unique irrigation.
    """
    name: str
    shape: str   # 'rectangle', 'square', 'circle'
    position: Tuple[float, float]
    dimensions: Union[Tuple[float, float], float]  # (length, breadth) or radius
    alignment: str = "center"
    edgecolor: str = "blue"
    linewidth: float = 1.5
    zorder: int = 1

    # Irrigation fields
    main_line_alignment: str = "optimal"
    number_of_drip_lines: int = 2
    main_line_inlet: Optional[Tuple[float, float]] = None
    main_line_end: Optional[Tuple[float, float]] = None
    needs_irrigation: bool = False

    def calculate_points(self) -> List[Tuple[float, float]]:
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
            # For circle, we do NOT create corner points. 
            # We'll handle circle drawing directly in the plotting function.
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
        if self.shape.lower() in ["rectangle", "square"]:
            pts = self.calculate_points()
            xs, ys = zip(*pts)
            return (min(xs), max(xs), min(ys), max(ys))
        elif self.shape.lower() == "circle":
            r = self.dimensions
            cx, cy = self.position
            return (cx - r, cx + r, cy - r, cy + r)
        else:
            raise ValueError(f"Unsupported shape in bounding_box: {self.shape}")


###############################################################################
#                          IRRIGATION CLASSES
###############################################################################

@dataclass
class IrrigationLine:
    name: str
    bed_name: str  # store which bed/structure it belongs to
    coordinates: List[Tuple[float, float]]
    color: str = "magenta"
    linewidth: float = 1.5
    zorder: int = 5

    @property
    def length(self) -> float:
        dist = 0.0
        for i in range(len(self.coordinates) - 1):
            x1, y1 = self.coordinates[i]
            x2, y2 = self.coordinates[i + 1]
            dist += math.dist((x1, y1), (x2, y2))  # requires Python 3.8+
        return dist


@dataclass
class IrrigationFitting:
    name: str
    bed_name: str
    fitting_type: str  # "elbow", "tee", "end_cap_main", "goof_plug", "valve", "quarter_inch_coupling"
    position: Tuple[float, float]
    color: str = "lime"
    marker: str = "o"
    size: float = 25
    zorder: int = 6


###############################################################################
#                                 LAYOUT
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
        For each structure that needs irrigation:
          1) Create main line (or skip if alignment is 'none').
          2) Create drip lines based on number_of_drip_lines.
        Refactored into smaller helper methods.
        """
        for s in self.structures:
            if not s.needs_irrigation:
                continue

            x_min, x_max, y_min, y_max = s.bounding_box()
            # Create main line
            main_line, main_fittings = self.create_main_line(s, x_min, x_max, y_min, y_max)
            if main_line:
                self.irrigation_lines.append(main_line)
                self.irrigation_fittings.extend(main_fittings)

            # Create drip lines
            drip_lines, drip_fittings = self.create_drip_lines(s, x_min, x_max, y_min, y_max)
            self.irrigation_lines.extend(drip_lines)
            self.irrigation_fittings.extend(drip_fittings)

    def create_main_line(self, s: Union[Structure, ShapedStructure],
                         x_min: float, x_max: float, y_min: float, y_max: float) \
            -> (Optional[IrrigationLine], List[IrrigationFitting]):

        bed_name = s.name
        w = x_max - x_min
        h = y_max - y_min

        alignment = s.main_line_alignment.lower()
        main_line_coords = []
        fittings = []

        if alignment == "optimal":
            # pick the shorter dimension => vertical if width < height
            if w < h:
                alignment = "left"
            else:
                alignment = "bottom"

        if alignment == "none":
            return None, []

        # If the user provided BOTH inlet and end, we directly use those:
        if s.main_line_inlet and s.main_line_end:
            main_line_coords = [s.main_line_inlet, s.main_line_end]
        elif s.main_line_inlet or s.main_line_end:
            raise ValueError(f"{bed_name}: Must provide BOTH main_line_inlet and main_line_end or neither.")
        else:
            # auto-generate based on alignment
            inset = 0.5
            if alignment == "top":
                main_line_coords = [(x_min + inset, y_max - inset), (x_max - inset, y_max - inset)]
            elif alignment == "bottom":
                main_line_coords = [(x_min + inset, y_min + inset), (x_max - inset, y_min + inset)]
            elif alignment == "left":
                main_line_coords = [(x_min + inset, y_min + inset), (x_min + inset, y_max - inset)]
            elif alignment == "right":
                main_line_coords = [(x_max - inset, y_min + inset), (x_max - inset, y_max - inset)]
            else:
                # unknown alignment
                return None, []

        # Create main line if we have 2 coords
        if len(main_line_coords) == 2:
            main_line = IrrigationLine(
                name=f"{bed_name}_MainLine",
                bed_name=bed_name,
                coordinates=main_line_coords,
                color="darkorange",
                linewidth=2.5,
                zorder=5
            )
            # fittings => end cap on far end, valve on start
            start_valve = IrrigationFitting(
                name=f"{bed_name}_MainLine_Valve",
                bed_name=bed_name,
                fitting_type="valve",
                position=main_line_coords[0],
                color="red",
                marker="^",
                size=40,
                zorder=6
            )
            end_cap = IrrigationFitting(
                name=f"{bed_name}_MainLine_EndCap",
                bed_name=bed_name,
                fitting_type="end_cap_main",
                position=main_line_coords[-1],
                color="blue",
                marker="s",
                size=40,
                zorder=6
            )
            fittings.extend([start_valve, end_cap])
            return main_line, fittings

        return None, []

    def create_drip_lines(self, s: Union[Structure, ShapedStructure],
                          x_min: float, x_max: float, y_min: float, y_max: float) \
            -> (List[IrrigationLine], List[IrrigationFitting]):

        bed_name = s.name
        lines = []
        fits = []

        w = x_max - x_min
        h = y_max - y_min

        # Determine alignment again (similar logic as main line)
        alignment = s.main_line_alignment.lower()
        if alignment == "optimal":
            if w < h:
                alignment = "left"
            else:
                alignment = "bottom"

        # If main line is horizontal => drip lines are vertical
        # If main line is vertical => drip lines are horizontal
        # If "none", skip drip lines
        if alignment == "none":
            return [], []

        drip_count = s.number_of_drip_lines
        if drip_count < 1:
            return [], []

        if alignment in ["top", "bottom"]:
            # main line horizontal => drip lines vertical
            # space them along x
            step = (x_max - x_min) / (drip_count + 1)
            for i in range(drip_count):
                x_coord = x_min + step * (i + 1)
                coords = [(x_coord, y_min + 1.0), (x_coord, y_max - 1.0)]
                dl = IrrigationLine(
                    name=f"{bed_name}_DripLine_{i+1}",
                    bed_name=bed_name,
                    coordinates=coords,
                    color="green",
                    linewidth=1.5,
                    zorder=5
                )
                lines.append(dl)

                # Coupling at the start
                coupling = IrrigationFitting(
                    name=f"{bed_name}_DripLine_{i+1}_Coupling",
                    bed_name=bed_name,
                    fitting_type="quarter_inch_coupling",
                    position=coords[0],
                    color="purple",
                    marker="d",
                    size=25,
                    zorder=6
                )
                # Goof plug at the end
                goof = IrrigationFitting(
                    name=f"{bed_name}_DripLine_{i+1}_GoofPlug",
                    bed_name=bed_name,
                    fitting_type="goof_plug",
                    position=coords[-1],
                    color="black",
                    marker="x",
                    size=25,
                    zorder=6
                )
                fits.extend([coupling, goof])

        elif alignment in ["left", "right"]:
            # main line vertical => drip lines horizontal
            step = (y_max - y_min) / (drip_count + 1)
            for i in range(drip_count):
                y_coord = y_min + step * (i + 1)
                coords = [(x_min + 1.0, y_coord), (x_max - 1.0, y_coord)]
                dl = IrrigationLine(
                    name=f"{bed_name}_DripLine_{i+1}",
                    bed_name=bed_name,
                    coordinates=coords,
                    color="green",
                    linewidth=1.5,
                    zorder=5
                )
                lines.append(dl)

                coupling = IrrigationFitting(
                    name=f"{bed_name}_DripLine_{i+1}_Coupling",
                    bed_name=bed_name,
                    fitting_type="quarter_inch_coupling",
                    position=coords[0],
                    color="purple",
                    marker="d",
                    size=25,
                    zorder=6
                )
                goof = IrrigationFitting(
                    name=f"{bed_name}_DripLine_{i+1}_GoofPlug",
                    bed_name=bed_name,
                    fitting_type="goof_plug",
                    position=coords[-1],
                    color="black",
                    marker="x",
                    size=25,
                    zorder=6
                )
                fits.extend([coupling, goof])

        return lines, fits

    def compute_irrigation_totals_by_bed(self):
        """
        Summarize usage per bed:
          {
             'Bed A': {
                  'main_line_length': X,
                  'drip_line_length': Y,
                  'fittings_count': {'valve': 1, 'end_cap_main': 1, ...}
             },
             'Bed B': { ... }
          }
        """
        summary = {}
        for line in self.irrigation_lines:
            b = line.bed_name
            summary.setdefault(b, {
                "main_line_length": 0.0,
                "drip_line_length": 0.0,
                "fittings_count": {}
            })
            if "_MainLine" in line.name:
                summary[b]["main_line_length"] += line.length
            elif "_DripLine_" in line.name:
                summary[b]["drip_line_length"] += line.length

        for f in self.irrigation_fittings:
            b = f.bed_name
            summary.setdefault(b, {
                "main_line_length": 0.0,
                "drip_line_length": 0.0,
                "fittings_count": {}
            })
            summary[b]["fittings_count"].setdefault(f.fitting_type, 0)
            summary[b]["fittings_count"][f.fitting_type] += 1

        return summary


###############################################################################
#                               DRAWING
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

    # Limits and labels
    ax.set_xlim(-5, layout.plot.new_length - 5)
    ax.set_ylim(-5, layout.plot.new_width - 5)
    ax.set_xlabel("Length (feet)", fontsize=10)
    ax.set_ylabel("Width (feet)", fontsize=10)
    ax.tick_params(axis="x", which="major", labelsize=8, rotation=90)
    ax.tick_params(axis="y", which="major", labelsize=8)
    ax.tick_params(axis="both", which="minor", labelsize=6)

    # Plot boundary
    ax.add_patch(
        patches.Rectangle(
            (0, 0),
            layout.plot.length,
            layout.plot.width,
            fill=False,
            edgecolor="blue",
            linewidth=1.5,
            label="Plot Boundary",
            zorder=0
        )
    )

    # Sort structures by zorder
    sorted_structures = sorted(layout.structures, key=lambda s: s.zorder)

    # Draw structures
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
                # rectangle or square
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
            # A standard polygon structure
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

    # Legend without duplicates

    # handles, labels = ax.get_legend_handles_labels()
    # unique_dict = dict(zip(labels, handles))
    # ax.legend(unique_dict.values(), unique_dict.keys(), loc="upper right", fontsize=8)

    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.show()


###############################################################################
#                            YAML LOADER
###############################################################################

def load_layout_from_yaml(yaml_file: str) -> Layout:
    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)

    # Create plot
    plot_data = data["plot"]
    plot = Plot(
        width=plot_data["width"],
        length=plot_data["length"],
        extra_space=plot_data["extra_space"]
    )

    structures = []
    # Polygons
    polygons = data.get("structures", {}).get("polygons", [])
    for poly in polygons:
        struct = Structure(
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
        structures.append(struct)

    # Shaped structures
    shaped_list = data.get("structures", {}).get("shaped_structures", [])
    for shaped in shaped_list:
        shape = shaped["shape"].lower()

        # Default irrigation fields from top-level of this item
        base_main_line_align = shaped.get("main_line_alignment", "optimal")
        base_num_drips = shaped.get("number_of_drip_lines", 2)
        base_needs_irr = shaped.get("needs_irrigation", False)
        base_inlet = shaped.get("main_line_inlet", None)
        base_end = shaped.get("main_line_end", None)

        if shape in ["rectangle", "square"]:
            if shape == "rectangle":
                dims = (shaped["dimensions"]["length"], shaped["dimensions"]["breadth"])
            else:
                dims = shaped["dimensions"]["side"]
        elif shape == "circle":
            dims = shaped["dimensions"]["radius"]
        else:
            raise ValueError(f"Unsupported shape {shape}")

        # We'll treat 'positions' as individual beds, possibly overriding fields
        base_name = shaped.get("name", "UnnamedGroup")
        for i, pos in enumerate(shaped["positions"]):
            # Pull possible overrides from the position
            bed_name = pos.get("name", f"{base_name}_{i+1}")
            ml_align = pos.get("main_line_alignment", base_main_line_align)
            num_drips = pos.get("number_of_drip_lines", base_num_drips)
            needs_irr = pos.get("needs_irrigation", base_needs_irr)
            inlet = pos.get("main_line_inlet", base_inlet)
            end = pos.get("main_line_end", base_end)

            s_obj = ShapedStructure(
                name=bed_name,
                shape=shape,
                position=(pos["x"], pos["y"]),
                dimensions=dims,
                alignment=shaped.get("alignment", "center"),
                edgecolor=shaped.get("edgecolor", "blue"),
                linewidth=shaped.get("linewidth", 1.5),
                zorder=shaped.get("zorder", 1),
                main_line_alignment=ml_align,
                number_of_drip_lines=num_drips,
                main_line_inlet=inlet,
                main_line_end=end,
                needs_irrigation=needs_irr
            )
            structures.append(s_obj)

    layout = Layout(plot=plot, structures=structures)

    # Optionally parse irrigation from top-level "irrigation" key
    irrigation_data = data.get("irrigation", {})
    lines_data = irrigation_data.get("lines", [])
    for ld in lines_data:
        coords = [tuple(pt) for pt in ld["coordinates"]]
        line_obj = IrrigationLine(
            name=ld["name"],
            bed_name="(Global)",  # or some global name
            coordinates=coords,
            color=ld.get("color", "magenta"),
            linewidth=ld.get("linewidth", 1.5),
            zorder=ld.get("zorder", 5)
        )
        layout.irrigation_lines.append(line_obj)

    fittings_data = irrigation_data.get("fittings", [])
    for fd in fittings_data:
        fit_obj = IrrigationFitting(
            name=fd["name"],
            bed_name="(Global)",  # or a bed if you want
            fitting_type=fd["fitting_type"],
            position=tuple(fd["position"]),
            color=fd.get("color", "lime"),
            marker=fd.get("marker", "o"),
            size=fd.get("size", 25),
            zorder=fd.get("zorder", 6)
        )
        layout.irrigation_fittings.append(fit_obj)

    return layout


###############################################################################
#                            MAIN EXECUTION
###############################################################################

if __name__ == "__main__":
    layout = load_layout_from_yaml("layout_config.yaml")

    # Generate lines/fittings for each bed that needs irrigation
    layout.finalize_irrigation_for_beds()

    # Draw
    draw_layout(layout, "layout_output.pdf")

    # Summaries
    bed_summaries = layout.compute_irrigation_totals_by_bed()
    print("Irrigation Summary By Bed:")
    for bed, info in bed_summaries.items():
        print(f"  {bed}:")
        print(f"    Main Line Length = {info['main_line_length']:.2f} ft")
        print(f"    Drip Line Length = {info['drip_line_length']:.2f} ft")
        print(f"    Fittings Count   = {info['fittings_count']}")
