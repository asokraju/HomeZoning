#!/usr/bin/env python3

"""
An end-to-end example combining:
1) Modularized data classes (structures, lines, layout)
2) Geometry via shapely
3) YAML + pydantic for validation
4) Single draw function with sub-object .draw(...) calls
5) Minimal testing with pytest (see bottom)
"""

import yaml
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Polygon, Point
from pydantic import BaseModel, Field, validator
from typing import List, Tuple, Optional, Union

###############################################################################
#                               DATA MODELS
###############################################################################

class PlotConfig(BaseModel):
    """Represents the main Plot from the YAML file."""
    width: float
    length: float
    extra_space: float = 0
    figure_size: float = 11.0  # For the vertical dimension in inches

    @property
    def new_width(self) -> float:
        return self.width + 2 * self.extra_space

    @property
    def new_length(self) -> float:
        return self.length + 2 * self.extra_space


class PolygonConfig(BaseModel):
    """Polygon structure from the YAML config."""
    name: str
    points: List[Tuple[float, float]]
    edgecolor: str = "red"
    linewidth: float = 1.5
    zorder: int = 1
    main_line_alignment: str = "optimal"
    number_of_drip_lines: int = 2
    main_line_inlet: Optional[Union[Tuple[float, float], str]] = None
    main_line_end: Optional[Tuple[float, float]] = None
    needs_irrigation: bool = False


class RectangleDimensions(BaseModel):
    length: float
    breadth: float


class ShapedDimensions(BaseModel):
    """Used for squares or circles."""
    side: Optional[float] = None
    radius: Optional[float] = None


class ShapedStructureConfig(BaseModel):
    """For rectangle, square, or circle."""
    name: str
    shape: str
    alignment: str = "center"
    edgecolor: str = "blue"
    linewidth: float = 1.5
    zorder: int = 1
    main_line_alignment: str = "optimal"
    number_of_drip_lines: int = 2
    main_line_inlet: Optional[Union[Tuple[float, float], str]] = None
    main_line_end: Optional[Tuple[float, float]] = None
    needs_irrigation: bool = False

    # positions => multiple "beds" or instances
    # each has x, y, plus optional overrides
    positions: list

    # For rectangle or square or circle
    dimensions: Union[RectangleDimensions, ShapedDimensions]

    @validator('shape')
    def check_valid_shape(cls, v):
        if v.lower() not in ['rectangle', 'square', 'circle']:
            raise ValueError(f"Unsupported shape: {v}")
        return v


class IrrigationLineConfig(BaseModel):
    name: str
    bed_name: str = "(Global)"
    coordinates: List[Tuple[float, float]]
    color: str = "magenta"
    linewidth: float = 1.5
    zorder: int = 5


class IrrigationFittingConfig(BaseModel):
    name: str
    bed_name: str = "(Global)"
    fitting_type: str
    position: Tuple[float, float]
    color: str = "lime"
    marker: str = "o"
    size: float = 25
    zorder: int = 6


class StructuresConfig(BaseModel):
    polygons: List[PolygonConfig] = []
    shaped_structures: List[ShapedStructureConfig] = []


class IrrigationConfig(BaseModel):
    lines: List[IrrigationLineConfig] = []
    fittings: List[IrrigationFittingConfig] = []


class LayoutConfig(BaseModel):
    plot: PlotConfig
    structures: StructuresConfig
    irrigation: Optional[IrrigationConfig] = None

###############################################################################
#                             DOMAIN OBJECTS
###############################################################################

class BaseStructure:
    """Base class for a structure in the garden. Must provide .geom() -> shapely geometry."""
    def draw(self, ax: plt.Axes):
        raise NotImplementedError()

    def bounding_box(self) -> Tuple[float, float, float, float]:
        """Return (x_min, x_max, y_min, y_max)."""
        bounds = self.geom().bounds  # shapely (minx, miny, maxx, maxy)
        return (bounds[0], bounds[2], bounds[1], bounds[3])

    def geom(self):
        raise NotImplementedError()


class PolygonStructure(BaseStructure):
    """Represents a polygon-based structure using a shapely Polygon."""
    def __init__(self, config: PolygonConfig):
        self.config = config
        self._polygon = Polygon(config.points)  # shapely Polygon from the points

    def geom(self):
        return self._polygon

    def draw(self, ax: plt.Axes):
        # Plot as a matplotlib patch
        poly_patch = patches.Polygon(
            self.config.points,
            closed=True,
            fill=False,
            edgecolor=self.config.edgecolor,
            linewidth=self.config.linewidth,
            zorder=self.config.zorder
        )
        ax.add_patch(poly_patch)


class ShapedStructure(BaseStructure):
    """Rectangles, Squares, Circles using shapely."""
    def __init__(self, name: str, shape: str, x: float, y: float,
                 dimensions: Union[RectangleDimensions, ShapedDimensions],
                 alignment: str, edgecolor: str, linewidth: float, zorder: int):
        """
        shape in ["rectangle", "square", "circle"]
        alignment in ["center", "top-left", "top-right", "bottom-left", "bottom-right"]
        dimensions:
          - rectangle => length, breadth
          - square => side
          - circle => radius
        """
        self.name = name
        self.shape = shape.lower()
        self.x = x
        self.y = y
        self.dimensions = dimensions
        self.alignment = alignment
        self.edgecolor = edgecolor
        self.linewidth = linewidth
        self.zorder = zorder
        # We'll build a shapely geometry for bounding + drawing

        self._geometry = self._build_geometry()

    def _build_geometry(self):
        if self.shape == "circle":
            # For circle, shapely has buffer on a Point
            radius = self.dimensions.radius
            circle_geom = Point(self.x, self.y).buffer(radius, resolution=64)
            return circle_geom

        elif self.shape in ["rectangle", "square"]:
            if self.shape == "rectangle":
                length = self.dimensions.length
                breadth = self.dimensions.breadth
            else:
                # square
                length = breadth = self.dimensions.side

            # For alignment, we'll compute the corner (x_min, y_min) etc.
            x_min, x_max, y_min, y_max = self._calculate_rect_corners(
                self.x, self.y, length, breadth, self.alignment
            )
            rectangle_coords = [
                (x_min, y_min),
                (x_min, y_max),
                (x_max, y_max),
                (x_max, y_min),
                (x_min, y_min)  # close it
            ]
            return Polygon(rectangle_coords)
        else:
            raise ValueError(f"Unsupported shape: {self.shape}")

    @staticmethod
    def _calculate_rect_corners(cx, cy, length, breadth, alignment):
        """Return (x_min, x_max, y_min, y_max) based on alignment."""
        if alignment == "center":
            x_min = cx - length/2
            x_max = cx + length/2
            y_min = cy - breadth/2
            y_max = cy + breadth/2
        elif alignment == "top-left":
            x_min = cx
            y_max = cy
            x_max = x_min + length
            y_min = y_max - breadth
        elif alignment == "top-right":
            x_max = cx
            y_max = cy
            x_min = x_max - length
            y_min = y_max - breadth
        elif alignment == "bottom-left":
            x_min = cx
            y_min = cy
            x_max = x_min + length
            y_max = y_min + breadth
        elif alignment == "bottom-right":
            x_max = cx
            y_min = cy
            x_min = x_max - length
            y_max = y_min + breadth
        else:
            raise ValueError(f"Invalid alignment: {alignment}")

        return x_min, x_max, y_min, y_max

    def geom(self):
        return self._geometry

    def draw(self, ax: plt.Axes):
        """Draw either a circle or polygon patch."""
        if self.shape == "circle":
            radius = self.dimensions.radius
            circle_patch = patches.Circle(
                (self.x, self.y),
                radius=radius,
                fill=False,
                edgecolor=self.edgecolor,
                linewidth=self.linewidth,
                zorder=self.zorder
            )
            ax.add_patch(circle_patch)
        else:
            # For rect/sq, just draw polygon coords
            xs, ys = self._geometry.exterior.coords.xy
            coords = list(zip(xs, ys))
            poly_patch = patches.Polygon(
                coords,
                closed=True,
                fill=False,
                edgecolor=self.edgecolor,
                linewidth=self.linewidth,
                zorder=self.zorder
            )
            ax.add_patch(poly_patch)


class IrrigationLine:
    """A polyline of irrigation in the layout."""
    def __init__(self, name, bed_name, coords, color, linewidth, zorder):
        self.name = name
        self.bed_name = bed_name
        self.coords = coords
        self.color = color
        self.linewidth = linewidth
        self.zorder = zorder

    def draw(self, ax: plt.Axes):
        xs, ys = zip(*self.coords)
        ax.plot(xs, ys, color=self.color, linewidth=self.linewidth, zorder=self.zorder)

    @property
    def length(self) -> float:
        dist = 0.0
        for i in range(len(self.coords) - 1):
            x1, y1 = self.coords[i]
            x2, y2 = self.coords[i+1]
            dist += math.dist((x1,y1), (x2,y2))
        return dist


class IrrigationFitting:
    """A single fitting at a point."""
    def __init__(self, name, bed_name, fitting_type, position, color, marker, size, zorder):
        self.name = name
        self.bed_name = bed_name
        self.fitting_type = fitting_type
        self.position = position
        self.color = color
        self.marker = marker
        self.size = size
        self.zorder = zorder

    def draw(self, ax: plt.Axes):
        ax.scatter(
            self.position[0], self.position[1],
            c=self.color, marker=self.marker,
            s=self.size, zorder=self.zorder
        )


###############################################################################
#                                LAYOUT
###############################################################################

class Layout:
    """
    The main collection of all Plot, Structures, and any global irrigation lines/fittings.
    You can also call methods to finalize auto-generated irrigation, etc.
    """
    def __init__(self, config: LayoutConfig):
        # Basic plot info
        self.plot_config = config.plot

        # Convert polygon structures -> PolygonStructure
        self.structures = []
        for poly_conf in config.structures.polygons:
            self.structures.append(PolygonStructure(poly_conf))

        # Convert shaped structures
        for shaped_conf in config.structures.shaped_structures:
            # Each shaped_conf can have multiple positions
            base_name = shaped_conf.name
            for i, pos in enumerate(shaped_conf.positions):
                # override name if found
                bed_name = pos.get("name", f"{base_name}_{i+1}")
                x = pos["x"]
                y = pos["y"]

                # figure out correct shape dimensions
                shape = shaped_conf.shape.lower()
                dims = shaped_conf.dimensions
                shaped_obj = ShapedStructure(
                    name=bed_name,
                    shape=shape,
                    x=x,
                    y=y,
                    dimensions=dims,
                    alignment=shaped_conf.alignment,
                    edgecolor=shaped_conf.edgecolor,
                    linewidth=shaped_conf.linewidth,
                    zorder=shaped_conf.zorder
                )
                self.structures.append(shaped_obj)

        # If we had immediate irrigation lines/fittings in config:
        self.irrigation_lines = []
        self.irrigation_fittings = []
        if config.irrigation:
            for line_conf in config.irrigation.lines:
                self.irrigation_lines.append(
                    IrrigationLine(
                        line_conf.name,
                        line_conf.bed_name,
                        line_conf.coordinates,
                        line_conf.color,
                        line_conf.linewidth,
                        line_conf.zorder
                    )
                )
            for fitting_conf in config.irrigation.fittings:
                self.irrigation_fittings.append(
                    IrrigationFitting(
                        fitting_conf.name,
                        fitting_conf.bed_name,
                        fitting_conf.fitting_type,
                        fitting_conf.position,
                        fitting_conf.color,
                        fitting_conf.marker,
                        fitting_conf.size,
                        fitting_conf.zorder
                    )
                )
        # We omit the full "finalize_irrigation_for_beds()" logic for brevity.

    def draw(self, filename="layout_output.pdf"):
        """Draw all structures, lines, and fittings, plus the plot boundary."""

        # 1) Create figure and axes
        fig, ax = self._create_figure_axes()

        # 2) Draw Plot boundary
        #   We'll just do a rectangle from (0,0) up to (length, width).
        #   The user can also incorporate extra_space in the axes if desired.
        ax.add_patch(
            patches.Rectangle(
                (0, 0),
                self.plot_config.length,
                self.plot_config.width,
                fill=False,
                edgecolor="blue",
                linewidth=1.5,
                label="Plot Boundary",
                zorder=0
            )
        )

        # 3) Draw each structure
        for s in self.structures:
            s.draw(ax)

        # 4) Draw any irrigation lines
        for line in self.irrigation_lines:
            line.draw(ax)

        # 5) Draw any fittings
        for fitting in self.irrigation_fittings:
            fitting.draw(ax)

        # 6) Save main figure
        plt.savefig(filename, format="pdf", bbox_inches="tight")

        # 7) Make separate legend figure
        handles, labels = ax.get_legend_handles_labels()
        unique_dict = dict(zip(labels, handles))
        fig_legend, ax_legend = plt.subplots(figsize=(3, 3))
        ax_legend.axis('off')
        ax_legend.legend(
            unique_dict.values(), unique_dict.keys(),
            loc='center', fontsize=8, frameon=False
        )
        legend_filename = filename.replace(".pdf", "_legend.pdf")
        fig_legend.savefig(legend_filename, format="pdf", bbox_inches="tight")

        plt.close(fig)
        plt.close(fig_legend)

    def _create_figure_axes(self):
        """Helper to create the figure and axes with gridlines, sizing, etc."""
        aspect_ratio = self.plot_config.new_width / self.plot_config.new_length
        fig_height = self.plot_config.figure_size
        fig_width = fig_height / aspect_ratio

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Setup gridlines
        # We'll define xlim/ylim from -5..(length-5) for aesthetics, or adjust as needed
        ax.set_xlim(-5, self.plot_config.new_length - 5)
        ax.set_ylim(-5, self.plot_config.new_width - 5)
        major_x = range(-5, int(self.plot_config.new_length)+1, 5)
        major_y = range(-5, int(self.plot_config.new_width)+1, 5)
        ax.set_xticks(major_x)
        ax.set_yticks(major_y)

        # Minor ticks
        ax.set_xticks(range(-5, int(self.plot_config.new_length)+1), minor=True)
        ax.set_yticks(range(-5, int(self.plot_config.new_width)+1), minor=True)

        ax.grid(which="minor", linestyle=":", linewidth=0.5, color="gray")
        ax.grid(which="major", linestyle="-", linewidth=0.8, color="black")

        ax.set_xlabel("Length (feet)")
        ax.set_ylabel("Width (feet)")
        return fig, ax

###############################################################################
#                          YAML LOADER + MAIN
###############################################################################

def load_layout_from_yaml(path: str) -> Layout:
    """Load the YAML config, validate with pydantic, then return a Layout object."""
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    validated = LayoutConfig(**data)
    return Layout(validated)


if __name__ == "__main__":
    # A minimal demonstration that loads from an embedded YAML string
    # (in real usage, you'd load from an external file).
    import textwrap

    DEMO_YAML = textwrap.dedent("""
    plot:
      width: 20
      length: 50
      extra_space: 5
      figure_size: 10

    structures:
      polygons:
        - name: House
          points: [[0,0],[0,10],[10,10],[10,0]]
          edgecolor: "gray"
          linewidth: 2.0
          needs_irrigation: false

      shaped_structures:
        - name: RaisedBed
          shape: "rectangle"
          alignment: "bottom-left"
          dimensions:
            length: 8
            breadth: 4
          positions:
            - x: 15
              y: 5
              name: "Bed1"
          needs_irrigation: true

    irrigation:
      lines:
        - name: WaterMain
          coordinates: [[-1,2],[2,2],[2,8]]
          color: "blue"
          linewidth: 1.5
      fittings:
        - name: Valve1
          fitting_type: "valve"
          position: [-1,2]
          color: "red"
    """)

    # For demonstration, let's load from this string:
    yaml_file = "/content/layout_config.yaml"
    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)
    # data = yaml.safe_load("/content/layout_config.yaml")
    layout_config = LayoutConfig(**data)
    layout = Layout(layout_config)

    layout.draw("demo_layout.pdf")
    print("Demo layout drawn -> demo_layout.pdf & demo_layout_legend.pdf")

###############################################################################
#                          TESTING EXAMPLE
###############################################################################
def test_polygon_structure_bounds():
    """
    Example of a pytest-style test to check bounding box correctness.
    Run `pytest improved_layout.py` to execute this test.
    """
    
    conf = PolygonConfig(
        name="TestPoly",
        points=[(0,0), (0,5), (5,5), (5,0)]
    )
    poly_struct = PolygonStructure(conf)
    x_min, x_max, y_min, y_max = poly_struct.bounding_box()
    assert x_min == 0
    assert x_max == 5
    assert y_min == 0
    assert y_max == 5


def test_irrigation_line_length():
    """Simple test to confirm line length is computed properly."""
    coords = [(0,0),(3,4)]
    line = IrrigationLine("TestLine", "TestBed", coords, "blue", 1.5, 5)
    assert line.length == 5.0  # 3-4-5 triangle
