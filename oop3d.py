# guess im writing oop lol
from netgen.csg import *
import trimesh


class WatertightThing:
    def __init__(self, name, volume, mass):
        raise NotImplementedError("watertight thing is abstract")

    def __add__(self, other):
        raise NotImplementedError("watertight thing is abstract")

    def __sub__(self, other):
        raise NotImplementedError("watertight thing is abstract")

    def __mul__(self, other):
        raise NotImplementedError("watertight thing is abstract")

    def to_stl(self, filename):
        raise NotImplementedError("watertight thing is abstract")

    def is_watertight(self):
        raise NotImplementedError("watertight thing is abstract")


class CSGThing(WatertightThing):
    def __init__(self, thing):
        self.thing = thing

    def __add__(self, other):
        assert isinstance(other, CSGThing)
        return CSGThing(self.thing + other.thing)

    def __sub__(self, other):
        assert isinstance(other, CSGThing)
        return CSGThing(self.thing - other.thing)

    def __mul__(self, other):
        assert isinstance(other, CSGThing)
        return CSGThing(self.thing * other.thing)

    def to_stl(self, filename):
        geo = CSGeometry()
        geo.Add(self.thing)
        mesh = geo.GenerateMesh(maxh=10)
        mesh.Export(filename, "STL Format")

    def is_watertight(self):
        return True
        return self.thing.IsWatertight()


class TrimeshThing(WatertightThing):
    def __init__(self, trimesh_mesh):
        self.thing = trimesh_mesh

    def __add__(self, other):
        assert isinstance(other, TrimeshThing)
        return TrimeshThing(self.thing.union(other.thing))

    def __sub__(self, other):
        assert isinstance(other, TrimeshThing)
        return TrimeshThing(self.thing.difference(other.thing))

    def __mul__(self, other):
        assert isinstance(other, TrimeshThing)
        return TrimeshThing(self.thing.intersection(other.thing))

    def to_stl(self, filename):
        self.thing.export(filename)

    def is_watertight(self):
        return self.thing.is_watertight
