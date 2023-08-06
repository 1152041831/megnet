"""
Crystal graph related
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
from monty.serialization import loadfn
from pymatgen.analysis.local_env import NearNeighbors
from pymatgen.core import Element, Structure

from megnet.data.graph import Converter, StructureGraph, StructureGraphFixedRadius

# 当前脚本文件所在目录的绝对路径 MODULE_DIR
MODULE_DIR = Path(__file__).parent.absolute()

# 获取预训练的元素嵌入elemental embeddings。返回一个字典，包含了元素符号和对应的长度为16的字符串表示。
def get_elemental_embeddings() -> dict:
    """
    Provides the pre-trained elemental embeddings using formation energies,
    which can be used to speed up the training of other models. The embeddings
    are also extremely useful elemental descriptors that encode chemical
    similarity that may be used in other ways. See

    "Graph Networks as a Universal Machine Learning Framework for Molecules
    and Crystals", https://arxiv.org/abs/1812.05055

    :return: dict of elemental embeddings as {symbol: length 16 string}
    """
    return loadfn(MODULE_DIR / "resources" / "elemental_embedding_1MEGNet_layer.json")

# 将晶体转换成以 z为原子特征，以距离为键（bond）特征的 图 ，可以选择包括状态（state）特征
class CrystalGraph(StructureGraphFixedRadius):
    """
    Convert a crystal into a graph with z as atomic feature and distance as bond feature
    one can optionally include state features
    """

    """
    nn_strategy：近邻策略的名称或近邻对象。默认值是 "MinimumDistanceNNAll"，表示使用最小距离近邻策略。
    atom_converter：原子特征转换器的对象。默认值为None，表示不使用原子特征转换器。
    bond_converter：化学键特征转换器的对象。默认值为None，表示不使用化学键特征转换器。
    cutoff：截断半径（cutoff radius），表示判断化学键是否存在的距离阈值。默认值为5.0。
    """

    def __init__(
        self,
        nn_strategy: str | NearNeighbors = "MinimumDistanceNNAll",
        atom_converter: Converter | None = None,
        bond_converter: Converter | None = None,
        cutoff: float = 5.0,
    ):
        """
        Convert the structure into crystal graph
        Args:
            nn_strategy (str): NearNeighbor strategy
            atom_converter (Converter): atom features converter
            bond_converter (Converter): bond features converter
            cutoff (float): cutoff radius
        """
        self.cutoff = cutoff
        # 调用父类StructureGraphFixedRadius的构造函数（使用super()函数），传递给父类的参数：近邻策略、原子特征转换器、化学键特征转换器和截断半径。
        # 通过这样的方式，CrystalGraph类实例化时会自动调用父类的构造函数来完成初始化工作。
        super().__init__(
            nn_strategy=nn_strategy, atom_converter=atom_converter, bond_converter=bond_converter, cutoff=self.cutoff
        )

# CrystalGraphWithBondTypes类继承自StructureGraph类，用于将晶体转换为一个 图 。
# 在这个图中，化学键的属性被覆盖为键的类型，该类型基于形成键的原子的金属性。
class CrystalGraphWithBondTypes(StructureGraph):
    """
    Overwrite the bond attributes with bond types, defined simply by
    the metallicity of the atoms forming the bond. Three types of
    scenario is considered, nonmetal-nonmetal (type 0), metal-nonmetal (type 1), and
    metal-metal (type 2)

    """

    def __init__(
        self,
        # nn_strategy：近邻策略的名称或近邻对象。默认值是"VoronoiNN"，表示使用 Voronoi 近邻策略。
        nn_strategy: str | NearNeighbors = "VoronoiNN",
        atom_converter: Converter | None = None,
        bond_converter: Converter | None = None,
    ):
        """

        Args:
            nn_strategy (str): NearNeighbor strategy
            atom_converter (Converter): atom features converter
            bond_converter (Converter): bond features converter
        """
        super().__init__(nn_strategy=nn_strategy, atom_converter=atom_converter, bond_converter=bond_converter)

    # 将 结构 转化为 图 的方法。输入的参数为：由pymatgen得到的Structure类型的数据；状态属性state_attributes
    def convert(self, structure: Structure, state_attributes: list | None = None) -> dict:
        """
        Convert structure into graph
        Args:
            structure (Structure): pymatgen Structure
            state_attributes (list): state attributes

        Returns: graph dictionary

        """
        graph = super().convert(structure, state_attributes=state_attributes)
        return self._get_bond_type(graph)

    # 获取键类型，输入的图是字典类型的
    @staticmethod
    def _get_bond_type(graph) -> dict:
        new_graph = deepcopy(graph)
        # 通过图中的原子序数获取创建一个对应元素的对象lement.from_Z(i)，存入elements中
        # "element": 化学物种的元素符号。
        # "oxidation_state": 化学物种的氧化态。
        elements = [Element.from_Z(i) for i in graph["atom"]]
        # 通过金属性确定类型
        # 这里使用的是元素对象的 is_metal 属性，该属性返回一个布尔值，表示原子是否为金属。
        # 如果两个元素都是金属，值为 2 （metal-metal (type 2)）；
        # 如果只有一个元素是金属，值为 1 （metal-nonmetal (type 1)）；
        # 如果两个元素都不是金属，值为 0 （nonmetal-nonmetal (type 0)）。
        # k为索引，(i, j)是graph["index1"], graph["index2"]配对的键值对。
        for k, (i, j) in enumerate(zip(graph["index1"], graph["index2"])):
            new_graph["bond"][k] = elements[i].is_metal + elements[j].is_metal
        return new_graph


class _AtomEmbeddingMap(Converter):
    """
    Fixed Atom embedding map, used with CrystalGraphDisordered
    """

    def __init__(self, embedding_dict: dict | None = None):
        """
        Args:
            embedding_dict (dict): element to element vector dictionary
        """
        # 获取元素嵌入字典。如果为空则从方法get_elemental_embeddings()中获取
        if embedding_dict is None:
            embedding_dict = get_elemental_embeddings()
        self.embedding_dict = embedding_dict

    # 用于将原子的序列转换为数值特征表示。
    # 它遍历每个原子，根据原子的符号和分数计算原子的嵌入特征（使用元素向量乘以分数并相加），并将结果添加到特征列表中。
    # 最后，返回表示所有原子特征的 NumPy 数组。
    def convert(self, atoms: list) -> np.ndarray:
        """
        Convert atom {symbol: fraction} list to numeric features
        """
        features = []
        for atom in atoms:
            emb = 0
            for k, v in atom.items():
                emb += np.array(self.embedding_dict[k]) * v
            features.append(emb)
        return np.array(features).reshape((len(atoms), -1))

# 定义了一个用于处理具有非确定性位置预测的晶体图的子类
class CrystalGraphDisordered(StructureGraphFixedRadius):
    """
    Enable disordered site predictions
    """

    def __init__(
        self,
        nn_strategy: str | NearNeighbors = "MinimumDistanceNNAll",
        atom_converter: Converter = _AtomEmbeddingMap(),
        bond_converter: Converter | None = None,
        cutoff: float = 5.0,
    ):
        """
        Convert the structure into crystal graph
        Args:
            nn_strategy (str): NearNeighbor strategy
            atom_converter (Converter): atom features converter
            bond_converter (Converter): bond features converter
            cutoff (float): cutoff radius
        """
        self.cutoff = cutoff
        super().__init__(
            nn_strategy=nn_strategy, atom_converter=atom_converter, bond_converter=bond_converter, cutoff=self.cutoff
        )

    # 用于返回包含元素分数描述的列表，该列表表示具有潜在位置混乱的晶体结构中的每个位置的占据情况。
    # 例如将 Fe0.5Ni0.5 转化为 {"Fe": 0.5, "Ni": 0.5}
    @staticmethod
    def get_atom_features(structure) -> list[dict]:
        """
        For a structure return the list of dictionary for the site occupancy
        for example, Fe0.5Ni0.5 site will be returned as {"Fe": 0.5, "Ni": 0.5}

        Args:
            structure (Structure): pymatgen Structure with potential site disorder

        Returns:
            a list of site fraction description
        """

        # Structure 类中的 sites 是一个属性，它返回一个列表，包含了晶体结构中每个位置的信息。
        # .species 是一个属性访问，用于获取该位置的化学物种（Species）信息。
        # i.species 返回的是一个 Species 对象，它包含了化学物种的元素符号和氧化态等信息
        # （"element": 化学物种的元素符号。"oxidation_state": 化学物种的氧化态。）
        # 使用.as_dict()转化为字典格式。例如 Species("Li", 1)，调用 .as_dict() 方法将返回 {"element": "Li", "oxidation_state": 1}。
        return [i.species.as_dict() for i in structure.sites]
