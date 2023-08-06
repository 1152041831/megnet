#!/usr/bin/env python
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

"""
A master convenience script with many tools for vasp and structure analysis.
"""

import argparse
import sys
from difflib import SequenceMatcher
from pathlib import Path

from pymatgen.core import Structure
from tabulate import tabulate

from megnet.utils.models import MEGNetModel

# 选取默认的模型路径，选择默认的模型
DEFAULT_MODEL_PATH = Path(__file__).parent / ".." / ".." / "mvl_models" / "mp-2019.4.1"
DEFAULT_MODELS = [str(f) for f in DEFAULT_MODEL_PATH.glob("*.hdf5")]


# 使用命令行输入下的预测方法
def predict(args):
    """
    Handle view commands.

    :param args: Args from command.
    """
    headers = ["Filename"]
    output = []
    models = []
    prefix = ""
    # 读取输入参数args下所有的模型文件，为每个文件构建一个megnet模型并添加进一个models列表中
    for i, mn in enumerate(args.models):
        model = MEGNetModel.from_file(mn)
        models.append(model)
        if i == 0:
            prefix = mn
        else:
            sm = SequenceMatcher(None, prefix, mn)
            match = sm.find_longest_match(0, len(prefix), 0, len(mn))
            prefix = prefix[0 : match.size]
        headers.append(f"{mn} ({model.metadata.get('unit', '').strip('log10')}")
    headers = [h.lstrip(prefix) for h in headers]

    for fn in args.structures:
        structure = Structure.from_file(fn)
        row = [fn]
        for model in models:
            val = model.predict_structure(structure).ravel()
            if "log10" in str(model.metadata.get("unit", "")):
                val = 10**val
            row.append(val)
        output.append(row)
    print(tabulate(output, headers=headers))

# 主函数
# 介绍了meg.py是一个命令行界面，可以使用它来将构建好的模型进行预测
# 输入命令 "meg sub-command -h" 可以查看更多的子命令
def main():
    """
    Handle main.
    """
    parser = argparse.ArgumentParser(
        description="""
    meg is command-line interface to useful MEGNet tasks, e.g., prediction
    using a built model, etc. To see the options for the
    sub-commands, type "meg sub-command -h"."""
    )

    # 创建子命令解析器 subparsers
    subparsers = parser.add_subparsers()

    # 预测方法
    parser_predict = subparsers.add_parser("predict", help="Predict property using MEGNET.")

    # 为预测方法设置需要的参数：structures。 需要输入的类型为str，nargs="+"表示可以接受多个值作为输入
    parser_predict.add_argument(
        "-s", "--structures", dest="structures", type=str, nargs="+", help="Structures to process"
    )

    # 为预测方法设置需要的参数：models。 表示使用哪个模型进行预测，默认的模型输入为 DEFAULT_MODELS
    parser_predict.add_argument(
        "-m", "--models", dest="models", type=str, nargs="+", default=DEFAULT_MODELS, help="Models to run."
    )
    parser_predict.set_defaults(func=predict)

    # 解析命令行参数，将结果存入args对象中。
    args = parser.parse_args()

    # 尝试获取args对象中名为“func”的属性，如果不存在则打印帮助信息并退出；
    # 如果存在，则调用该函数并将args参数对象传递给它。
    try:
        getattr(args, "func")
    except AttributeError:
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    main()
