#   -*- coding: utf-8 -*-
from pybuilder.core import use_plugin, init, Author

use_plugin("python.core")
use_plugin("python.unittest")
use_plugin("python.flake8")
use_plugin("python.coverage")
use_plugin("python.distutils")

name = "mutual-information"
authors = [Author('Xingmin Aaron Zhang', 'kingmanzhang@gmail.com')]
description = 'mutual information-based synergy between variables for one response'
with open('./README.md', 'r') as f:
    long_description = f.read()
long_description_content_type = 'text/markdown'
url = 'https://github.com/kingmanzhang/mutual-information'
version = '0.0.4'
license = 'MIT'

default_task = "publish"

@init
def set_properties(project):
    project.depends_on("numpy")
    project.depends_on("pandas")
    project.depends_on("networkx")
    project.depends_on("treelib")

    project.set_property("coverage_threshold_warn", 50)
    project.set_property("coverage_break_build", False)

    project.get_property("distutils_commands").append("bdist_wheel")
