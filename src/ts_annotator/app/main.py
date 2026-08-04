"""CLI — ``tsa <pasta-do-projeto>`` abre o anotador num workspace (ver workspace.py).

O projeto é uma pasta com ``project.yaml`` + subpastas convencionais
(visualizations/, layers/, annotations/, models/, predictions/). Exemplo mínimo
de ``project.yaml`` em docs/WORKSPACE.md.
"""
from __future__ import annotations

import argparse
import logging
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="tsa",
        description="TSA — anotador map-native de séries temporais de satélite")
    ap.add_argument("project", nargs="?",
                    help="pasta do projeto (contém project.yaml). Se omitido, abre o "
                         "wizard de novo projeto.")
    ap.add_argument("--new", action="store_true",
                    help="força o wizard de novo projeto (ignora 'project')")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from PyQt6.QtWidgets import QApplication, QDialog

    from ts_annotator.app.theme import apply_theme
    from ts_annotator.app.workspace import WorkspaceError, load_workspace
    from ts_annotator.ui.annotator_window import AnnotatorWindow

    app = QApplication(sys.argv[:1])
    apply_theme(app)

    project = args.project
    if args.new or not project:
        from ts_annotator.ui.new_project import NewProjectDialog
        dlg = NewProjectDialog(start_dir=project)
        if dlg.exec() != QDialog.DialogCode.Accepted or not dlg.project_dir:
            return 0   # usuário cancelou o wizard
        project = dlg.project_dir

    try:
        ctx = load_workspace(project)
    except WorkspaceError as e:
        print(f"erro no workspace: {e}", file=sys.stderr)
        return 2
    win = AnnotatorWindow(ctx)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
