"""MDSmooth ChimeraX bundle: filter MD-trajectory RMSD to build clearer morph movies."""

from chimerax.core.toolshed import BundleAPI


class _MDSmoothAPI(BundleAPI):

    api_version = 1

    @staticmethod
    def register_command(bi, ci, logger):
        # Called once per command listed in bundle_info.xml the first time it
        # is used. Registers the command with ChimeraX.
        from chimerax.core.commands import register
        from . import cmd

        if ci.name == "mdsmooth":
            register(ci.name, cmd.mdsmooth_desc, cmd.mdsmooth, logger=logger)
        elif ci.name == "mdsmooth installLearnedCV":
            register(ci.name, cmd.install_learned_cv_desc,
                     cmd.install_learned_cv, logger=logger)

    @staticmethod
    def start_tool(session, bi, ti):
        # Called when the user launches the tool from the Tools menu.
        from .tool import MDSmoothTool

        if ti.name == "MDSmooth":
            return MDSmoothTool(session, ti.name)


bundle_api = _MDSmoothAPI()
