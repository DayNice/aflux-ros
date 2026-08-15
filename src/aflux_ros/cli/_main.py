from cyclopts import App

from ._bag import app as bag_app

app = App(name="aflux-ros", help="Inspect Robot Operating System (ROS) artifacts.")
app.register_install_completion_command()
app.command(bag_app, name="bag")
