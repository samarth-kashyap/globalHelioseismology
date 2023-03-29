import os

current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(os.path.dirname(current_dir))
with open(f"{package_dir}/.config", "r") as f:
    dirnames = f.readlines()[0].split("\n")

class dirConfig():
    def __init__(self):
        self.package_dir = package_dir
        self.data_dir = dirnames[0]
        self.mode_dir = dirnames[1]
        self.output_dir = dirnames[2]
        
