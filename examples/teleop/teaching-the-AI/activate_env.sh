# Set up a terminal for this project.  Use it like this, once per terminal:
#
#     source activate_env.sh
#
# (`source`, not `python`, and not `./activate_env.sh` - it has to change THIS
#  terminal, and a normal script only changes its own.)
#
# Two things happen. The first is ordinary: switch on the `kinova` conda
# environment, which is where all the libraries live.
#
# The second is the fix for a genuinely confusing error. Some robot libraries
# (pinocchio, and the maths behind the arm controller) were built against a
# newer system library than Ubuntu ships. Without the line below you get:
#
#     ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `CXXABI_1.3.15' not found
#
# which looks like a broken install but is only a matter of which folder the
# computer searches first. LD_LIBRARY_PATH puts conda's copy at the front.

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate kinova
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# So `import robot_ai` works no matter which folder you run python from.
export PYTHONPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd ):$PYTHONPATH"

echo "environment ready:  $CONDA_DEFAULT_ENV"
echo "next:               python check_setup.py"
