# rtac_base

This package contains various types / algorithms which are extensively used in
the RTAC projects developments (namely the rtac_display, rtac_optix and
rtac_acoustics repositories).

Disclaimer : this packages and the dependent ones have only been tested on
Ubuntu. Don't expect to see it work right away with another system.

# Installation

### Dependencies

The two main dependencies are Eigen3 and CUDA (you can skip if these are already
installed on your system).

#### Installing Eigen

The Eigen version being provided in the Ubuntu package manager being rather old,
this software uses more recent version of Eigen provided by Eigen [official
repository](https://gitlab.com/libeigen/eigen).

```
git clone https://gitlab.com/libeigen/eigen.git
cd eigen
make build && cd build
cmake -DCMAKE_INSTALL_PREFIX=<your install location> ..
make install
```

#### Installing CUDA on Ubuntu

To our knowledge, when these line were written there are still no consensus on
how to install CUDA on your machine while being sure not to break everything. We
present here how we do it, which takes a little more efforts but has the benefit
of not breaking you graphical session. The idea is to let Ubuntu manage the
graphical driver, and install CUDA by manually selecting the CUDA version
depending on which graphic driver was/is installed.

**CAUTION** Installing graphical drivers has the potential to break your
graphical session. If this happens, you can save yourself by logging into a TTY
(CTRL-ALT-F[1-2-3-4]), removing all that is nvidia related using apt, and
deleting the /etc/X11/xorg.conf file. This will re-enable the use of the mesa
driver and give you back a graphical session. (But use at your own risks...).


1. Check your graphic card model using lshw and install a suitable driver.

```
lshw | grep display -A10 -B10
```
Find the model of your NVIDIA graphic card and search on the [NVIDIA
website](https://www.nvidia.com/Download/index.aspx) the driver version suitable
for your graphic card **DO NOT DOWNLOAD THE DRIVER, JUST CHECK THE VERSION**

Instead, install the most suitable version for your system from the **ADDITIONAL
DRIVER** dialog in Ubuntu. Install it and reboot.



2. Check the CUDA version and install the highest one compatible with your
driver.

Check the CUDA version you can install using the Table II in [CUDA release
notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html).

Download the CUDA installation file from
[here](https://developer.nvidia.com/cuda-downloads).

**TAKE THE RUNFILE, NOT A .DEB. THE .DEB MIGHT INSTALL AN INCOMPATIBLE DRIVER**

Run the runfile as sudo but **DO NOT INSTALL THE DRIVER** (You already did
that).

Let the installer do his thing, and voilà ! CUDA should be installed under
/usr/local





