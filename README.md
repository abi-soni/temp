Install Docker Engine on Ubuntu
To get started with Docker Engine on Ubuntu, make sure you meet the prerequisites, and then follow the installation steps.

Prerequisites
OS requirements
To install Docker Engine, you need the 64-bit version of one of these Ubuntu versions:

Ubuntu Lunar 23.04
Ubuntu Kinetic 22.10
Ubuntu Jammy 22.04 (LTS)
Ubuntu Focal 20.04 (LTS)
Ubuntu Bionic 18.04 (LTS)
Docker Engine is compatible with x86_64 (or amd64), armhf, arm64, and s390x architectures.

Uninstall old versions
Older versions of Docker went by the names of docker, docker.io, or docker-engine, you might also have installations of containerd or runc. Uninstall any such older versions before attempting to install a new version:


 sudo apt-get remove docker docker-engine docker.io containerd runc
apt-get might report that you have none of these packages installed.

Images, containers, volumes, and networks stored in /var/lib/docker/ aren’t automatically removed when you uninstall Docker. If you want to start with a clean installation, and prefer to clean up any existing data, read the uninstall Docker Engine section.

Installation methods
You can install Docker Engine in different ways, depending on your needs:

Docker Engine comes bundled with Docker Desktop for Linux. This is the easiest and quickest way to get started.

Set up and install Docker Engine from Docker’s apt repository.

Install it manually and manage upgrades manually.

Use a convenience scripts. Only recommended for testing and development environments.

Install using the apt repository
Before you install Docker Engine for the first time on a new host machine, you need to set up the Docker repository. Afterward, you can install and update Docker from the repository.

Set up the repository
Update the apt package index and install packages to allow apt to use a repository over HTTPS:


 sudo apt-get update
 sudo apt-get install ca-certificates curl gnupg
Add Docker’s official GPG key:


 sudo install -m 0755 -d /etc/apt/keyrings
 curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
 sudo chmod a+r /etc/apt/keyrings/docker.gpg
Use the following command to set up the repository:


 echo \
  "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
Note

If you use an Ubuntu derivative distro, such as Linux Mint, you may need to use UBUNTU_CODENAME instead of VERSION_CODENAME.

Install Docker Engine
Update the apt package index:


 sudo apt-get update
Install Docker Engine, containerd, and Docker Compose.

Latest
Specific version

To install the latest version, run:


 sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
Verify that the Docker Engine installation is successful by running the hello-world image.


 sudo docker run hello-world
This command downloads a test image and runs it in a container. When the container runs, it prints a confirmation message and exits.

You have now successfully installed and started Docker Engine.

Tip

Receiving errors when trying to run without root?

The docker user group exists but contains no users, which is why you’re required to use sudo to run Docker commands. Continue to Linux postinstall to allow non-privileged users to run Docker commands and for other optional configuration steps.

Upgrade Docker Engine
To upgrade Docker Engine, follow the installation instructions, choosing the new version you want to install.

Install from a package
If you can’t use Docker’s apt repository to install Docker Engine, you can download the deb file for your release and install it manually. You need to download a new file each time you want to upgrade Docker Engine.

Go to https://download.docker.com/linux/ubuntu/dists/.

Select your Ubuntu version in the list.

Go to pool/stable/ and select the applicable architecture (amd64, armhf, arm64, or s390x).

Download the following deb files for the Docker Engine, CLI, containerd, and Docker Compose packages:

containerd.io_<version>_<arch>.deb
docker-ce_<version>_<arch>.deb
docker-ce-cli_<version>_<arch>.deb
docker-buildx-plugin_<version>_<arch>.deb
docker-compose-plugin_<version>_<arch>.deb
Install the .deb packages. Update the paths in the following example to where you downloaded the Docker packages.


 sudo dpkg -i ./containerd.io_<version>_<arch>.deb \
  ./docker-ce_<version>_<arch>.deb \
  ./docker-ce-cli_<version>_<arch>.deb \
  ./docker-buildx-plugin_<version>_<arch>.deb \
  ./docker-compose-plugin_<version>_<arch>.deb
The Docker daemon starts automatically.

Verify that the Docker Engine installation is successful by running the hello-world image.


 sudo service docker start
 sudo docker run hello-world
This command downloads a test image and runs it in a container. When the container runs, it prints a confirmation message and exits.

You have now successfully installed and started Docker Engine.
