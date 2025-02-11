FROM debian:10

WORKDIR /home
ENV HOME=/home
RUN cd ~
RUN apt-get update
RUN apt-get install -y git nano python3-pip python3-dev pkg-config wget usbutils curl \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libopencv-dev

RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
| tee /etc/apt/sources.list.d/coral-edgetpu.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
RUN apt-get update
RUN apt-get install -y edgetpu-examples udev sudo

RUN echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="18d1", ATTR{idProduct}=="9302", MODE="0666"' > /etc/udev/rules.d/CORALUSB

# Installer OpenSSH Client
RUN apt-get install -y openssh-client

# CrÃ©er le rÃ©pertoire SSH et copier les clÃ©s SSH de la machine hÃ´te
RUN mkdir -p /root/.ssh && chmod 0700 /root/.ssh
COPY --chmod=600 id_rsa /root/.ssh/id_rsa
COPY --chmod=644 id_rsa.pub /root/.ssh/id_rsa.pub

# Ajouter GitHub aux hÃ´tes connus
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts

# Cloner le dÃ©pÃ´t GitHub
RUN git clone git@github.com:KentoPAC/JuniaProjetTennis.git /home/JuniaProjetTennis

# Copier le fichier de dÃ©pendances Python
COPY requirements.txt /home/requirements.txt

# Installer les dÃ©pendances Python, y compris TensorFlow
RUN pip3 install --no-cache-dir --upgrade pip \
    && pip3 install --no-cache-dir -r /home/requirements.txt tensorflow

# DÃ©finir le rÃ©pertoire de travail et copier le projet
WORKDIR /home/JuniaProjetTennis
COPY . /home/JuniaProjetTennis

# Lancer le script automatiquement
CMD ["python3", "classify_coral3.py"]