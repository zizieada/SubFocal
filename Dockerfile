FROM tensorflow/tensorflow:2.5.1-gpu

RUN NVIDIA_GPGKEY_FPR=ae09fe4bbd223a84b2ccfce3f60f4b3d7fa2af80 && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 1E9377A2BA9EF27F && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 605C66F00D6C9793 && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 648ACFD622F3D138 && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 0E98404D386FA1D9 && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys BA6932366A755776
    
RUN apt-get update && apt-get install -y --no-install-recommends --allow-unauthenticated \
        imagemagick \
&& apt clean \
&& apt-get autoremove -y \
&& apt-get autoclean -y \
&& rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN \
  python -m pip --no-cache-dir install --upgrade \
          pandas \
          numpy \
          scipy \
          scikit-image \
          scikit-learn \
          imageio \
          tensorboard \
          tensorflow-addons \
          pillow
