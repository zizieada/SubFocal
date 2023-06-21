FROM tensorflow/tensorflow:2.5.1-gpu

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
