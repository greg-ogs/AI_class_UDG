FROM tensorflow/tensorflow:nightly-gpu
# (Optional) Install any additional Python packages you need
RUN pip install matplotlib tensorflow-hub pandas
RUN pip install diffusers
RUN pip install transformers
RUN pip install accelerate
RUN pip install torchvision
RUN pip install torchaudio
RUN pip install sentencepiece

# (Optional) Set working directory
WORKDIR /app