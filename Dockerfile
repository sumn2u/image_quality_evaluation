FROM  continuumio/miniconda3
LABEL Author, Suman Kunwar

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . $APP_HOME

#---------------- Prepare the environment
RUN conda update --name base conda &&\
    conda env create --file environment.yaml &&\
    conda install libavif

RUN pip install imageio[pyav]

EXPOSE 5000

SHELL ["conda", "run", "--name", "app", "/bin/bash", "-c"]

ENTRYPOINT ["conda", "run", "--name", "app", "python", "main.py"]
