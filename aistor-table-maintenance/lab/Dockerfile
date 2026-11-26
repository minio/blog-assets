FROM python:3.12-slim
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_ROOT_USER_ACTION=ignore PYTHONUNBUFFERED=1
RUN pip install --no-cache-dir \
      "pyiceberg[pyarrow]==0.10.0" \
      duckdb==1.5.5 \
      pandas==2.3.3 \
      boto3==1.40.70 \
      jupyterlab==4.4.10 \
      matplotlib==3.10.7
