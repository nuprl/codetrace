# syntax=docker/dockerfile:1
   
FROM ubuntu:22.04
WORKDIR /interp
COPY /mechinterp /interp
ENV PATH="/interp/mechinterp/bin/activate"
RUN python -c "import sys; print(sys.version)"
# ENTRYPOINT ["python3", "main.py"]