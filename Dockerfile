# Dockerfile

# Use an official Python 3.12 runtime as a parent image
FROM python:3.12-slim-bookworm

# Install uv installer requirements: curl and certificates
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Download and install uv
ADD https://astral.sh/uv/install.sh uv-installer.sh
RUN sh uv-installer.sh && rm uv-installer.sh
ENV PATH="/root/.local/bin:$PATH"

# Set the working directory
WORKDIR /code

# This tells Python to look for modules in the /code directory (e.g., helper_lib)
ENV PYTHONPATH="/code"

# Copy the pyproject.toml file
COPY pyproject.toml ./

# Install dependencies using uv
RUN uv pip install --system -r pyproject.toml

# Copy the application code and the trained generator model
COPY ./app ./app
COPY ./helper_lib ./helper_lib
COPY ./gan_generator.pth ./

# Expose the port the app runs on
EXPOSE 8000

# Run uvicorn server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]