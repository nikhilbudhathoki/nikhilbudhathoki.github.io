# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /code

# Copy the current directory contents into the container at /app
COPY . /code
COPY . /requirements.txt ./

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY ./src ./src

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Run Streamlit app when the container launches
CMD ["streamlit", "run", "src/main.py"]
